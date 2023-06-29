from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import pdb
import numpy as np
import numpy
import sys
import collections
import copy
import time
from datetime import timedelta
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F        

import pdb

from sklearn.cluster import KMeans

from maskcl import datasets
from maskcl import models
from maskcl.models.hm import HybridMemory
from maskcl.models.embeddingmodel import Embedding_model
from maskcl.trainers import CACLTrainer_USL,CACLSIC_USL, Mask_CACLSIC_USL
from maskcl.evaluators import Evaluator, extract_features
from maskcl.utils.data import IterLoader
from maskcl.utils.data import transforms as T
from maskcl.utils.data.sampler import RandomMultipleGallerySampler
from maskcl.utils.data.preprocessor import Preprocessor
from maskcl.utils.logging import Logger
from maskcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from maskcl.utils.faiss_rerank import compute_jaccard_distance


import os


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])
    
    train_transformer2 = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.Grayscale(num_output_channels=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
        ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform1=train_transformer,transform2 = train_transformer2),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader



def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    train = True
    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))
        train = False

    test_loader = DataLoader(
        Preprocessor(testset, train, root=dataset.images_dir, transform1=test_transformer,transform2 = test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader





def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)
    return model

def evaluate_mean(evaluator1, dataset, test_loaders):
    maxap = 0
    maxcmc = 0
    mAP_sum = 0
    cmc_sum = 0
    cmc_sum_10 = 0

    for i in range(len(dataset)):
        cmc_scores, mAP = evaluator1.evaluate(test_loaders[i], dataset[i].query, dataset[i].gallery, cmc_flag=False)
        maxap = max(mAP, maxap)
        maxcmc = max(cmc_scores[0], maxcmc)
        mAP_sum += mAP
        cmc_sum += cmc_scores[0]
        cmc_sum_10 += cmc_scores[9]

    mAP = (mAP_sum) / len(test_loaders)
    cmc_now = (cmc_sum) / len(test_loaders)
    cmc_now_10 = cmc_sum_10 / (len(test_loaders))

    return mAP, cmc_now, cmc_now_10

def cluster_finement_new(index2label, pseudo_labels, rerank_dist, pseudo_labels_tight):
    source_before = (index2label == 1).sum()
    num_label = len(pseudo_labels)
    rerank_dist_tensor = torch.tensor(rerank_dist)
    N = pseudo_labels.size(0)
    
    label_sim_expand = pseudo_labels.expand(N, N)
    label_sim_tight_expand = pseudo_labels_tight.expand(N, N)
    
    label_sim = label_sim_expand.eq(label_sim_expand.t()).float()
    label_sim_tight = label_sim_tight_expand.eq(label_sim_tight_expand.t()).float()
    
    sim_distance = rerank_dist_tensor.clone() * label_sim
    dists_labels = label_sim.sum(-1)
    
    dists_label_add = dists_labels.clone()
    dists_label_add[dists_label_add > 1] -= 1
    
    sim_add_average = sim_distance.sum(-1) / torch.pow(dists_labels, 2)
    
    cluster_I_average = torch.zeros(torch.max(pseudo_labels).item() + 1)
    for sim_dists, label in zip(sim_add_average, pseudo_labels):
        cluster_I_average[label.item()] += sim_dists
    
    sim_tight = label_sim.eq(1 - label_sim_tight.clone()).float()
    dists_tight = sim_tight * rerank_dist_tensor.clone()
    
    dists_label_tight_add = (1 + sim_tight.sum(-1))
    dists_label_tight_add[dists_label_tight_add > 1] -= 1
    
    sim_add_average = dists_tight.sum(-1) / torch.pow(dists_label_tight_add, 2)
    
    cluster_tight_average = torch.zeros(torch.max(pseudo_labels_tight).item() + 1)
    for sim_dists, label in zip(sim_add_average, pseudo_labels_tight):
        cluster_tight_average[label.item()] += sim_dists
    
    cluster_final_average = torch.zeros(len(sim_add_average))
    for i, label_tight in enumerate(pseudo_labels_tight):
        cluster_final_average[i] = cluster_tight_average[label_tight.item()]
    
    return cluster_final_average, cluster_I_average



def main():    
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)
    




def main_worker(args):
    global start_epoch, best_mAP
    best_mAP =0
    best_rank1 = 0
    start_time = time.monotonic()
    cudnn.benchmark = True
    
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset, 'maskcl.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    args.data_dir = '/root/pxu1/datasets/{}_all'.format(args.dataset)
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    
    if args.dataset == 'prcc':
        datasets_prcc = []
        test_loaders_prcc = []

        for _ in range(10):
            dataset_cur = get_data(args.dataset, args.data_dir)
            test_loader_cur = get_test_loader(dataset_cur, args.height, args.width, args.batch_size, args.workers)
            datasets_prcc.append(dataset_cur)
            test_loaders_prcc.append(test_loader_cur)
    
    
   
    
    model_rgb = create_model(args)
    model_mask = create_model(args)
    model_fusion = Embedding_model(model_rgb.module.num_features)
    model_fusion.cuda()
    model_fusion = torch.nn.DataParallel(model_fusion)
    
    evaluator1 = Evaluator(model_rgb)
    
    
    memory_rgb = HybridMemory(model_rgb.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()
    memory_mask = HybridMemory(model_mask.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()
    memory_fusion = HybridMemory(model_fusion.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()
    
    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                    args.batch_size, args.workers, testset=sorted(dataset.train))

    features, _, _ = extract_features(model_rgb, cluster_loader, print_freq=50)
    features = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in sorted(dataset.train)], 0)
    _, features2, _= extract_features(model_mask, cluster_loader, print_freq=50)
    features2 = torch.cat([features2[f].unsqueeze(0) for f, _, _, _ in sorted(dataset.train)], 0)
    
    
    memory_rgb.features = F.normalize(features, dim=1).cuda()
    memory_mask.features = F.normalize(features2, dim=1).cuda()
    memory_fusion.features = F.normalize(features, dim=1).cuda()
    
    del cluster_loader, features, features2
    
    

        
    params = []
    print('prepare parameter')
    models = [model_rgb, model_mask, model_fusion]
    for model in models:
        for key, value in model.named_parameters():
            if value.requires_grad:
                params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


    # Trainer
    print('==> Start training')
    trainer = Mask_CACLSIC_USL(model_rgb, model_mask, model_fusion, memory_rgb, memory_mask, memory_fusion)
    start_score = 0
    
    
    for epoch in range(args.epochs):
        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        features = memory_rgb.features.clone()
        now_time_before_cluster =  time.monotonic()
        rerank_dist = compute_jaccard_distance(features.cpu(), k1=args.k1, k2=args.k2)
        del features
        
        if (epoch==0):
            params = {
                        'eps': args.eps,
                        'eps_tight': args.eps - args.eps_gap,
                        'eps_loose': args.eps + args.eps_gap,
                        'min_samples': 4,
                        'metric': 'precomputed',
                        'n_jobs': -1
                    }
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(params['eps'], params['eps_tight'], params['eps_loose']))
            cluster = DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric=params['metric'], n_jobs=params['n_jobs'])
            cluster_tight = DBSCAN(eps=params['eps_tight'], min_samples=params['min_samples'], metric=params['metric'], n_jobs=params['n_jobs'])
            cluster_loose = DBSCAN(eps=params['eps_loose'], min_samples=params['min_samples'], metric=params['metric'], n_jobs=params['n_jobs'])
            
            
        def generate_pseudo_labels(cluster_id, num):
            labels = []
            outliers = 0
            for i, ((fname, _, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                label = id if id != -1 else num + outliers
                labels.append(label)
                if id == -1:
                    outliers += 1
            return torch.Tensor(labels).long()

        
        pseudo_labels = cluster.fit_predict(rerank_dist)
        pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
        pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
        pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)        
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        N = pseudo_labels.size(0)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances\n'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum()))
        
        
    
            
        pseudo_labels_old = pseudo_labels.clone()
        index2label_old = index2label.copy()
        # =====================================================
        if args.cr: 
            print('cluster refinement')
            pseudo_labeled_dataset = []
            outliers = 0
            cluster_final_averge, cluster_I_average = cluster_finement_new(index2label, pseudo_labels, rerank_dist,pseudo_labels_tight )
            for i, ((fname, _, clothes_id, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
                D_score = cluster_final_averge[i]
                if  args.ratio_cluster * D_score.item() <= cluster_I_average[label.item()]:
                    pseudo_labeled_dataset.append((fname,label.item(),clothes_id, cid))
                else:
                    pseudo_labeled_dataset.append((fname,len(cluster_I_average)+outliers,clothes_id, cid))
                    pseudo_labels[i] = len(cluster_I_average)+outliers
                    outliers+=1
        #  =====================================================
            now_time_after_cluster =  time.monotonic()
            print(
                'the time of cluster refinement is {}'.format(now_time_before_cluster-now_time_after_cluster)
            )            
        else:
            print('No cluster refinement')
            pseudo_labeled_dataset = []
            for i, ((fname, pid, clothes_id, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
                pseudo_labeled_dataset.append((fname,label.item(),clothes_id, cid))
                
        
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label.item()]+=1
        index2label = np.fromiter(index2label.values(), dtype=float)
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances\n'
                    .format(epoch, (index2label>1).sum(), (index2label==1).sum()))
        label_count = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
        label_count = label_count.sum(-1)
        
               
        # # KNN find the dist:
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label.item() == -1:
                    continue
                centers[labels[i].item()].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)
            return centers
        
        #基于灰度的聚类中心
        def cosinematrix(A, p=2, dim=1):
            if A.dim() != 2:
                raise ValueError("Input matrix must be 2-dimensional.")
            norm = torch.norm(A, p=p, dim=dim).unsqueeze(dim)
            prod = torch.mm(A, A.t())
            cos = prod.div(torch.mm(norm.t(), norm))
            return cos
        
        
        faetures4 = memory_fusion.features.clone()
        center_feature = generate_cluster_features(pseudo_labels, faetures4)
        del faetures4
        
        center_feature = center_feature.cpu()
        center_feature = cosinematrix(center_feature)
        center_feature_men = center_feature.sort(descending=True)[0]
        center_feature = center_feature.sort(descending=True)[1]
        
        knn_epoch = int((args.allknn/args.epochs) * epoch)
        print(knn_epoch)

        
        
        
        if knn_epoch > 0:
            memory_center_sort_index = torch.zeros(N, knn_epoch+2)
            memory_center_sort_similar = torch.zeros(N, knn_epoch+2)
            pse_set = list(set(list(pseudo_labels.numpy())))
            for index in range(len(label_count)):
                memory_center_sort_index[index] = center_feature[pse_set.index(pseudo_labels[index].item())][0:knn_epoch+2]
                memory_center_sort_similar[index] = center_feature_men[pse_set.index(pseudo_labels[index].item())][0:knn_epoch+2]       
                
            
            memory_fusion.feature_sort = memory_center_sort_index.clone().cuda()
            memory_fusion.knn_epoch = torch.tensor(knn_epoch)
            memory_fusion.center_feature_men = memory_center_sort_similar.clone().cuda()
            
            memory_rgb.feature_sort = memory_center_sort_index.clone().cuda()
            memory_rgb.knn_epoch = torch.tensor(knn_epoch)
            memory_rgb.center_feature_men = memory_center_sort_similar.clone().cuda()
            
            
            memory_mask.feature_sort = memory_center_sort_index.clone().cuda()
            memory_mask.knn_epoch = torch.tensor(knn_epoch)
            memory_mask.center_feature_men = memory_center_sort_similar.clone().cuda()
            del center_feature, center_feature_men, pse_set

        
        memory_rgb.label_count = label_count
        memory_mask.label_count = label_count
        memory_fusion.label_count = label_count
        memory_rgb.labels = pseudo_labels.cuda()
        memory_mask.labels = pseudo_labels.cuda()
        memory_fusion.labels = pseudo_labels.cuda()
        
        train_loader1 = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset)
        train_loader1.new_epoch()
        trainer.train(epoch, train_loader1, optimizer,print_freq=args.print_freq, train_iters=len(train_loader1))
        now_time_after_epoch =  time.monotonic()
        print(
            'the time of cluster refinement is {}'.format(now_time_after_epoch-now_time_before_cluster)
        )
        
        
        if epoch > 30:
            args.eval_step = 1 
        
        
        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            if args.dataset == 'prcc':
                mAP, cmc_now, cmc_now_10 = evaluate_mean(evaluator1, datasets_prcc, test_loaders_prcc)
            else:
                cmc_socore1,mAP1 = evaluator1.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
                mAP, cmc_now, cmc_now_10  = mAP1, cmc_socore1[0], cmc_socore1[9]
            
            print('===============================================')
            print('the RGB model performance')
            print('model mAP: {:5.1%}'.format(mAP))
            print('model cmc: {:5.1%}'.format(cmc_now))
            print('model cmc_10: {:5.1%}'.format(cmc_now_10))
            print('===============================================')
            
            
            is_best = (mAP>best_mAP)
            is_bset = (cmc_now > best_rank1)
            best_mAP = max(mAP, best_mAP)
            best_rank1 = max(cmc_now, best_rank1)
            
            
            
            save_checkpoint({
                'state_dict': model_mask.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best,fpath=osp.join(args.logs_dir, args.dataset, 'model_2_checkpoint.pth.tar'))
            save_checkpoint({
                'state_dict': model_rgb.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best,fpath=osp.join(args.logs_dir, args.dataset, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model cmc: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, cmc_now, best_rank1, ' *' if is_best else ''))
        lr_scheduler.step()

        
        
    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, args.dataset, 'model_best.pth.tar'))
    model_rgb.load_state_dict(checkpoint['state_dict'])
    
    if args.dataset == 'prcc':
        mAP, cmc_now, cmc_now_10 = evaluate_mean(evaluator1, datasets_prcc, test_loaders_prcc)
    else:
        cmc_socore1,mAP1 = evaluator1.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
        mAP, cmc_now, cmc_now_10  = mAP1, cmc_socore1[0], cmc_socore1[9]
        
    
    print('=================RGB===================')
    print('the RGB model performance')
    print('model mAP: {:5.1%}'.format(mAP))
    print('model cmc: {:5.1%}'.format(cmc_now))
    print('model cmc_10: {:5.1%}'.format(cmc_now_10))
    print('===============================================')
    
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mask_CACL")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='ltcc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.60,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--output_weight', type=float, default=1.0,
                        help="loss outputs for weight ")
    parser.add_argument('--ratio_cluster', type=float, default=1.0,
                        help="cluster hypter ratio ")
    parser.add_argument('--allknn', type=int, default=0,
                        help="knn ")
    parser.add_argument('--cr', action="store_true", default=False,
                        help="use cluster refinement in CACL")
    
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--loss-size', type=int, default=2)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--sic_weight', type=float, default=1,
                        help="loss outputs for sic ")
    # training configs
    parser.add_argument('--seed', type=int, default=111)#
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
