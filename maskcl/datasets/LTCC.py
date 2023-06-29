import numpy as np
import pdb
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path as osp
import glob

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json



class LTCCdataset(BaseImageDataset):
    # def __init__(self, dir_path, IS_TRAIN = True, data_transforms=None):

    #     img_paths = glob.glob(osp.join(dir_path, '*.png'))
    #     self.bbox_urls = [img_path.strip() for img_path in img_paths]
    #     self.pids = [int(urls.strip().split('/')[-1].strip().split('_')[0]) for urls in self.bbox_urls]
    #     self.IS_TRAIN = IS_TRAIN
    #     if self.IS_TRAIN == False:
    #         self.cids = [int(urls.strip().split('/')[-1].split('_')[-2][1:]) for urls in self.bbox_urls]
    #     else:
    #         self.cids = None
        
    def __init__(self, root, verbose=True, **kwargs):
        super(LTCCdataset, self).__init__()
        
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        print(self.gallery_dir)
        
        
        self.gallery_change = osp.join(self.dataset_dir,'info/cloth-change_id_test.txt')
        
        with open(self.gallery_change) as bbox_url_file:
            bbox_url_list = bbox_url_file.readlines()
            
        self.gallery_change_list = []
        for item in bbox_url_list:
            self.gallery_change_list.append(int(item))
            
        train = self._process_dir_train(self.train_dir, relabel=True)
        query = self._process_dir_query(self.query_dir, relabel=False)
        gallery = self._process_dir_gallery(self.gallery_dir, relabel=False)
        
        if verbose:
            print("=> LTCCdataset loaded")
            self.print_dataset_statistics(train, query, gallery)
        
        # pdb.set_trace()
        self.train = train
        self.query = query
        self.gallery = gallery
    
    # def __getitem__(self, item):
    #     bbox_url = self.bbox_urls[item]
    #     pid = self.pids[item]
    #     if self.IS_TRAIN == False:
    #         tid = self.cids[item]
    #     bbox = Image.open(bbox_url, 'r')
    #     if self.data_transforms is not None:
    #         try:
    #             bbox = self.data_transforms(bbox)
    #         except:
    #             print("Cannot transform bbox: {}".format(bbox_url))
        
    #     if self.IS_TRAIN == True:
    #         return bbox_url, pid, item, item
    #     else:
    #         return bbox_url, pid, tid,item
        
        
    #     img_paths = glob.glob(osp.join(dir_path, '*.png'))
    #     self.bbox_urls = [img_path.strip() for img_path in img_paths]
    #     self.pids = [int(urls.strip().split('/')[-1].strip().split('_')[0]) for urls in self.bbox_urls]
    #     self.IS_TRAIN = IS_TRAIN
    #     if self.IS_TRAIN == False:
    #         self.cids = [int(urls.strip().split('/')[-1].split('_')[-2][1:]) for urls in self.bbox_urls]
    #     else:
    #         self.cids = None
        
    def _process_dir_train(self, dir_path, relabel=False):
        img_paths = os.listdir(dir_path)  
        dataset = []
        for image in img_paths:
            pid = int(image.split('_')[0])
            clothes = int(image.split('_')[1])
            cid = int(image.split('_')[2][1:])
            dataset.append((osp.join(dir_path, image),pid,clothes,cid-1))
        return dataset
    
    
    def _process_dir_gallery(self, dir_path, relabel=False):
        img_paths = os.listdir(dir_path)  
        dataset = []
        for image in img_paths:
            pid = int(image.split('_')[0])    
            clothes = int(image.split('_')[1])
            cid = int(image.split('_')[2][1:])
            if pid in self.gallery_change_list:
                dataset.append((osp.join(dir_path, image),pid,clothes,cid-1))
        return dataset
    
    def _process_dir_query(self, dir_path, relabel=False):
        img_paths = os.listdir(dir_path)  
        dataset = []
        for image in img_paths:
            pid = int(image.split('_')[0])    
            clothes = int(image.split('_')[1])
            cid = int(image.split('_')[2][1:])
            if pid in self.gallery_change_list:
                dataset.append((osp.join(dir_path, image),pid,clothes,cid-1))
        return dataset
    
