from __future__ import print_function, absolute_import
import time
import numpy as np
import collections
import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class CACLTrainer_UDA(object):
    def __init__(self, encoder, memory, source_classes):
        super(CACLTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes

    def train(self, epoch, data_loader_source, data_loader_target,
                    optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, _, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()
            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)
            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # forward
            f_out = self._forward(inputs)

            # de-arrange batch
            f_out = f_out.view(device_num, -1, f_out.size(-1))
            f_out_s, f_out_t = f_out.split(f_out.size(1)//2, dim=1)
            f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1, f_out.size(-1))

            # compute loss with the hybrid memory
            loss_s = self.memory(f_out_s, s_targets)
            loss_t = self.memory(f_out_t, t_indexes+self.source_classes)

            loss = loss_s+loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})\t'
                      'Loss_t {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class CACLTrainer_USL(object):
    def __init__(self, encoder, memory):
        super(CACLTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            inputs, _, indexes = self._parse_data(inputs)
            f_out = self._forward(inputs)
            loss = self.memory(f_out, indexes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
    
    
    
    

    
    
    
class CACLSIC_USL(object):
    def __init__(self, encoder1, encoder2, memory1, memory2):
        super(CACLSIC_USL, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.memory1 = memory1
        self.memory2 = memory2

    def train(self, epoch, data_loader1, data_loader2, optimizer, print_freq=10, train_iters=400):
        self.encoder1.train()
        self.encoder2.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses3 = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs1 = data_loader1.next()
            inputs3 = data_loader2.next()
            
            data_time.update(time.time() - end)

            inputs1,inputs2,_,indexes1 = self._parse_data(inputs1)

            bn_x1, full_conect1, bn_x2, full_conect2, bn_x3, full_conect3 = self._forward(inputs1,inputs2)

            flag = 0
            loss1 = self.memory1(bn_x1, full_conect2.clone(), full_conect3.clone(), indexes1, back = flag)
            flag = 1
            loss2 = self.memory2(bn_x2, full_conect1.clone(), full_conect3.clone(), indexes1, back = flag)
            flag = 1
            loss3 = self.memory2(bn_x3, full_conect1.clone(), full_conect2.clone(), indexes1, back = flag)
            
            
            loss = (loss1 + loss2 + loss3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses.update(loss.item())
            losses.update(loss.item())
            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader1),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
    def _parse_data(self, inputs):
        imgs1, imgs2, _, pids, _, indexes = inputs
        return imgs1.cuda(),imgs2.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs1, inputs2):
        bn_x1, full_conect1 = self.encoder1(inputs1)
        bn_x2, full_conect2 = self.encoder2(inputs2)
        return bn_x1, full_conect1, bn_x2, full_conect2




class Mask_CACLSIC_USL(object):
    def __init__(self, encoder1, encoder2, encoder4, memory1, memory2, memory4):
    # def __init__(self, *encoders, memories):
        super(Mask_CACLSIC_USL, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder4 = encoder4
        self.memory1 = memory1
        self.memory2 = memory2
        self.memory4 = memory4


    def train(self, epoch, data_loader1, optimizer, print_freq=10, train_iters=400):
        self.encoder1.train()
        self.encoder2.train()
        self.encoder4.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses4 = AverageMeter()
        losses5 = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs1 = data_loader1.next()
            
            data_time.update(time.time() - end)

            inputs1,inputs2, _, indexes1 = self._parse_data(inputs1)

            bn_x1, full_conect1, bn_x2, full_conect2, fusion1 = self._forward(inputs1,inputs2)

            flag = 2
            loss1 = self.memory1(bn_x1, full_conect2.clone(), indexes1, back = flag)
            flag = 1
            loss2 = self.memory2(bn_x2, full_conect1.clone(), indexes1, back = flag)
            flag = 0
            loss4 = self.memory4(fusion1, full_conect1.clone(), indexes1, back = flag)
            
            loss = (loss1 + loss2 + loss4)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses1.update(loss1.item())
            losses2.update(loss2.item())
            losses4.update(loss4.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ALL {:.3f} ({:.3f})\t'
                      'Loss_RGB {:.3f} ({:.3f})\t'
                      'Loss_Mask {:.3f} ({:.3f})\t'
                      'Loss_fusion {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader1),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses1.val, losses1.avg,
                              losses2.val, losses2.avg,
                              losses4.val, losses4.avg
                              ))
                
                
    def _parse_data(self, inputs):
        img, imag_mask, _, pids, clothesid, _, indexes = inputs
        return img.cuda(),imag_mask.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs1, inputs2):
        
        bn_x1, full_conect1 = self.encoder1(inputs1)
        bn_x2, full_conect2 = self.encoder2(inputs2)
        fusion1 = self.encoder4(bn_x1, bn_x2)
        
        return bn_x1, full_conect1, bn_x2, full_conect2, fusion1