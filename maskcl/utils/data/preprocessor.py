from __future__ import absolute_import
import pdb
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image


class Preprocessor(Dataset):
    def __init__(self, dataset, train = True, root=None, transform1=None,transform2=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, clothesid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        
        if self.train:
            fname_mask = fname.split('/')
            if 'ltcc' in fname_mask:
                mask_path = '/root/pxu1/datasets/ltcc_all/ltccmask/train'
                fname_mask = osp.join(mask_path, fname_mask[-1])
                img_mask = Image.open(fname_mask).convert('RGB')
                
            if 'celebreidlight' in fname_mask:
                mask_path = '/root/pxu1/datasets/celebreidlight_all/celebreidlightmask/train'
                fname_mask = osp.join(mask_path, fname_mask[-1][:-3] + 'png')
                img_mask = Image.open(fname_mask).convert('RGB')
            elif 'celebreidlight' not in fname_mask and 'celebreid' in fname_mask:
                mask_path = '/root/pxu1/datasets/celebreidmask/train'
                fname_mask = osp.join(mask_path, fname_mask[-1][:-3] + 'png')
                img_mask = Image.open(fname_mask).convert('RGB')  
                
            if 'prcc' in fname_mask:
                mask_path = '/root/pxu1/datasets/prcc_all/prccmask/train'
                fname_mask = osp.join(mask_path, fname_mask[-2],fname_mask[-1][:-3] + 'png')
                img_mask = Image.open(fname_mask).convert('RGB')
                
            if 'VC-Clothes' in fname_mask:
                mask_path = '/root/pxu1/datasets/VC-Clothes_all/VC-Clothesmask/train'
                fname_mask = osp.join(mask_path,fname_mask[-1][:-3] + 'png')
                img_mask = Image.open(fname_mask).convert('RGB')
                
            if 'nkuhp' in fname_mask:
                mask_path = '/root/pxu1/datasets/data/nkuhpmask/bounding_box_train'
                fname_mask = osp.join(mask_path, fname_mask[-1][:-3] + 'png')
                img_mask = Image.open(fname_mask).convert('RGB')
            if 'market1501' in fname_mask:
                mask_path = '/root/pxu1/datasets/data/market1501mask/Market-1501-v15.09.15/bounding_box_train'
                fname_mask = osp.join(mask_path, fname_mask[-1][:-3] + 'png')
                img_mask = Image.open(fname_mask).convert('RGB')
            if 'dukemtmcreid' in fname_mask:
                mask_path = '/root/pxu1/datasets/data/dukemtmcreidmask/DukeMTMC-reID/bounding_box_train'
                fname_mask = osp.join(mask_path, fname_mask[-1][:-3] + 'png')
                img_mask = Image.open(fname_mask).convert('RGB')
            
        if self.transform1 is not None:
            img1 = self.transform1(img)
            img2 = self.transform2(img)
        
        if self.train:
            img_mask = self.transform2(img_mask)
        else:
            img_mask = img1
            
        
        return img1, img_mask, fname, pid,clothesid, camid, index
