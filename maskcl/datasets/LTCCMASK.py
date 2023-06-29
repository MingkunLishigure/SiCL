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



class LTCCMASKdataset(BaseImageDataset):
    def __init__(self, root, verbose=True, **kwargs):
        super(LTCCMASKdataset, self).__init__()
        
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')        
        
            
        train = self._process_dir_train(self.train_dir, relabel=True)
        
        if verbose:
            print("=> LTCCdataset loaded")
            self.print_mask_dataset_statistics(train)
        
        # pdb.set_trace()
        self.train = train
        
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
    
    
