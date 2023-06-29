from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

import os

import random

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class Mask_PRCCdataset(BaseImageDataset):
    
    # dataset_dir = '.'

    def __init__(self, root, verbose=True, **kwargs):
        super(Mask_PRCCdataset, self).__init__()
        # self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir = root
        #self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
        self.train_dir = osp.join(self.dataset_dir, 'rgb/train')
        # self.query_dir = osp.join(self.dataset_dir, 'rgb/test/C')
        # self.gallery_dir = osp.join(self.dataset_dir, 'rgb/test/A')

        #self._download_data()
        #self._check_before_run()

        train = self._process_dir_train(self.train_dir, relabel=True)
        # query = self._process_dir_train(self.query_dir, relabel=False)
        # gallery = self._process_dir_gallery(self.gallery_dir, relabel=False)

        if verbose:
            print("=> PRCCdataset loaded")
            self.print_mask_dataset_statistics(train)

        # self.train = train
        # self.query = query
        # self.gallery = gallery

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir_train(self, dir_path, relabel=False):
        
        img_paths = os.listdir(dir_path)  
        dataset = []
        for pid in img_paths:
            paths_small = osp.join(dir_path,pid)
            images_paths_url = os.listdir(paths_small)
            for url in images_paths_url:
                dataset.append((osp.join(paths_small,url),int(pid),1, 1))
        return dataset
    
    def _process_dir_gallery(self, dir_path, relabel=False):
        img_paths = os.listdir(dir_path)
        dataset = []
        for pid in img_paths:
            paths_small = osp.join(dir_path,pid)
            images_paths_url = os.listdir(paths_small)
            index = random.randint(0,len(images_paths_url)-1)# 随机抽取一个作为id
            dataset.append((osp.join(paths_small,images_paths_url[index]),int(pid),0, 0))
        return dataset
            
            