# encoding: utf-8
import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, clos= [], [], []
        for _, pid, clothes, camid in data:
            pids += [pid]
            cams += [camid]
            clos += [clothes]
        pids = set(pids)
        cams = set(cams)
        clos = set(clos)
        num_pids = len(pids)
        num_clos = len(clos)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams, num_clos

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_clos = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_clos = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_clos = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset   | # ids | # images | # cameras | # clothes")
        print("  ----------------------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams, num_train_clos))
        print("  query    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams, num_query_clos))
        print("  gallery  | {:5d} | {:8d} | {:9d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_clos))
        print("  ----------------------------------------------------")
        
    def print_mask_dataset_statistics(self, train):
        num_train_pids, num_train_imgs, num_train_cams, num_train_clos = self.get_imagedata_info(train)
        # num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        # num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------------------")
        print("  subset   | # ids | # images | # cameras | # clothes")
        print("  ----------------------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams, num_train_clos))
        # print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        # print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------------------")
