# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp
import numpy as np

from .bases import BaseImageDataset


class MarketTriplet(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(MarketTriplet, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.mask_train_dir = osp.join(self.dataset_dir, 'mask', 'train')
        self.mask_val_dir = osp.join(self.dataset_dir, 'mask', 'val')
        self.mask_query_dir = osp.join(self.dataset_dir, 'mask', 'query')
        self.mask_gallery_dir = osp.join(self.dataset_dir, 'mask', 'gallery')

        self._check_before_run()

        pid2label = self.get_pid2label(self.train_dir)
        train = self._process_dir(self.train_dir, pid2label=pid2label, relabel=True)
        val = self._process_dir(self.val_dir, pid2label=pid2label, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.mask_train = self._process_dir_mask(self.mask_train_dir, pid2label=pid2label, relabel=True)
        self.mask_val = self._process_dir_mask(self.mask_val_dir, pid2label=pid2label, relabel=True)
        self.mask_query = self._process_dir_mask(self.mask_query_dir, relabel=False)
        self.mask_gallery = self._process_dir_mask(self.mask_gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.val = val
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_val_pids, self.num_val_imgs, self.num_val_cams = self.get_imagedata_info(self.val)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def get_pid2label(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid_container = np.sort(list(pid_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, pid2label=None, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        # if relabel is True:
        #     pid_container = set()
        #     for img_path in img_paths:
        #         pid, _ = map(int, pattern.search(img_path).groups())
        #         if pid == -1: continue  # junk images are just ignored
        #         pid_container.add(pid)
        #     pid_container = np.sort(list(pid_container))
        #     pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel and pid2label is not None:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_dir_mask(self, dir_path, pid2label=None, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel and pid2label is not None:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
