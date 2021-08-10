#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
import cv2
from .augment import ddfa_augment
import numba
from .face3d import face3d
fm = face3d.face_model.FaceModel()

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

class DDFAv2_Dataset(data.Dataset):
    def __init__(self, root, transform=None, aug=True, **kargs):
        self.root = root
        self.transform = transform
        self.file_list = list(Path(root).glob('**/*.jpg'))
        self.img_loader = img_loader
        self.aug = aug  

    def __getitem__(self, idx):
        img_path = str(self.file_list[idx])
        target_path = img_path.replace('jpg','npy')

        img = self.img_loader(img_path)
        target = np.load(target_path).reshape(-1,)

        if self.aug:
            img, target = ddfa_augment(img, target, True)

        pts = fm.reconstruct_vertex(img, target)[fm.bfm.kpt_ind][:,:2]
        for pt in pts:
            pt = tuple(pt.astype(np.uint8))
            cv2.circle(img, pt, 1, (0,255,0), -1, 10)
        cv2.imwrite(f'input_samples/{idx}.jpg', img)

        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.file_list)
