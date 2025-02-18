#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
import cv2
from .augment import ddfa_augment
from .face3d import face3d
import scipy.io as sio
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
        img, params = self._generate_face_sample(idx)

        img = self.transform(img)

        return img, params
    
    def _generate_face_sample(self, idx):
        img_path = str(self.file_list[idx])
        label_path = img_path.replace('jpg','mat')

        img = self.img_loader(img_path)
        label = sio.loadmat(label_path)

        params = label['params']
        roi_box = label['roi_box']

        if self.aug:
            img, params = ddfa_augment(img, params, roi_box, True)

        # pts = fm.reconstruct_vertex(img, params)[fm.bfm.kpt_ind][:,:2]
        # for pt in pts:
        #     pt = tuple(pt.astype(np.uint8))
        #     cv2.circle(img, pt, 1, (0,255,0), -1, 10)
        # cv2.imwrite(f'input_samples/{idx}.jpg', img)

    def __len__(self):
        return len(self.file_list)
