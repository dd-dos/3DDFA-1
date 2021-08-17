#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
import numpy as np
import torch.utils.data as data
import cv2
from .augment import ddfa_augment
from .face3d import face3d
import scipy.io as sio
from .params import params_mean_101, params_std_101
fm = face3d.face_model.FaceModel()

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

class DDFAv2_Dataset(data.Dataset):
    def __init__(self, root, transform=None, aug=True):
        self.root = root
        self.transform = transform
        self.file_list = list(Path(root).glob('**/*.jpg'))
        self.img_loader = img_loader
        self.aug = aug

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img, params = self._generate_face_sample(idx)

        '''
        This part is for showing samples before feeding the model.
        '''
        # pts = fm.reconstruct_vertex(img, params)[fm.bfm.kpt_ind][:,:2]
        # face3d.utils.show_pts(img, pts)

        img = self.transform(img)
        params = self._transform_params(params)

        return img, params
    
    def _generate_face_sample(self, idx):
        img_path = str(self.file_list[idx])
        label_path = img_path.replace('jpg','mat')

        img = self.img_loader(img_path)
        label = sio.loadmat(label_path)

        params = label['params']
        roi_box = label['roi_box'][0]

        if self.aug:
            img, params = ddfa_augment(img, params, roi_box, False)

        return img, params

    def _transform_params(self, params):
        t_params = params.reshape(-1,).astype(np.float32)
        t_params = (t_params - params_mean_101) / params_std_101

        return t_params
