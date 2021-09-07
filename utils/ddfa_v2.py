#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
import numpy as np
import torch.utils.data as data
import cv2
from .augment import ddfa_augment, random_crop
from .face3d import face3d
import scipy.io as sio
fm = face3d.face_model.FaceModel()

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

class DDFAv2_Dataset(data.Dataset):
    def __init__(self, 
                root, 
                transform=None, 
                aug=True, 
                hide_face_rate=0.5, 
                rotate_rate=0.5, 
                vanilla_aug_rate=0.6,
                ):
        if isinstance(root, list):
            self.file_list = root
        else:
            self.file_list = list(Path(root).glob('**/*.jpg'))
        self.transform = transform
        self.img_loader = img_loader
        self.aug = aug
        self.hide_face_rate = hide_face_rate
        self.rotate_rate = rotate_rate
        self.vanilla_aug_rate = vanilla_aug_rate

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img, params = self._generate_face_sample(idx)

        '''
        This part is for showing samples before feeding the model.
        '''
        # pts = fm.reconstruct_vertex(img, params)[fm.bfm.kpt_ind][:,:2]
        # # face3d.utils.show_pts(img, pts)
        # face3d.utils.draw_pts(img, pts)
        # import ipdb; ipdb.set_trace(context=10)

        img = self.transform(img)
        params = self._transform_params(params)

        return img, params
    
    def _generate_face_sample(self, idx):
        img_path = str(self.file_list[idx])
        label_path = img_path.replace('jpg','mat')

        img = self.img_loader(img_path)
        label = sio.loadmat(label_path)

        params = label['params'].reshape(101,1)
        roi_box = label['roi_box'][0]

        if self.aug:
            img, params, roi_box = random_crop(img, roi_box, params, target_size=128)
            img, params = ddfa_augment(
                img=img, 
                params=params, 
                roi_box=roi_box, 
                full=False, 
                hide_face_rate=self.hide_face_rate, 
                rotate_rate=self.rotate_rate, 
                vanilla_aug_rate=self.vanilla_aug_rate
            )

        return img, params

    def _transform_params(self, params):
        t_params = params.reshape(-1,).astype(np.float32)
        t_params = (t_params - fm.bfm.params_mean_101) / fm.bfm.params_std_101

        return t_params

