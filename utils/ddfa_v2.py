#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path
import numpy as np
import torch.utils.data as data
import cv2
from .augment import ddfa_augment, random_crop
from .face3d import face3d
from .face3d.face3d.utils import show_pts
import scipy.io as sio
fm = face3d.face_model.FaceModel()

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

class DDFAv2_Dataset(data.Dataset):
    def __init__(self, 
                root, 
                transform=None, 
                aug=True, 
                hide_face_rate=0.75, 
                rotate_rate=0.75,
                vanilla_aug_rate=0.9,
                flip_rate=0.5,
                shift=True
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
        self.flip_rate=flip_rate
        self.shift = shift

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

        params = label['params'].reshape(-1,1)
        roi_box = label['roi_box'][0]

        # pts = fm.reconstruct_vertex(img, params, False)[fm.bfm.kpt_ind]
        # show_pts(img, pts, 'original')

        if self.aug:
            img, params, roi_box = random_crop(img, roi_box, params, target_size=128, expand_ratio=1.5)
            # pts = fm.reconstruct_vertex(img, params, False)[fm.bfm.kpt_ind]
            # show_pts(img, pts, 'first crop')

            radius = max((roi_box[2]-roi_box[0]), (roi_box[3]-roi_box[1])) / 2
            img, params = ddfa_augment(
                img=img, 
                params=params, 
                roi_box=roi_box, 
                full=True, 
                hide_face_rate=self.hide_face_rate, 
                rotate_rate=self.rotate_rate, 
                vanilla_aug_rate=self.vanilla_aug_rate,
                flip_rate=self.flip_rate,
            )

            # pts = fm.reconstruct_vertex(img, params, False)[fm.bfm.kpt_ind]
            # show_pts(img, pts, 'augment')
            # cv2.waitKey(0)

            img, params, roi_box = random_crop(img, roi_box, params, target_size=128, radius=radius, shift=self.shift)
            
            # pts = fm.reconstruct_vertex(img, params, False)[fm.bfm.kpt_ind]
            # show_pts(img, pts, 'second crop')

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        return img, params

    def _transform_params(self, params):
        t_params = params.reshape(-1,).astype(np.float32)
        if fm.bfm.params_mean is None:
            params_mean = np.zeros(t_params.shape)
        else:
            params_mean = fm.bfm.params_mean

        if fm.bfm.params_std is None:
            params_std = np.ones(t_params.shape)
        else:
            params_std = fm.bfm.params_std

        t_params = (t_params - params_mean) / params_std

        return t_params

if __name__=='__main__':
    dataset = DDFAv2_Dataset('data/AFLW2000_3ddfa')
    for idx in dataset:
        _ = dataset[idx]