from pathlib import Path
import numpy as np
import torch.utils.data as data
import cv2
from utils.augment import ddfa_augment, random_crop, rotate_samples, flip
from utils.face3d import face3d
import scipy.io as sio
fm = face3d.face_model.FaceModel()

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

def task(item):
    img = img_loader(item)

    label_path = item.replace('jpg','mat')
    label = sio.loadmat(label_path)

    params = label['params'].reshape(-1,1)
    roi_box = label['roi_box'][0]

    img, params, roi_box = random_crop(img, roi_box, params, target_size=128, expand_ratio=1.5)
    radius = max((roi_box[2]-roi_box[0]), (roi_box[3]-roi_box[1])) / 2
    

    img, params, roi_box = random_crop(img, roi_box, params, target_size=128, radius=radius)

if __name__ == '__main__':
