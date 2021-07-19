import imgaug.augmenters as iaa
import numpy as np
from scipy.ndimage.interpolation import rotate
from .params import *
import math
import cv2
from scipy import ndimage
import torch
import random
import numba

# Use case: increase generalization; input image is not too bad.
vanilla_aug = iaa.OneOf([
    iaa.Sometimes(0, iaa.Grayscale()),
    iaa.Grayscale(),
    iaa.imgcorruptlike.Pixelate(severity=(1, 3)),	
    iaa.imgcorruptlike.JpegCompression(severity=(3, 5)),	
    iaa.KMeansColorQuantization(n_colors=(70, 100)),	
    iaa.UniformColorQuantization(n_colors=(10, 15)),
    # iaa.MotionBlur(k=(10, 15), angle=(-45, 45)),
    # iaa.GaussianBlur((2.0, 5.0)),
    # iaa.AverageBlur(k=(7, 10)),
    # iaa.MedianBlur(k=(7, 13)),
    iaa.LinearContrast((1.5, 2)),
    iaa.LogContrast(gain=(0.5, 1.5)),
    iaa.SigmoidContrast(gain=7, cutoff=(0.4, 0.6)),
    iaa.AdditivePoissonNoise((10, 15), per_channel=True),
    iaa.AdditivePoissonNoise((10, 15)),
    iaa.AdditiveLaplaceNoise(scale=(5, 15)),
    iaa.AdditiveLaplaceNoise(scale=(5, 15), per_channel=True),
    iaa.ChannelShuffle(p=1),
    iaa.Sequential([
        iaa.Resize(0.35),
        iaa.Resize(1/0.35)
    ]),
    # iaa.Sequential([
    #     iaa.Resize(0.25),
    #     iaa.Resize(1/0.25)
    # ]),
])

@numba.njit()
def rotate_vertex(img, param, angle):
    """
    Create param for a rotated 3dmm.

    Params:
    :param: 3dmm parameters.
            -> np.ndarray.
    :angle: rotate angle. 
    """
    # TODO: jit dis.
    img_height, img_width = img.shape[:2]

    if len(param) == 62:
        param = param * param_std + param_mean
    # p, offset, alpha_shp, alpha_exp = ddfa_utils._parse_param(param)

    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    # offset = p_[:, 3].reshape(3, 1)
    offset = np.zeros((3,1), dtype=np.float64)
    offset[:,0] = p_[:, 3]
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    
    # img = cv2.flip(img,0)
    # rotate_matrix = angle2rotmat(angle / 180 * math.pi)
    rad_angle = angle / 180 * math.pi
    rotate_matrix = np.array([
        [np.cos(rad_angle), -np.sin(rad_angle), 0.],
        [np.sin(rad_angle), np.cos(rad_angle), 0.],
        [0., 0., 1.]
    ])

    center_old = np.array([img_width/2, img_height/2, 1])
    center_new = rotate_matrix @ center_old
    rotate_offset = center_old-center_new
    rotate_offset = rotate_offset.reshape(3,-1)

    # rotated_vertex = rotate_matrix @ vertex + rotate_offset
    new_p = rotate_matrix @ p.astype(np.float64)
    new_offset = rotate_matrix @ offset + rotate_offset

    new_camera_matrix = np.concatenate((new_p, new_offset), axis=1)
    param[:12] = new_camera_matrix.reshape(12,)
    param = (param-param_mean) / param_std

    return param


def rotate_samples(img, param, angle):
    if isinstance(param, torch.Tensor):
        param = param.numpy()

    r_param = rotate_vertex(img, param, angle)
    r_img = ndimage.rotate(img, angle, reshape=False)

    return r_img, torch.from_numpy(r_param)


def ddfa_augment(img, param):
    if np.random.rand() < 0.95:
        img = vanilla_aug(image=img)
   
    # img, param = rotate_vertex(img, param, 90)
    # vertex = ddfa_utils.reconstruct_vertex(param.numpy())

    # pts = vertex[:2].T
    # for _pts in pts:
    #     _pts = tuple(_pts.astype(np.uint8))
    #     cv2.circle(img, _pts, 3, (0,255,0), -1, 10)

    if np.random.rand() > 0.75:
        angles = np.linspace(0, 360, num=13)
        img, param = rotate_vertex(img, param, random.choice(angles))
    else:
        img, param = rotate_vertex(img, param, 180)

    return img, param


def angle2rotmat(angle):
    """
    Convert angle to rotation matrix using z-axis
    """
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return R