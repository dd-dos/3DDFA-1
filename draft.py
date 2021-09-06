import cv2
import numpy as np
from utils.augment import n_rotate_vertex
from vdc_loss import VDCLoss
import scipy.io as sio
from utils.augment import shift
from utils.face3d.face3d.face_model import FaceModel
from utils.face3d.utils import show_pts

fm = FaceModel()

if __name__=='__main__':
    img = cv2.imread('data/0560_0_fliplr.jpg')
    params = sio.loadmat('data/0560_0_fliplr.mat')['params']

    s_img, s_params = shift(img, params.reshape(-1,), (20, 40))
    re_pts = fm.reconstruct_vertex(s_img, s_params.reshape(-1,), False)[fm.bfm.kpt_ind]
    show_pts(s_img, re_pts)
