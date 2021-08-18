from utils.face3d.face3d.utils import show_vertices
import imgaug.augmenters as iaa
import numpy as np
from .params import *
import math
import cv2
from scipy import ndimage
import torch
import random
import numba
import pathlib
import os
# from .face3d import face3d
# fm = face3d.face_model.FaceModel()

cwd = pathlib.Path(__file__).parent.resolve()
hand_folder = os.path.join(cwd, 'hand')
hand_path_list = list(map(str, pathlib.Path(hand_folder).glob('*.png')))
hand_list = [cv2.imread(hand, cv2.IMREAD_UNCHANGED) for hand in hand_path_list]

def ddfa_augment(img, params, roi_box, full=False):
    if full:
        img = hide_face(img, params, roi_box)
        angles = np.linspace(0, 360, num=13)
        img, params = rotate_samples(img, params, random.choice(angles))
        img = vanilla_aug(image=img)
    else:
        if np.random.rand() < 0.5:
            img = hide_face(img, params, roi_box)

        if np.random.rand() < 0.5:
            angles = np.linspace(0, 360, num=13)
            img, params = rotate_samples(img, params, random.choice(angles))

        if np.random.rand() < 0.75:
            img = vanilla_aug(image=img)

    return np.ascontiguousarray(img), params


def hide_face(img, params, roi_box):
    rate = np.random.rand()
    if rate < 0.3:
        img = hand_face(img, roi_box)
    elif 0.3 <= rate < 0.6:
        img = crop_range(img, ratio=1/3)
    else:
        img, params = random_shift(img, params)

    return img

# Use case: increase generalization; input image is not too bad.
vanilla_aug = iaa.OneOf([
    iaa.Sometimes(0, iaa.Grayscale()),
    iaa.Grayscale(),
    # iaa.imgcorruptlike.Pixelate(severity=(1, 3)),	
    # iaa.imgcorruptlike.JpegCompression(severity=(3, 5)),	
    iaa.KMeansColorQuantization(n_colors=(80, 100)),	
    iaa.UniformColorQuantization(n_colors=(10, 15)),
    iaa.LinearContrast((1.5, 2)),
    iaa.LogContrast(gain=(0.5, 1.5)),
    iaa.SigmoidContrast(gain=7, cutoff=(0.4, 0.6)),
    iaa.AdditivePoissonNoise((10, 15), per_channel=True),
    iaa.AdditivePoissonNoise((10, 15)),
    iaa.AdditiveLaplaceNoise(scale=(5, 15)),
    iaa.AdditiveLaplaceNoise(scale=(5, 15), per_channel=True),
    iaa.ChannelShuffle(p=1),
    # iaa.imgcorruptlike.SpeckleNoise(severity=(1,3))
])


@numba.njit()
def n_rotate_vertex(img, params, angle):
    """
    Create params for a rotated 3dmm.

    Params:
    :params: 3dmm parameters.
            -> np.ndarray.
    :angle: rotate angle. 
    """
    img_height, img_width = img.shape[:2]

    p_ = params[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = np.zeros((3,1), dtype=np.float64)
    offset[:,0] = p_[:, 3]

    p = np.ascontiguousarray(p)
    offset = np.ascontiguousarray(offset)
    
    rad_angle = angle / 180 * math.pi
    rotate_matrix = np.array([
        [np.cos(rad_angle), -np.sin(rad_angle), 0.],
        [np.sin(rad_angle), np.cos(rad_angle), 0.],
        [0., 0., 1.]
    ], dtype=np.float64)
    
    center_old = np.array([img_width/2, img_height/2, 1], dtype=np.float64)
    center_new = rotate_matrix @ center_old
    rotate_offset = center_old-center_new
    rotate_offset = rotate_offset.reshape(3,)
    
    #################################################
    flip_matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    flip_offset = np.array([0, img_height, 0], dtype=np.float64)
    norm_trans = np.array([img_width/2, img_height/2, 0], dtype=np.float64)

    # shp = params[12:72].reshape(-1, 1)
    # exp = params[72:].reshape(-1, 1)
    # vertex = fm.bfm.reduced_generated_vertices(shp, exp)[fm.bfm.kpt_ind]

    # trans_v = vertex @ p.T + offset.reshape(3,) + norm_trans
    # flip_v = trans_v @ flip_matrix.T + flip_offset
    # rot_v = flip_v @ rotate_matrix.T + rotate_offset
    
    # new_p = (p.T @ flip_matrix.T @ rotate_matrix.T).T
    # new_offset = ((offset.reshape(3,) + norm_trans) @ flip_matrix.T + flip_offset) @ rotate_matrix.T + rotate_offset
    
    # new_all = vertex @ new_p.T + new_offset
    #################################################
    new_p = flip_matrix @ rotate_matrix @ flip_matrix @ p
    new_offset = (((offset.reshape(3,) + norm_trans) @ flip_matrix.T + flip_offset) @ rotate_matrix.T + rotate_offset - \
                flip_offset) @ flip_matrix - norm_trans

    new_camera_matrix = np.concatenate((new_p, new_offset.reshape(3,1)), axis=1)
    params[:12] = new_camera_matrix.reshape(12,1)

    # vertex = fm.reconstruct_vertex(img, params)[fm.bfm.kpt_ind]
    # face3d.utils.show_pts(r_img, vertex)
    # import ipdb; ipdb.set_trace(context=10)

    return params


def rotate_samples(img, params, angle):
    if isinstance(params, torch.Tensor):
        params = params.numpy().astype(np.float64)
    else:
        params = params.astype(np.float64)

    # r_param = rotate_vertex(img, params, angle)
    r_param = n_rotate_vertex(img, params, angle)
    r_img = ndimage.rotate(img, -angle, reshape=False)

    return r_img, r_param


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


@numba.njit()
def get_landmarks_wrapbox(landmarks):
    box_left = int(np.floor(np.min(landmarks.T[0])))
    box_right = int(np.ceil(np.max(landmarks.T[0])))
    box_top = int(np.floor(np.min(landmarks.T[1])))
    box_bot = int(np.ceil(np.max(landmarks.T[1])))

    return box_left, box_top, box_right, box_bot


def hand_face(face_img, face_location):
    hand_img = random.choice(hand_list)
    hand_h, hand_w, _ = hand_img.shape
    face_h = face_location[3] - face_location[1]
    face_w = face_location[2] - face_location[0]

    # hand_img = adjust_overlay_color(face_img[y1:y2, x1:x2], hand_img)

    area_ratio = random.uniform(0.8, 1.2)
    hand_area = hand_w*hand_h
    new_hand_area = face_w*face_h*area_ratio
    hand_resize_ratio = np.sqrt(new_hand_area/hand_area)
    w_resize = int(hand_resize_ratio * hand_w)
    h_resize = int(hand_resize_ratio * hand_h)

    # Resize hand, keep aspect ratio.
    hand_resized = cv2.resize(hand_img,(w_resize, h_resize))

    # cv2.rectangle(face_img, tuple(face_location[:2]), tuple(face_location[2:4]), color=(0,255,0), thickness=1)
    x_range = np.linspace(face_location[0] + face_w * 0.3, face_location[0] + face_w * 3 / 4, num=10)
    y_range = np.linspace(face_location[1], face_location[1] + face_h * 3 / 4, num=10)
    paste_x = int(random.choice(x_range))
    paste_y = int(random.choice(y_range))

    full_img = overlay_transparent(face_img, hand_resized, (paste_x, paste_y), scale=0.93)

    return full_img


@numba.njit()
def overlay_transparent(src, overlay, pos, scale=1):
    """
    :src: Input Color Background Image
    :overlay: transparent Image (BGRA)
    :pos:  position where the image to be blit.
    :scale: scale factor of alpha channel.
    :return: Resultant Image
    """
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    x, y = pos

    #loop over all pixels and apply the blending equation
    for i in numba.prange(h):
        for j in numba.prange(w):
            if y+i >= rows or x+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0)*scale # read the alpha channel 
            src[y+i][x+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[y+i][x+j]
    return src


def crop_range(img, 
              max_width_crop=None, 
              max_height_crop=None,
              min_width_crop=None,
              min_height_crop=None,
              ratio=1/4):
    height, width, _ = img.shape

    if max_width_crop is None:
        max_width_crop = int(width*ratio)
    
    if max_height_crop is None:
        max_height_crop = int(height*ratio)

    if min_width_crop is None:
        min_width_crop = int(max_width_crop/2)
    
    if min_height_crop is None:
        min_height_crop = int(max_height_crop/2)

    x_crop_length = random.randint(int(min_width_crop), int(max_width_crop))
    y_crop_length = random.randint(int(min_height_crop), int(max_height_crop))

    def cropWidth(img):
        x_rd = np.random.rand()
        if x_rd > 0.5:
            # print('x_rd > 0.5')
            img[0:x_crop_length, :, :] = 0.
        else:
            # print('x_rd <= 0.5')
            img[width-x_crop_length:, :, :] = 0.

        return img

    def cropHeight(img):
        y_rd = np.random.rand()
        if y_rd > 0.5:
            # print('y_rd > 0.5')
            img[:, 0:y_crop_length, :] = 0.
        else:
            # print('y_rd <= 0.5')
            img[:, height-y_crop_length:, :] = 0.

        return img

    rd = np.random.rand()
    if 0.45 > rd >= 0:
        # print('w')
        img = cropWidth(img)
    elif 0.9 > rd >= 0.45:
        # print('h')
        img = cropHeight(img)
    else:
        # print('both')
        img = cropWidth(img)
        img = cropHeight(img)

    return img


def random_shift(img, params):
    shift_x = np.random.randint(5,16)
    shift_y = np.random.randint(5,16)
    return shift(img, params, shift_value=(shift_x, shift_y))

@numba.njit()
def shift(img, params, shift_value=(10,10)):
    img_height, img_width = img.shape[:2]
    canvas = np.zeros((img_height*2,img_width*2,3), dtype=np.uint8)
    crop_size = int(img_height/2)
    canvas[crop_size:img_height+crop_size, crop_size:img_width+crop_size, :] = img
    
    new_x1 = crop_size+shift_value[0]
    new_y1 = crop_size+shift_value[1]
    new_x2 = crop_size+img_width+shift_value[0]
    new_y2 = crop_size+img_height+shift_value[1]
    
    new_img = canvas[new_y1:new_y2, new_x1:new_x2]

    '''
    params at index 3 and 7 account for the translation along x and y axis.
    '''
    params[3] -= shift_value[0]
    params[7] += shift_value[1]

    return new_img, params