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
from . import ddfa
import os

cwd = pathlib.Path(__file__).parent.resolve()
hand_folder = os.path.join(cwd, 'hand')
hand_path_list = list(map(str, pathlib.Path(hand_folder).glob('*.png')))
hand_list = [cv2.imread(hand, cv2.IMREAD_UNCHANGED) for hand in hand_path_list]

def ddfa_augment(img, param):
    if np.random < 0.5:
        img = hide_face(img, param)

    if np.random.rand() < 0.95:
        img = vanilla_aug(image=img)
   
    if np.random.rand() > 0.5:
        angles = np.linspace(0, 360, num=13)
        img, param = rotate_vertex(img, param, random.choice(angles))

    return img, param


def hide_face(img, param):
    if np.random.rand() < 0.5:
        vertex = ddfa.reconstruct_vertex(param)
        bbox = get_landmarks_wrapbox(vertex[:2].T)
        img = hand_face(img, bbox)
    else:
        img = crop_range(img)

    return img

# Use case: increase generalization; input image is not too bad.
vanilla_aug = iaa.OneOf([
    iaa.Sometimes(0, iaa.Grayscale()),
    iaa.Grayscale(),
    iaa.imgcorruptlike.Pixelate(severity=(1, 3)),	
    iaa.imgcorruptlike.JpegCompression(severity=(3, 5)),	
    iaa.KMeansColorQuantization(n_colors=(80, 100)),	
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

    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = np.zeros((3,1), dtype=np.float64)
    offset[:,0] = p_[:, 3]
    
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

    x1, y1, x2, y2 = face_location
    # hand_img = adjust_overlay_color(face_img[y1:y2, x1:x2], hand_img)

    area_ratio = random.uniform(1., 1.5)
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
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of alpha channel.
    :return: Resultant Image
    """
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    x, y = pos

    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
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
            img[0:x_crop_length, :, :] = 0.
        else:
            img[width-x_crop_length:, :, :] = 0.

        return img

    def cropHeight(img):
        y_rd = np.random.rand()
        if y_rd > 0.5:
            img[:, 0:y_crop_length, :] = 0.
        else:
            img[:, height-y_crop_length:, :] = 0.

        return img

    rd = np.random.rand()
    if 0.45 > rd >= 0:
        img = cropWidth(img)
    elif 0.9 > rd >= 0.45:
        img = cropHeight(img)
    else:
        img = cropWidth(img)
        img = cropHeight(img)

    return img

