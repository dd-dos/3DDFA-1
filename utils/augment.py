from utils.face3d.face3d.utils import show_vertices
import imgaug.augmenters as iaa
import numpy as np
import math
import cv2
from scipy import ndimage
import torch
import random
import numba
import pathlib
import os
from .face3d import face3d
from .face3d.utils import draw_pts
fm = face3d.face_model.FaceModel()

cwd = pathlib.Path(__file__).parent.resolve()
hand_folder = os.path.join(cwd, 'hand')
hand_path_list = list(map(str, pathlib.Path(hand_folder).glob('*.png')))
hand_list = [cv2.imread(hand, cv2.IMREAD_UNCHANGED) for hand in hand_path_list]

def ddfa_augment(img, params, roi_box, full=False, hide_face_rate=0.5, rotate_rate=0.5, vanilla_aug_rate=0.6, flip_rate=0.5):
    if full:
        img = hide_face(img, params, roi_box)
        angles = np.linspace(0, 360, num=13)
        img, params = rotate_samples(img, params, random.choice(angles))
        img = vanilla_aug(image=img)
    else:
        if np.random.rand() < hide_face_rate:
            img = hide_face(img, roi_box)

        if np.random.rand() < rotate_rate:
            angle = random.choice(np.linspace(0, 360, num=13))
            img, params = rotate_samples(img, params, angle)

        if np.random.rand() < vanilla_aug_rate:
            img = vanilla_aug(image=img)

        if np.random.rand() < flip_rate:
            img, params = flip(img, params)

    return np.ascontiguousarray(img), params


def hide_face(img, roi_box):
    if np.random.rand() < 0.5:
        img = hand_face(img, roi_box)
    else:
        height, width = img.shape[:2]
        box_width = roi_box[2] - roi_box[0]
        box_height = roi_box[3] - roi_box[1]
        min_width = max(roi_box[0], width-roi_box[2])
        min_height = max(roi_box[1], height-roi_box[3])

        img = crop_range(img, 
                        max_width_crop=min_width+box_width/4,
                        max_height_crop=min_height+box_height/4,
                        min_width_crop=min_width,
                        min_height_crop=min_height)

    return img

# Use case: increase generalization; input image is not too bad.
vanilla_aug = iaa.OneOf([
    iaa.Grayscale(),
    iaa.imgcorruptlike.Pixelate(severity=(1, 2)),	
    iaa.imgcorruptlike.JpegCompression(severity=(1, 2)),	
    iaa.KMeansColorQuantization(n_colors=(80, 100)),	
    iaa.UniformColorQuantization(n_colors=(10, 15)),
    iaa.LinearContrast((1.2,1.7)),
    iaa.LogContrast(gain=(0.5, 1.5)),
    iaa.SigmoidContrast(gain=7, cutoff=(0.4, 0.5)),
    iaa.AdditivePoissonNoise((10, 15), per_channel=True),
    iaa.AdditivePoissonNoise((10, 15)),
    iaa.AdditiveLaplaceNoise(scale=(5, 15)),
    iaa.AdditiveLaplaceNoise(scale=(5, 15), per_channel=True),
    iaa.AdditiveGaussianNoise(scale=(20,45)),
    iaa.AdditiveGaussianNoise(scale=(20,45), per_channel=True),
    iaa.ChannelShuffle(p=1),
    iaa.imgcorruptlike.SpeckleNoise(severity=(1,2)),
    iaa.imgcorruptlike.DefocusBlur(severity=(1,2)),
    iaa.MotionBlur(k=(7, 10), angle=(-45, 45)),
    iaa.GaussianBlur((1.0, 3.0)),
    iaa.AverageBlur(k=(6, 8)),
    iaa.MedianBlur(k=(5, 7)),
    iaa.CoarseDropout((0.02,0.05), size_percent=(0.15,0.5)),
    iaa.CoarseDropout((0.02,0.05), size_percent=(0.15,0.5), per_channel=0.5),
    iaa.Multiply((0.5, 1.5)),
    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.Cutout(nb_iterations=(1, 2), size=0.15, squared=False, fill_mode=("constant", "gaussian"), 
                cval=(0, 255), fill_per_channel=0.5),
    iaa.Add((-40, 40)),
    iaa.Add((-40, 40), per_channel=0.5)
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
    rotation_matrix = np.array([
        [np.cos(rad_angle), -np.sin(rad_angle), 0.],
        [np.sin(rad_angle), np.cos(rad_angle), 0.],
        [0., 0., 1.]
    ], dtype=np.float64)
    
    center_old = np.array([img_width/2, img_height/2, 1], dtype=np.float64)
    center_new = rotation_matrix @ center_old
    rotate_offset = center_old-center_new
    rotate_offset = rotate_offset.reshape(3,)
    
    #################################################
    flip_matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    flip_offset = np.array([0, img_height, 0], dtype=np.float64)
    norm_trans = np.array([img_width/2, img_height/2, 0], dtype=np.float64)

    # shp = params[12:52].reshape(-1, 1)
    # exp = params[52:].reshape(-1, 1)
    # vertex = fm.bfm.reduced_generated_vertices(shp, exp)[fm.bfm.kpt_ind]

    # trans_v = vertex @ p.T + offset.reshape(3,) + norm_trans
    # flip_v = trans_v @ flip_matrix.T + flip_offset
    # rot_v = flip_v @ rotation_matrix.T + rotate_offset
    
    # new_p = (p.T @ flip_matrix.T @ rotation_matrix.T).T
    # new_offset = ((offset.reshape(3,) + norm_trans) @ flip_matrix.T + flip_offset) @ rotation_matrix.T + rotate_offset
    
    # new_all = vertex @ new_p.T + new_offset
    #################################################
    new_p = flip_matrix @ rotation_matrix @ flip_matrix @ p
    new_offset = (((offset.reshape(3,) + norm_trans) @ flip_matrix.T + flip_offset) @ rotation_matrix.T + rotate_offset - \
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

@numba.njit()
def random_crop_substep(img, roi_box, params, expand_ratio=None, target_size=None, radius=None):
    camera_matrix = params[:12].reshape(3, -1)

    trans = camera_matrix[:, 3]
    R1 = camera_matrix[0:1, :3]
    R2 = camera_matrix[1:2, :3]
    scale = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    rotation_matrix = np.concatenate((r1, r2, r3), 0)

    # Get the box that wrap all landmarks.
    box_left = roi_box[0]
    box_right = roi_box[2]
    box_top = roi_box[1]
    box_bot = roi_box[3]

    # Crop image to get the largest square region that satisfied:
    # 1. Contains all landmarks
    # 2. Center of the landmarks box is the center of the region.
    center = [(box_right+box_left)/2, (box_bot+box_top)/2]
    
    # Get the diameter of largest region 
    # that a landmark can reach when rotating.
    box_height = box_bot-box_top
    box_width = box_right-box_left

    if radius is None:
        radius = max(box_height, box_width) / 2

    max_length = 2*np.sqrt(2)*radius

    # Crop a bit larger.
    if expand_ratio is None:
        expand_ratio = random.uniform(0.8, 1.1)
    else:
        expand_ratio = expand_ratio

    crop_size = int(max_length/2 * expand_ratio)

    img_height, img_width, channel = img.shape
    canvas = np.zeros((img_height+2*crop_size, img_width+2*crop_size, channel), dtype=np.uint8)
    canvas[crop_size:img_height+crop_size, crop_size:img_width+crop_size, :] = img

    # shift_value = int(max_length/2 * expand_ratio - radius)
    '''
    0.125 is purely selected from visualization.
    '''
    # shift_value_x = int(box_width * 0.125 + shift_value)
    # shift_value_y = int(box_height * 0.125 + shift_value)
    # shift_value_x = shift_value
    # shift_value_y = shift_value

    # shift_x = random.randrange(-shift_value_x, shift_value_x)
    # shift_y = random.randrange(-shift_value_y, shift_value_y)

    # shift_x = shift_value_x
    # shift_y = shift_value_y

    center_x = int(center[0] + crop_size)
    center_y = int(center[1] + crop_size)

    # Top left bottom right.
    y1 = center_y-crop_size
    x1 = center_x-crop_size
    y2 = center_y+crop_size
    x2 = center_x+crop_size

    n_box_left = box_left + crop_size - x1
    n_box_right = box_right + crop_size - x1
    n_box_top = box_top + crop_size - y1
    n_box_bot = box_bot + crop_size - y1
    n_roi_box = [n_box_left, n_box_top, n_box_right, n_box_bot]

    cropped_img = canvas[y1:y2, x1:x2]

    flip_matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    flip_offset = np.array([0, img_height, 0], dtype=np.float64)
    norm_trans = np.array([img_width/2, img_height/2, 0], dtype=np.float64)

    cropped_trans = (flip_offset + np.array([-x1, -y1, 0])).reshape(3,) @ flip_matrix.T + norm_trans + trans

    if target_size is None:
        resized_scale = scale
        resized_trans = cropped_trans
    else:
        resized_scale = scale / (2*crop_size) * target_size
        resized_trans = cropped_trans / (2*crop_size) * target_size

    re_scaled_rot_matrix = resized_scale * rotation_matrix
    re_camera_matrix = np.concatenate((re_scaled_rot_matrix, resized_trans.reshape(-1,1)), axis=1)
    re_params = np.concatenate((re_camera_matrix.reshape(12,1), params[12:].reshape(-1,1)), axis=0)

    return cropped_img, re_params, n_roi_box


def random_crop(img, roi_box, params, expand_ratio=None, target_size=None, radius=None):
    '''
    Random crop and resize image to target size.
    '''
    cropped_img, re_params, n_roi_box = random_crop_substep(img, roi_box, params, expand_ratio, target_size, radius)

    if target_size is None:
        re_img = cropped_img
        re_roi_box = np.array(n_roi_box)
    else:
        re_img = cv2.resize(cropped_img, (target_size, target_size))
        re_roi_box = np.array(n_roi_box) / cropped_img.shape[0] * target_size
    # re_pts = fm.reconstruct_vertex(re_img, re_params)[fm.bfm.kpt_ind][:,:2]
    # draw_pts(re_img, re_pts)
    # import ipdb; ipdb.set_trace(context=10)

    return re_img, re_params, re_roi_box


@numba.njit()
def flip_substep(img, params):
    img_height, img_width = img.shape[:2]

    camera_matrix = params[:12].reshape(3, -1)

    trans = camera_matrix[:, 3]
    R1 = camera_matrix[0:1, :3]
    R2 = camera_matrix[1:2, :3]
    scale = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    rotation_matrix = np.concatenate((r1, r2, r3), 0)

    flip_matrix = np.array([[1,0,0],[0,-1,0],[0,0,1]], dtype=np.float64)
    flip_offset = np.array([0, img_height, 0], dtype=np.float64)

    norm_trans = np.array([img_width/2, img_height/2, 0], dtype=np.float64)

    flipped_rotation_matrix = flip_matrix @ (scale*rotation_matrix)
    flipped_trans = (trans + norm_trans) @ flip_matrix.T + flip_offset - norm_trans

    flipped_camera_matrix = np.concatenate((flipped_rotation_matrix, flipped_trans.reshape(-1,1)), axis=1)
    
    flipped_params = np.concatenate((flipped_camera_matrix.reshape(12,1), params[12:].reshape(-1,1)), axis=0)

    return flipped_params

def flip(img, params):
    flipped_params = flip_substep(img, params)
    flipped_img = cv2.flip(img, 0)

    return flipped_img, flipped_params