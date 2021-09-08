import numpy as np
import torch

def crop_balance(img, detected_face, expand_ratio=1, shift=0.):
    if len(detected_face) == 5:
        left, top, right, bottom, _ = detected_face
    else:
        left, top, right, bottom = detected_face

    # old_size = (right - left + bottom - top) / 2.
    old_size = max(right-left, bottom-top)
    center_x = right - (right - left) / 2.
    center_y = bottom - (bottom - top) / 2. + old_size * shift
    size = int(old_size * expand_ratio)
    roi_box = [0] * 4
    roi_box[0] = int(center_x - size / 2)
    roi_box[1] = int(center_y - size / 2)
    roi_box[2] = int(roi_box[0] + size)
    roi_box[3] = int(roi_box[1] + size)

    cropped_img = img[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]

    '''
    drake
    '''
    # left = int(left)
    # top = int(top)
    # right = int(right)
    # bottom = int(bottom)

    # image_width, image_height = img.shape[1], img.shape[0]
    # width, height = right - left, bottom - top
    # expand_ratio -= 1
    # padding_height = int(height * expand_ratio)
    # padding_width = int(width * expand_ratio)

    # left = max(0, left - padding_width)
    # right = min(image_width, right + padding_width)

    # top = max(0, top - padding_height)
    # bottom = min(image_height, bottom + padding_height)

    # center_x = int((left + right)/2)
    # center_y = int((bottom + top)/2)

    # cropped_img = img[top:bottom, left:right]
    # import cv2
    # cv2.imwrite("xxx.png", cropped_img)
    return cropped_img, size, np.array([center_x, center_y])

def cropped_to_orginal(pts, length, center, resize):
    """
    Get original coordinate of pts inside a cropped image.

    Params:
    :pts: list of points inside cropped images.
    :length: we expect the original image to be a square one,
            this value is the size of the image.
    :center: center of the cropped image relative to the original one.
    :resize: resize size if the cropped image is resized.
    """
    coord_original_cropped_pts = pts / resize * length
    coord_original_cropped_pts[0] += center[0]-length/2
    coord_original_cropped_pts[1] += center[1]-length/2

    return coord_original_cropped_pts


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor
