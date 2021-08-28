from cv2 import CAP_PROP_PVAPI_PIXELFORMAT
import numpy as np

def crop_balance(img, detected_face, expand_ratio=1):
    # det = detected_face.numpy().copy()
    
    # box_width = det[2]-det[0]
    # box_height = det[3]-det[1]
    # # base_length = int(np.ceil(np.sqrt(np.power(box_height,2)+np.power(box_width,2))))
    # base_length = int(max(box_height, box_width))
    # length = base_length*expand_ratio

    # center = np.array([det[0]+(det[2]-det[0])/2, det[1]+(det[3]-det[1])/2])
        
    # x1 = int(center[0]-length/2)
    # x2 = int(center[0]+length/2)
    # y1 = int(center[1]-length/2)
    # y2 = int(center[1]+length/2)
    
    # # Crop image and adjust center
    # cropped_img = img[y1:y2, x1:x2]

    left, top, right, bottom, _ = detected_face
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.1)
    roi_box = [0] * 4
    roi_box[0] = int(center_x - size / 2)
    roi_box[1] = int(center_y - size / 2)
    roi_box[2] = int(roi_box[0] + size)
    roi_box[3] = int(roi_box[1] + size)

    cropped_img = img[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]

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