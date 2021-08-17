import numpy as np

def crop_balance(img, det, expand_ratio=1):
    box_width = det[2]-det[0]
    box_height = det[3]-det[1]
    base_length = int(np.ceil(np.sqrt(np.power(box_height,2)+np.power(box_width,2))))
    # base_length = int(max(box_height, box_width))
    length = base_length*expand_ratio

    center = np.array([det[0]+(det[2]-det[0])/2, (det[1]+(det[3]-det[1])/2)*1.05])
        
    x1 = int(center[0]-length/2)
    x2 = int(center[0]+length/2)
    y1 = int(center[1]-length/2)
    y2 = int(center[1]+length/2)
    
    # Crop image and adjust center
    cropped_img = img[y1:y2, x1:x2]

    return cropped_img, length, center

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


def check_close_eye(eye, threshold=0.2):
    p2_minus_p6 = np.linalg.norm(eye[1] - eye[5])
    p3_minus_p5 = np.linalg.norm(eye[2] - eye[4])
    p1_minus_p4 = np.linalg.norm(eye[0] - eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
 
    if ear <= threshold:
        return True
    else:
        return False
 
def get_eyes(pts):
    left = pts[36:42]
    right = pts[42:48]

    return {
        'left': left,
        'right': right
    }
