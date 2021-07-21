import numpy as np

def crop_balance(img, detected_face, expand_ratio=1):
    img_height, img_width, _ = img.shape
    det = detected_face.numpy().copy()
    
    box_width = det[3]-det[1]
    box_height = det[2]-det[0]
    # base_length = int(np.ceil(np.sqrt(np.power(box_height,2)+np.power(box_width,2))))
    base_length = int(max(box_height, box_width))
    length = base_length*expand_ratio

    center = np.array([det[0]+(det[2]-det[0])/2, det[1]+(det[3]-det[1])/2])
        
    x1 = int(center[0]-length/2)
    x2 = int(center[0]+length/2)
    y1 = int(center[1]-length/2)
    y2 = int(center[1]+length/2)
    
    # Crop image and adjust center
    cropped_img = img[y1:y2, x1:x2]
    inp_shape = np.array([x1, y1, x2, y2])

    return cropped_img, length, center, inp_shape

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
    coord_original_cropped_pts.T[0] += center[0]-length/2
    coord_original_cropped_pts.T[1] += center[1]-length/2

    return coord_original_cropped_pts