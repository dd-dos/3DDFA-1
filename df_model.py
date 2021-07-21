from pathlib import Path
import torch
from torch import nn
import torchvision
import mobilenet_v1
import logging
from utils import imutils
from utils import ddfa
import numpy as np
from PIL import Image
import cv2
logging.getLogger().setLevel(logging.INFO)
device = str(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

class DenseFaceModel:
    def __init__(self, model_path, detector_path, input_size=120):
        self.dense_face_model = mobilenet_v1.mobilenet_1()
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)[
            'state_dict'
        ]
        # self.dense_face_model = getattr(mobilenet_v1, 'mobilenet_1')(num_classes=62)
        model_dict = self.dense_face_model.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.dense_face_model.load_state_dict(model_dict)
        self.dense_face_model.eval()

        self.input_size = input_size
        normalize = ddfa.NormalizeGjz(mean=127.5, std=128)  # may need optimization

        self.transformer = torchvision.transforms.Compose([
            ddfa.ToTensorGjz(),
            normalize
        ])

        if device=='cpu':
            self.face_detector = torch.jit.load(detector_path)
        elif device=='cuda':
            self.face_detector = torch.jit.load(detector_path)

    @torch.no_grad()
    def get_landmarks(self, img, detected_faces=None):
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
        
        original_img = img.copy()
        if detected_faces is None:
            detected_faces = self.face_detector.forward(torch.tensor(img))[0]
            detected_faces = [det for det in detected_faces if det[-1] >= 0.9]

        if len(detected_faces) == 0:
            logging.warn("Warning: No faces were detected.")

        # Padding
        pad = int(max(img.shape)/4)
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        for det in detected_faces:
            det += pad        
        
        landmarks = []

        for idx, det in enumerate(detected_faces):
            cropped_inp, length, center, inp_shape = imutils.crop_balance(img, det, expand_ratio=1.2)
            inp = cv2.resize(cropped_inp, (self.input_size,self.input_size))
            base_inp = inp.copy()
            inp = self.transformer(inp)
            inp = inp.to(device)
            inp.unsqueeze_(0)

            out = self.dense_face_model(inp).squeeze(0)
            vertex = ddfa.reconstruct_vertex(out.numpy())
            # import ipdb; ipdb.set_trace(context=10)
            pts_img = vertex[:2].T
            pts_img = imutils.cropped_to_orginal(vertex[:2].T, length, center, self.input_size)

            # De-pad
            pts_img.T[0] = pts_img.T[0] - pad
            pts_img.T[1] = pts_img.T[1] - pad
            det -= pad

            landmarks.append(pts_img)
            # detected_faces[idx] = torch.tensor(inp_shape)

        return landmarks, original_img, detected_faces        

    def draw_landmarks(self, img, detected_faces=None):
        result = self.get_landmarks(img, detected_faces)
        if result is None:
            return img
            
        preds, img, detected_faces = result

        for idx, det in enumerate(detected_faces): 
            det = det.numpy().astype(int)
            cv2.rectangle(img,(det[0],det[1]),(det[2],det[3]),(255,255,255))
            landmark = preds[idx]
            for pts in landmark:
                pts = pts.astype(int)
                cv2.circle(img, pts, 2, (0,255,0), -1, 2)

        return img.astype(np.uint8)

def show_pts(img, pts):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    _img = img.copy()

    try:
        for i in range(pts.shape[0]):
            _pts = pts[i].astype(int)
            _img = cv2.circle(_img, (_pts[0], _pts[1]),3,(0,255,0), -1, 8)
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace(context=10)
    
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(_img).show()

def show_ndarray_img(img):
    _img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    Image.fromarray(_img).show()