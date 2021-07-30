import logging
import math

import cv2
import numpy as np
import torch
import torchvision
from utils import ddfa, estimate_pose, imutils
import utils.render as render

import mobilenet_v1
from bfm import BFMModel

logging.getLogger().setLevel(logging.INFO)
device = str(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

class DenseFaceModel:
    def __init__(self, model_path, detector_path, bfm_fp):
        """
        Main class for processing dense face.

        Params:
        :model_path: path to pretrained 3DDFA model.
        :detector_path: path to jitted retinaface model.
        :bfm_fp: basel face model file path.
        """
        self.bfm = BFMModel(
            bfm_fp=bfm_fp
        )
        self.tri = self.bfm.tri

        self.dense_face_model = mobilenet_v1.mobilenet_1()
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)[
            'state_dict'
        ]
        model_dict = self.dense_face_model.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.dense_face_model.load_state_dict(model_dict)
        self.dense_face_model.eval()

        self.input_size = 120 # Input image size of 3DDFA model.
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
    def get_3dmm_params(self, img, detected_faces=None):
        """
        Get 3dmm parameters.

        Params:
        :img: input image.
        :detected_faces: list of detected faces. If not specified,
                        built-in face detector will be used.

        Returns:
        :params_list: list of 3dmm parameters corresponding to the
                      detected faces.
        :extras: extra data that might be needed to visualize image.
        """
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
        
        if detected_faces is None:
            detected_faces = self.face_detector.forward(torch.tensor(img))[0]
            detected_faces = [det for det in detected_faces if det[-1] >= 0.9]

        if len(detected_faces) == 0:
            logging.warn("Warning: No faces were detected.")

        # Padding
        pad = int(max(img.shape)/4)
        padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        for det in detected_faces:
            det += pad        
        
        params_list = []
        extras = []

        for idx, det in enumerate(detected_faces):
            cropped_inp, length, center = imutils.crop_balance(padded_img, det, expand_ratio=1.2)
            if max(cropped_inp.shape[0], cropped_inp.shape[1]) < 10: 
                continue
            try:
                inp = cv2.resize(cropped_inp, (self.input_size,self.input_size), interpolation=cv2.INTER_CUBIC)
            except:
                continue
            
            # cv2.imwrite('test.jpg', inp)
            inp = self.transformer(inp)
            inp = inp.to(device)
            inp.unsqueeze_(0)

            out = self.dense_face_model(inp).squeeze(0)
            params_list.append(out.numpy())
            extras.append(
                {
                    'length': length,
                    'center': center,
                    'pad': pad,
                    'detected_face': det-pad
                }
            )
        
        return params_list, extras

    def get_landmarks_and_angles(self, img, detected_faces=None, dense=False):
        """
        Get 68 3D landmarks.

        Params:
        :img: input image.
        :detected_faces: list of detected faces. If not specified,
                        built-in face detector will be used.

        Returns:
        :landmarks: 68 3D landmarks.
        :detected_faces: list of detected faces.
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        params_list, extras = self.get_3dmm_params(img, detected_faces)
        landmarks = []
        angles_list = []

        if detected_faces == None:
            detected_faces = []

        for idx, params in enumerate(params_list):
            center = extras[idx]['center']
            length = extras[idx]['length']
            pad = extras[idx]['pad']

            if dense:
                vertex = self.bfm.reconstruct_vertex(params, dense=dense)
            else:
                vertex = ddfa.reconstruct_vertex(params, dense=dense)

            pts_img = imutils.cropped_to_orginal(vertex, length, center, self.input_size)

            # De-pad
            pts_img[0] -= pad
            pts_img[1] -= pad

            landmarks.append(pts_img.T)
            detected_faces.append(extras[idx]['detected_face'])

            _, pose = estimate_pose.parse_pose(params)

            angles_list.append({
                'yaw': pose[0] / math.pi * 180, 
                'pitch': pose[1] / math.pi * 180,
                'roll': pose[2] / math.pi * 180,
            })

        return landmarks, detected_faces, angles_list

    def draw_landmarks(self, img, detected_faces=None):
        """
        Draw landmarks to image.
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        result = self.get_landmarks_and_angles(img, detected_faces)
        if result is None:
            return img
            
        preds, detected_faces, angles_list = result

        for idx, det in enumerate(detected_faces): 
            angles = angles_list[idx]
            det = det.numpy().astype(int)
            cv2.rectangle(img, (det[0],det[1]),(det[2],det[3]),(255,255,255))
            cv2.putText(img=img, 
                        text=f'face{idx}: yaw: {angles["yaw"]:.2f} pitch: {angles["pitch"]:.2f} roll: {angles["roll"]:.2f}',
                        org=(50+idx*50,50+idx*50), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=(255,0,0),
                        thickness=1,
                        lineType=cv2.LINE_AA)
            cv2.putText(img=img, 
                        text=f'face: {idx}',
                        org=(det[0], det[1]), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75,
                        color=(0,255,0),
                        thickness=1,
                        lineType=cv2.LINE_AA)
            landmark = preds[idx]
            for pts in landmark:
                pts = pts.astype(int)
                cv2.circle(img, pts[:2], 2, (0,255,0), -1, 2)

        return img.astype(np.uint8) 

    def get_rotate_angles_list(self, img, detected_faces=None):
        """
        Get euler angles_list of face(s) in image.

        Params:
        :img: input image.
        :detected_faces: list of detected faces. If not specified,
                        built-in face detector will be used.
        """
        params_list, extras = self.get_3dmm_params(img, detected_faces)
        result = []

        for idx, params in enumerate(params_list):
            _, pose = estimate_pose.parse_pose(params)

            result.append({
                'detected_face': extras[idx]['detected_face'],
                'yaw': pose[0],
                'pitch': pose[1],
                'roll': pose[2],
            })

        return result

    def draw_mesh(self, img, detected_faces=None):
        if isinstance(img, str):
            img = cv2.irmead(img)

        ver_lst, detected_faces, angles_list = self.get_landmarks_and_angles(img, detected_faces, dense=True)
        ver_lst = [ver.T for ver in ver_lst]
        res = render(img, ver_lst, self.tri)
        return res