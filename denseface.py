import logging
import math
import os
import shutil

import cv2
import numpy as np
import torch
import torchvision

import mobilenet_v1
import mobilenet_v2
from utils import estimate_pose, imutils
from utils.face3d import face3d
from utils.face3d.face3d.utils import *

logging.getLogger().setLevel(logging.INFO)

class FaceAlignment:
    def __init__(self, 
                 model_path, 
                 landmarks_type='3D',
                 device='cuda',
                 max_batch_size=8,
                 expand_ratio=1.2,
                 input_size=120,
                 num_classes=62,
                 backbone='mobilenet_v2',
                 arch='mobilenet_1',
                 params_mean_std='',
                 debug=False):
        """
        Main class for processing dense face.

        Params:
        :model_path: path to pretrained 3DDFA model.
        :landmarks_type: '2D' or '3D'.
            '2D': 68 landmarks in xy coordinates. 
            '3D': 68 landmarks in xyz coordinates.
        :device: cpu or cuda.
        :max_batch_size: maximum batch size for batch processing. 
            If -1 is passed, use all inputs as a batch.
        :expand_ratio: ratio to expand image.
        :input_size: input image size.
        """
        self.fm = face3d.face_model.FaceModel(params_mean_std)

        if backbone == 'mobilenet_v1':
            self.dense_face_model = getattr(mobilenet_v1, arch)(num_classes=num_classes)
        elif backbone == 'mobilenet_v2':
            self.dense_face_model = getattr(mobilenet_v2, arch)(num_classes=num_classes)

        checkpoint = torch.load(model_path, map_location=device)['state_dict']
        model_dict = self.dense_face_model.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.dense_face_model.load_state_dict(model_dict)
        # self.dense_face_model = torch.jit.load('model.pt')
        self.dense_face_model.eval()

        # for param in self.dense_face_model.parameters():
        #     param.requires_grad = False
        # self.dense_face_model.fc = torch.nn.Linear(1024, 89)

        self.transformer = torchvision.transforms.Compose([
            imutils.ToTensorGjz(),
            imutils.NormalizeGjz(mean=127.5, std=128)
        ])

        self.device = device
        self.input_size = input_size
        self.landmarks_type = landmarks_type
        self.expand_ratio = expand_ratio
        self.max_batch_size = max_batch_size
        self.debug = debug
        if self.debug:
            shutil.rmtree('debug', ignore_errors=True)
            os.makedirs('debug', exist_ok=True)
            

    @torch.no_grad()
    def get_3dmm_params(self, image_or_path, detected_faces):
        """
        Get 3dmm parameters.

        Params:
        :image_or_path: input image.
        :detected_faces: list of detected faces. If not specified,
                        built-in face detector will be used.

        Returns:
        :params_list: list of 3dmm parameters corresponding to the
                      detected faces.
        :extra_list: extra data that might be needed to visualize image.
        """
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path, cv2.IMREAD_COLOR)
        else:
            img = image_or_path
        
        # Padding
        pad = int(max(img.shape)/4)
        padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        for det in detected_faces:
            det += pad        
        
        params_list = []
        extra_list = []

        for idx, det in enumerate(detected_faces):
            cropped_inp, length, center = imutils.crop_balance(padded_img, det, expand_ratio=self.expand_ratio)
            inp = cv2.resize(cropped_inp, (self.input_size,self.input_size), interpolation=cv2.INTER_CUBIC)

            if self.debug:
                ori_inp = inp.copy()
                cv2.imwrite('debug/inp.jpg', inp)

            inp = self.transformer(inp)
            inp = inp.to(self.device)
            inp.unsqueeze_(0)

            out = self.dense_face_model(inp).squeeze(0)
            vertex = self.fm.reconstruct_vertex(ori_inp, out.numpy())[self.fm.bfm.kpt_ind]

            if self.debug:
                for i in range(vertex[:,:2].shape[0]):
                    _pts = vertex[:,:2][i].astype(int)
                    _img = cv2.circle(ori_inp, (_pts[0], _pts[1]),1,(0,255,0), -1, 8)
                cv2.imwrite('debug/out.jpg', _img)

            params_list.append(out.numpy().reshape(-1,))
            extra_list.append(
                {
                    'length': length,
                    'center': center,
                    'pad': pad,
                }
            )
            det -= pad
        
        return params_list, extra_list

    @torch.no_grad()
    def draw_landmarks(self, img, detected_faces, draw_eyes=False, draw_angles=False, no_background=True, connected=True):
        """
        Draw landmarks to image.
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        result = self.get_landmarks_and_angles(img, detected_faces)
        if result is None:
            return img
            
        pts = result['landmarks']
        angles_list = result['rotation_angle']

        if no_background:
            img = np.zeros(img.shape, dtype=np.uint8)

        for idx, det in enumerate(detected_faces): 
            angles = angles_list
            det = det.numpy().astype(int)
            cv2.rectangle(img, (det[0],det[1]),(det[2],det[3]),(255,255,255))
            if draw_angles:
                cv2.putText(img=img, 
                            text=f'face{idx}: yaw: {angles[2]:.2f} pitch: {angles[0]:.2f} roll: {angles[1]:.2f}',
                            org=(50+idx*50,50+idx*50), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(255,0,0),
                            thickness=1,
                            lineType=cv2.LINE_AA)
                print(f'face{idx}: yaw: {angles[2]:.2f} pitch: {angles[0]:.2f} roll: {angles[1]:.2f}')
                cv2.putText(img=img, 
                            text=f'face: {idx}',
                            org=(det[0], det[1]), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(0,255,0),
                            thickness=1,
                            lineType=cv2.LINE_AA)

            if connected:
                landmarks = pts[idx][self.fm.bfm.kpt_ind].astype(int)

                for pts in landmarks:
                    cv2.circle(img, pts[:2], 2, (0,255,0), -1, 2)

                chin = [(landmarks[i], landmarks[i+1]) for i in range(16)]
                right_eyebrow = [(landmarks[i], landmarks[i+1]) for i in range(17, 21)]
                left_eyebrow = [(landmarks[i], landmarks[i+1]) for i in range(22, 26)]
                right_eye = [(landmarks[i], landmarks[i+1]) for i in range(36, 41)]
                left_eye = [(landmarks[i], landmarks[i+1]) for i in range(42, 47)]
                nose = [(landmarks[i], landmarks[i+1]) for i in range(27, 30)]
                nose_hole = [(landmarks[i], landmarks[i+1]) for i in range(31, 35)]
                outer_lip = [(landmarks[i], landmarks[i+1]) for i in range(48, 59)]
                inner_lip = [(landmarks[i], landmarks[i+1]) for i in range(60, 67)]

                other = [(landmarks[41], landmarks[36]), 
                        (landmarks[47], landmarks[42]),
                        (landmarks[59], landmarks[48]),
                        (landmarks[67], landmarks[60])]

                all = other + chin + right_eyebrow + left_eyebrow + right_eye + left_eye + nose + nose_hole + outer_lip + inner_lip

                for pt in all:
                    cv2.line(img, pt[0], pt[1], (0,255,0), 2)                    
            else:
                landmarks = pts[idx][self.fm.bfm.kpt_ind].astype(int)

                for pts in landmarks:
                    cv2.circle(img, pts[:2], 2, (0,255,0), -1, 2)

            if draw_eyes:
                eyes_dict = imutils.get_eyes(landmarks)
                if imutils.check_close_eye(eyes_dict['left']) and imutils.check_close_eye(eyes_dict['right']):
                    eyes = 'close'
                else:
                    eyes = 'open'
                cv2.putText(img=img, 
                            text=f'eyes: {eyes}',
                            org=(det[0], det[1]), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.75,
                            color=(0,255,0),
                            thickness=1,
                            lineType=cv2.LINE_AA)

        return img.astype(np.uint8) 

    def get_landmarks_and_angles(self, image_or_path, detected_faces):
        """
        Get landmarks of image and head pose angle.

        Return:
        dictionary: {
            :landmarks: 68 points landmarks,
            :rotation_angle: Euler angles,
        }
        """
        landmarks, angles = \
        self.get_landmarks_and_angles_from_image(
            image_or_path, detected_faces
        )

        if self.landmarks_type == '2D':
            landmarks = [landmarks[i][:2] for i in range(len(landmarks))]

        rotation_angles = [
            angles[0]['pitch'],
            angles[0]['roll'],
            angles[0]['yaw']
        ]

        meta = {
            'landmarks': landmarks,
            'rotation_angle': rotation_angles,
        }

        return meta

    def get_landmarks_and_angles_from_image(self, image_or_path, detected_faces):
        """
        Predict 68 landmarks points of an image.
        If no bounding box is passed, this function will use built-in 
        retinaface model.

        Params:
        :image_or_path: a single image. Accept both ndarray and path.
        :detected_faces: list of faces in image. If specified, 
                         please use the format [top, left, bottom, right, confidence].
        
        Return:
        :landmarks: 68 landmarks points.
        :angles: euler angles.
        """
        if detected_faces is None or len(detected_faces) == 0:
            logging.warn("No faces were detected.")
            return None
        
        params_list, extra_list = self.get_3dmm_params(image_or_path, detected_faces)
        landmarks = []
        angles = []

        for idx, params in enumerate(params_list):
            center = extra_list[idx]['center']
            length = extra_list[idx]['length']
            pad = extra_list[idx]['pad']

            vertex = self.fm.reconstruct_vertex(np.zeros((self.input_size,self.input_size,3)), params)[:,:2].T

            pts_img = imutils.cropped_to_orginal(vertex, length, center, self.input_size)

            # De-pad
            pts_img[0] -= pad
            pts_img[1] -= pad

            landmarks.append(pts_img.T)

            # _, pose = estimate_pose.parse_pose(params)
            _,_,_,pose,_= self.fm._parse_params(params)
            # pose = (0,0,0)
            # angles.append({
            #     'yaw': pose[0] / math.pi * 180, 
            #     'pitch': pose[1] / math.pi * 180,
            #     'roll': pose[2] / math.pi * 180,
            # })
            angles.append({
                'yaw': pose[1], 
                'pitch': pose[0],
                'roll': pose[2],
            })

        return landmarks, angles
