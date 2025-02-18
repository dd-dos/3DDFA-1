import logging
import math

import cv2
import numpy as np
import torch
import torchvision
from utils import estimate_pose, imutils, ddfa

import mobilenet_v1

logging.getLogger().setLevel(logging.INFO)

class FaceAlignment:
    def __init__(self, 
                 model_path, 
                 landmarks_type='3D',
                 device='cpu',
                 max_batch_size=8,
                 expand_ratio=1.2,
                 input_size=120,
                 num_classes=62):
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
        self.dense_face_model = mobilenet_v1.mobilenet_1(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)['state_dict']
        model_dict = self.dense_face_model.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.dense_face_model.load_state_dict(model_dict)
        self.dense_face_model.eval()

        # for param in self.dense_face_model.parameters():
        #     param.requires_grad = False
        # self.dense_face_model.fc = torch.nn.Linear(1024, 89)

        self.transformer = torchvision.transforms.Compose([
            ddfa.ToTensorGjz(),
            ddfa.NormalizeGjz(mean=127.5, std=128)
        ])

        self.device = device
        self.input_size = input_size
        self.landmarks_type = landmarks_type
        self.expand_ratio = expand_ratio
        self.max_batch_size = max_batch_size

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
            # show_ndarray_img(inp)
            # import ipdb; ipdb.set_trace(context=10)
            # ori_inp = inp.copy()
            cv2.imwrite('inp.jpg', inp)
            inp = self.transformer(inp)
            inp = inp.to(self.device)
            inp.unsqueeze_(0)

            out = self.dense_face_model(inp).squeeze(0)
            # import ipdb; ipdb.set_trace(context=10)
            # vertex = fm.reconstruct_vertex(ori_inp, out.numpy())[fm.bfm.kpt_ind]
            # show_pts(ori_inp, vertex[:,:2])
            # import ipdb; ipdb.set_trace(context=10)

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
    def get_batch_3dmm_params(self, image_or_path_list, detected_faces_list):
        image_batch = torch.zeros(len(detected_faces_list), 3, self.input_size, self.input_size, device=self.device)
        extra_list = []

        for idx, image_or_path in enumerate(image_or_path_list):
            if isinstance(image_or_path, str):
                img = cv2.imread(image_or_path)
            else:
                img = image_or_path

            detected_faces = detected_faces_list[idx]

            if detected_faces is None or len(detected_faces) == 0:
                logging.warn(f"No faces were detected at image {idx}th.")
                continue

            if len(detected_faces) > 1:
                logging.warn(f"There are more than 1 face in image. Please check image {idx}th")
            
            pad = int(max(img.shape)/4)
            padded_img = cv2.copyMakeBorder(
                img, pad, pad, pad, pad, 
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            for det in detected_faces:
                det += pad

            cropped_inp, length, center = \
            imutils.crop_balance(
                padded_img, 
                detected_faces[0], 
                expand_ratio=self.expand_ratio
            )
            
            inp = cv2.resize(cropped_inp, (self.input_size,self.input_size), interpolation=cv2.INTER_CUBIC)
            inp = self.transformer(inp)
            inp = inp.to(self.device)

            image_batch[idx] = inp
            extra_list.append(
                {
                    'length': length,
                    'center': center,
                    'pad': pad,
                }
            )
        
        image_batch = image_batch.to(self.device)
        if self.max_batch_size == -1:
            params_batch = self.dense_face_model(image_batch).cpu()
            empty_cache_memory()
        else:
            div_batch = len(image_batch) // self.max_batch_size
            mod_batch = len(image_batch) % self.max_batch_size
            all_result = []
            for index in range(div_batch):
                sub_batch_tensor = image_batch[index*self.max_batch_size:
                                                (index+1)*self.max_batch_size, :, :, :]
                results = self.dense_face_model(sub_batch_tensor).cpu()
                all_result.extend(results)
                empty_cache_memory()

            if mod_batch > 0:
                sub_batch_tensor = image_batch[-mod_batch:, :, :, :]
                results = self.dense_face_model(sub_batch_tensor).cpu()

                all_result.extend(results)
                empty_cache_memory()
            
            params_batch = torch.stack(all_result, dim=0)

        return params_batch.numpy(), extra_list

    @torch.no_grad()
    def draw_landmarks(self, img, detected_faces, draw_eyes=False, draw_angles=False, no_background=True):
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

            landmarks = pts[idx]

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

            for pts in landmarks:
                pts = pts.astype(int)
                cv2.circle(img, pts[:2], 2, (0,255,0), -1, 2)

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

    def get_batch_landmarks_and_angles(self, image_or_path_list, detected_faces_list):
        """
        Get landmarks and head pose angle of an image list.

        TODO: utilize numba.
        """
        landmarks_list, angles_list = \
        self.get_batch_landmarks_and_angles_from_images(
            image_or_path_list, detected_faces_list
        )

        if self.landmarks_type == '2D':
            landmarks_list = [landmarks_list[i][:2] for i in range(len(landmarks_list))]
        
        for i in range(len(angles_list)):
            angles_list[i] = [
                angles_list[i]['pitch'],
                angles_list[i]['roll'],
                angles_list[i]['yaw'],
            ] 

        meta = {
            'landmarks': landmarks_list,
            'rotation_angles': angles_list,
        }

        # for idx in range(len(detected_faces_list)):
        #     pts = landmarks_list[idx][:, :2]
        #     img = image_or_path_list[idx]
            
        #     for _pts in pts:
        #         cv2.circle(img, _pts.astype(int), 3, (0,255,0), -1, 10)

        #     cv2.imwrite(f'samples/{idx}.jpg', img)

        return meta

    def get_landmarks(self, image_or_path, detected_faces):
        """
        Predict 68 landmarks points of an image.
        If no bounding box is passed, this function will use built-in 
        retinaface model.

        Params:
        :image_or_path: a single image. Accept both ndarray and path.
        :detected_faces: list of faces in image. If specified, 
                        please use the format [top, left, bottom, right, confidence].
        
        Return:
        :landmarks: 68 landmarks points,
        """
        landmarks = self.get_landmarks_from_image(image_or_path, detected_faces)

        return landmarks

    def get_batch_landmarks(self, image_or_path_list, detected_faces_list):
        """
        Predict the landmarks for each face present in the image.
        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detected_faces is None the method will also run a face detector.

        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.
        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})

        Return:
        :landmarks_list: list of 68 points landmarks
        """
        landmarks_list = self.get_batch_landmarks_from_images(image_or_path_list, detected_faces_list)

        return landmarks_list

    def get_landmarks_from_image(self, image_or_path, detected_faces):
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
        """
        if detected_faces is None or len(detected_faces) == 0:
            logging.warn("No faces were detected.")
            return 
        
        params_list, extra_list = self.get_3dmm_params(image_or_path, detected_faces)
        landmarks = []

        for idx, params in enumerate(params_list):
            center = extra_list[idx]['center']
            length = extra_list[idx]['length']
            pad = extra_list[idx]['pad']

            vertex = ddfa.reconstruct_vertex(params)
            # vertex = fm.reconstruct_vertex(
            #     np.zeros((self.input_size,self.input_size,3)), 
            #     params
            # )[fm.bfm.kpt_ind].T

            pts_img = imutils.cropped_to_orginal(vertex, length, center, self.input_size)

            # De-pad
            pts_img[0] -= pad
            pts_img[1] -= pad

            landmarks.append(pts_img.T)

        return landmarks

    def get_batch_landmarks_from_images(self, image_or_path_list, detected_faces_list):
        """
        Predict the landmarks for each face present in the image.
        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detected_faces is None the method will also run a face detector.
        
        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.
        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        
        Return:
        :landmarks_list: list of 68 points landmarks.
        """
        landmarks_list = []

        params_batch, extra_list = \
        self.get_batch_3dmm_params(
            image_or_path_list, 
            detected_faces_list
        )

        for idx in range(len(extra_list)):
            params = params_batch[idx]
            extra = extra_list[idx]

            center = extra[idx]['center']
            length = extra[idx]['length']
            pad = extra[idx]['pad']

            vertex = ddfa.reconstruct_vertex(params)

            pts_img = imutils.cropped_to_orginal(
                vertex, length, center, self.input_size
            )

            # De-pad
            pts_img[0] -= pad
            pts_img[1] -= pad

            landmarks_list.append(pts_img.T)
        
        return landmarks_list

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

            vertex = ddfa.reconstruct_vertex(params)

            pts_img = imutils.cropped_to_orginal(vertex, length, center, self.input_size)

            # De-pad
            pts_img[0] -= pad
            pts_img[1] -= pad

            landmarks.append(pts_img.T)

            _, pose = estimate_pose.parse_pose(params)
            # _,_,_,pose,_= fm._parse_params(params)
            # pose = (0,0,0)
            angles.append({
                'yaw': pose[0] / math.pi * 180, 
                'pitch': pose[1] / math.pi * 180,
                'roll': pose[2] / math.pi * 180,
            })
            # angles.append({
            #     'yaw': pose[0], 
            #     'pitch': pose[1],
            #     'roll': pose[2],
            # })

        return landmarks, angles
    
    def get_batch_landmarks_and_angles_from_images(self, image_or_path_list, detected_faces_list):
        """
        Predict the landmarks for each face present in the image.
        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detected_faces is None the method will also run a face detector.
        
        Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.
        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        
        Return:
        :landmarks_list: list of 68 points landmarks.
        :angles_list: list of euler angles.
        """
        landmarks_list = []
        angles_list = []

        params_batch, extra_list = \
        self.get_batch_3dmm_params(
            image_or_path_list, 
            detected_faces_list
        )

        for idx in range(len(extra_list)):
            params = params_batch[idx]
            extra = extra_list[idx]

            center = extra['center']
            length = extra['length']
            pad = extra['pad']

            # vertex = ddfa.reconstruct_vertex(params)
            vertex = fm.reconstruct_vertex(np.zeros((256,256,3)), params)[fm.bfm.kpt_ind]

            pts_img = imutils.cropped_to_orginal(
                vertex, length, center, self.input_size
            )

            # De-pad
            pts_img[0] -= pad
            pts_img[1] -= pad

            landmarks_list.append(pts_img.T)

            _, pose = estimate_pose.parse_pose(params)
            angles_list.append({
                'yaw': pose[0] / math.pi * 180, 
                'pitch': pose[1] / math.pi * 180,
                'roll': pose[2] / math.pi * 180,
            })
        
        return landmarks_list, angles_list
