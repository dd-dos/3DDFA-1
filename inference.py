import argparse
import logging
from pathlib import Path
import torch

import cv2
import numpy as np
import torch

from denseface import FaceAlignment

logging.getLogger().setLevel(logging.INFO)

def argparser():
    P = argparse.ArgumentParser(description='Test landmarks')
    P.add_argument('--model-path', type=str, default='snapshot/phase1_wpdc_best.pth.tar', help='Path to model weight')
    P.add_argument('--detector-path', type=str, default='retinaface/scripted_model_cpu_19042021.pt', help='Ratio to expand input')
    P.add_argument('--mode', type=str, default='video', help='testing mode: video, img')
    P.add_argument('--video-path', type=str, default=None, help='path to input video')
    P.add_argument('--img-path', type=str, default=None, help='path to input image')
    P.add_argument('--folder-path', type=str, default=None, help='path to input folder')
    P.add_argument('--save-path', type=str, default=None, help='path to save result')
    P.add_argument('--config_path', type=str, help='config path')
    P.add_argument('--bfm-fp', type=str, default='configs/bfm_noneck_v3.pkl')
    P.add_argument('--input-size', type=int, default='120')
    P.add_argument('--num-classes', type=int, default='62')
    
    args = P.parse_args()

    return args

@torch.no_grad()
def test_video(args):
    """
    Detect 68 landmarks of a video. If no video is specified, 
    using built-in camera instead.
    """
    if args.video_path is not None:
        path = Path(args.video_path)
        if path.exists():
            cap = cv2.VideoCapture(str(path))
        else:
            cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    
    save_video = False
    if args.save_path is not None:
        if args.save_path.endswith('avi') or args.save_path.endswith('mp4'):
            result = cv2.VideoWriter(args.save_path, 
                                    cv2.VideoWriter_fourcc(*'MJPG'),
                                    10, size)
            save_video = True
        else:
            logging.error(f'Invalid save path: {path}.')
            save_video = False

    face_detector = torch.jit.load('retinaface/scripted_model_cpu_19042021.pt')

    dense_model = FaceAlignment(
        model_path=args.model_path, 
        input_size=args.input_size, 
        device='cpu', 
        num_classes=args.num_classes,
        expand_ratio=1.2)
    # pose_model = facelib.models.PoseModel(args.model_path, img_size=size)
    
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (max(frame_width, frame_height), max(frame_width, frame_height)))
        if not ret:
            break
        
        # frame = cv2.flip(frame, 0)
        detector_info = face_detector.forward(torch.tensor(frame))
        detected_faces = detector_info[0]
        foo_lms = detector_info[1]
        detected_faces = [det for det in detected_faces if det[-1] >= 0.9]
        print(f'frame shape: {frame.shape}')
        print(f'detected faces: {detected_faces}')

        # for landmarks in foo_lms:
        #     # points = landmarks.reshape((2,5)).T
        #     for idx in range(5):
        #         pts = (int(landmarks[idx].item()), int(landmarks[5+idx].item()))
        #         cv2.circle(frame, pts, 2, (0,255,0), -1, 2)

        import time
        key = cv2.waitKey(1) & 0xFF

        t0 = time.time()
        # frame = \
        #     dense_model.draw_landmarks(
        #         frame, 
        #         detected_faces,
        #         draw_eyes=False,
        #         no_background=False,
        #         draw_angles=True)
        try:
            frame = \
                dense_model.draw_landmarks(
                    frame, 
                    detected_faces,
                    draw_eyes=False,
                    no_background=False,
                    draw_angles=True)
        except Exception as e:
            print(e)
        print(time.time()-t0)
        # processed_frame = dense_model.draw_mesh(frame)
        # angles_dict = dense_model.get_rotate_angles(img, detected_faces)
        logging.info(f'Landmarks detection took {time.time() - t0}')
     
        if save_video:
            result.write(frame)
        # frame = model.get_head_pose(frame)

        cv2.imshow('', frame)
        # cv2.waitKey(0)

        if key == ord('q'):
            break

    cap.release()
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")

def test_image(args):
    img = cv2.imread(args.img_path)

    face_detector = torch.jit.load('retinaface/scripted_model_cpu_19042021.pt')
    if img.shape[0] > 128:
        detected_faces = face_detector.forward(torch.tensor(img))[0]
        detected_faces = [det for det in detected_faces if det[-1] >= 0.9]
    else:
        detected_faces = [torch.tensor([0, 0, img.shape[0], img.shape[1]])]

    dense_model = FaceAlignment(
        model_path=args.model_path, 
        input_size=args.input_size, 
        device='cpu', 
        num_classes=args.num_classes,)

    # processed_frame = dense_model.draw_landmarks(
    #     args.img_path, 
    #     detected_faces=torch.tensor([[0,0,img.shape[0], img.shape[0]]])
    # )
    processed_frame = dense_model.draw_landmarks(
        args.img_path, 
        detected_faces=detected_faces,
        draw_angles=False,
        no_background=False
    )

    cv2.imshow('', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def test_full(args):
#     """
#     Detect 68 3D landmarks of an image.
#     """
#     if args.img_path is not None:
#         path = Path(args.img_path)
#         if path.exists():
#             img = cv2.imread(args.img_path)
#             height, width, _ = img.shape
#             model = facelib.PoseModel(args.model_path,
#                                   img_size=(height, width))
#             pose_image = model.draw_pose(img)
#             cv2.imshow('', pose_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()

if __name__=="__main__":
    args = argparser()

    if args.mode=='video':
        test_video(args)
    elif args.mode == 'img':
        test_image(args)
    elif args.mode == 'folder':
        img_list = list(Path(args.folder_path).glob('**/*.jpg'))
        for img_path in img_list:
            args.img_path = str(img_path)
            test_image(args)
    else:
        logging.error(f'Invalid mode {args.mode}')
    