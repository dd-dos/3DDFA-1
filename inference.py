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
    P.add_argument('--flip', action='store_true')
    P.add_argument('--squeeze', type=int, default=0)
    P.add_argument('--expand-ratio', type=float, default=1.1)
    P.add_argument('--backbone', type=str, default='mobilenet_v2')
    P.add_argument('--arch', type=str, default='mobilenet_2')
    P.add_argument('--params-mean-std', type=str, default='snapshot/2021-09-08-1/params_mean_std_12_pose_60_shp_29_exp.mat')

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
        expand_ratio=args.expand_ratio,
        backbone=args.backbone,
        arch=args.arch,
        params_mean_std=args.params_mean_std)
    # pose_model = facelib.models.PoseModel(args.model_path, img_size=size)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.squeeze != 0:
            frame = cv2.copyMakeBorder(frame, 0, 0, args.squeeze, args.squeeze, cv2.BORDER_CONSTANT, 0)
            frame = cv2.resize(frame, (frame_width, frame_height))

        if args.flip:
            frame = cv2.flip(frame, 0)

        detector_info = face_detector.forward(torch.tensor(frame))
        detected_faces = detector_info[0]
        detected_faces = [det for det in detected_faces if det[-1] >= 0.9]
        if len(detected_faces) == 0:
            print("No face detected!")
            cv2.imshow('', frame)
            cv2.waitKey(1)
            continue

        import time
        key = cv2.waitKey(1) & 0xFF

        t0 = time.time()
        try:
            frame = \
                dense_model.draw_landmarks(
                    frame, 
                    detected_faces,
                    draw_eyes=False,
                    no_background=False,
                    draw_angles=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
        print(time.time()-t0)
        # processed_frame = dense_model.draw_mesh(frame)
        # angles_dict = dense_model.get_rotate_angles(img, detected_faces)
        logging.info(f'Landmarks detection took {time.time() - t0}')
     
        if save_video:
            result.write(frame)

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
        num_classes=args.num_classes,
        expand_ratio=1.1)

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
    