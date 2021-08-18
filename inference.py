import argparse
import logging
import sys
from pathlib import Path
import cv2
from tddfa.denseface import FaceAlignment
import torch
import time
import torch
import torchvision


logging.getLogger().setLevel(logging.INFO)

def argparser():
    P = argparse.ArgumentParser(description='Test landmarks')
    P.add_argument('--model-path', type=str, default='snapshot/phase1_wpdc_best.pth.tar', help='Path to model weight')
    P.add_argument('--detector-path', type=str, default='retinaface/scripted_model_cpu_19042021.pt', help='Ratio to expand input')
    P.add_argument('--mode', type=str, default='video', help='testing mode: video, img')
    P.add_argument('--video-path', type=str, default=None, help='path to input video')
    P.add_argument('--img-path', type=str, default=None, help='path to input image')
    P.add_argument('--save-path', type=str, default=None, help='path to save result')
    P.add_argument('--config_path', type=str, help='config path')
    P.add_argument('--bfm-fp', type=str, default='configs/bfm_noneck_v3.pkl')
    
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

    dense_model = FaceAlignment(args.model_path, input_size=128, device='cpu', num_classes=101)
    # pose_model = facelib.models.PoseModel(args.model_path, img_size=size)
    
    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (max(frame_width, frame_height), max(frame_width, frame_height)))
        if not ret:
            break
        
        detected_faces = face_detector.forward(torch.tensor(frame))[0]
        detected_faces = [det for det in detected_faces if det[-1] >= 0.9]
        # frame = cv2.flip(frame, 0)
        key = cv2.waitKey(1) & 0xFF

        try:
            processed_frame = dense_model.draw_landmarks(frame, detected_faces)
        except Exception as e:
            print(e)
            processed_frame = frame
        # processed_frame = dense_model.draw_mesh(frame)
        # angles_dict = dense_model.get_rotate_angles(img, detected_faces)
        # logging.info(f'Landmarks detection took {time.time() - time0}')
     
        if save_video:
            result.write(processed_frame)
        # frame = model.get_head_pose(frame)

        cv2.imshow('', processed_frame)

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
    detected_faces = face_detector.forward(torch.tensor(img))[0]
    detected_faces = [det for det in detected_faces if det[-1] >= 0.9]

    dense_model = FaceAlignment(
        args.model_path, 
        input_size=256, 
        device='cpu', 
        num_classes=101
    )

    # processed_frame = dense_model.draw_landmarks(
    #     args.img_path, 
    #     detected_faces=torch.tensor([[0,0,img.shape[0], img.shape[0]]])
    # )
    processed_frame = dense_model.draw_landmarks(
        args.img_path, 
        detected_faces=detected_faces
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
    else:
        logging.error(f'Invalid mode {args.mode}')
    