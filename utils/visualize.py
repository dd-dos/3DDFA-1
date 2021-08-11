import cv2
import torch
import torchvision
import numpy as np
import torch
import torchvision.transforms as T
from .face3d import face3d
fm = face3d.face_model.FaceModel()

@torch.no_grad()
def log_training_samples(imgs, preds, gts, writer, id, job):
    for idx in range(imgs.shape[0]):
        arr_img = de_normalize(imgs[idx], [127.5,127.5,127.5],[128,128,128]).cpu().numpy()
        # arr_img = arr_img.numpy()*255
        arr_img = arr_img.transpose(1,2,0).astype(np.uint8)
        arr_img = np.ascontiguousarray(arr_img, dtype=np.uint8)

        gt_img = arr_img.copy()

        pred = preds[idx].cpu().numpy()
        gt = gts[idx].cpu().numpy()

        pred_vertex = fm.reconstruct_vertex(pred)[fm.bfm.kpt_ind]
        gt_vertex = fm.reconstruct_vertex(gt)[fm.bfm.kpt_ind]

        pred_pts = pred_vertex[:2].T
        gt_pts = gt_vertex[:2].T
        
        for _pts in pred_pts:
            _pts = tuple(_pts.astype(np.uint8))
            cv2.circle(arr_img, _pts, 3, (0,255,0), -1, 10)
        
        for _pts in gt_pts:
            _pts = tuple(_pts.astype(np.uint8))
            cv2.circle(gt_img, _pts, 3, (0,255,0), -1, 10)

        comparision = np.concatenate((gt_img, arr_img), axis=1)
        comparision = cv2.cvtColor(comparision, cv2.COLOR_BGR2RGB)

        tensor_img = torch.tensor(comparision.transpose(2,0,1)/255.,
                                  dtype=torch.float32).unsqueeze(0)
        grid = torchvision.utils.make_grid(tensor_img)
        writer.add_image(f"{job}/sample-{idx}", grid, id)     


def de_normalize(tensor_img: torch.Tensor, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]):
    de_std = 1/np.array(std)
    de_mean = -np.array(mean)*de_std
    de_normalizer = T.Normalize(de_mean, de_std)
    _img = de_normalizer(tensor_img)

    return _img