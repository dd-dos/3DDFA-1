from utils.ddfa_v2 import DDFAv2_Dataset
from utils.ddfa import ToTensorGjz, NormalizeGjz
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm
import os
import shutil
import cv2
import numpy as np
from utils.augment import n_rotate_vertex

if __name__=='__main__':
    shutil.rmtree('input_samples', ignore_errors=True)
    os.makedirs('input_samples', exist_ok=True)
    # dataset = DDFAv2_Dataset(
    #     'data/300VW-3D_cropped_3ddfa',
    #     transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
    # )

    # train_loader = DataLoader(dataset, batch_size=32, num_workers=0,
    #                         shuffle=True, pin_memory=True, drop_last=True)

    # for item in tqdm.tqdm(train_loader, total=len(train_loader)):
    #     pass

    img = cv2.imread('data/300VW-3D_cropped_3ddfa/519/0002.jpg')
    param = np.load('data/300VW-3D_cropped_3ddfa/519/0002.npy')

    from utils.face3d.face3d.face_model import FaceModel
    fm = FaceModel()
    # fm.reconstruct_vertex(img, param)
    n_rotate_vertex(img, param, 45)

