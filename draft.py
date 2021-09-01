import cv2
import numpy as np
import scipy.io as sio
from pathlib import Path
import os
import tqdm
import shutil

if __name__=='__main__':
    # shutil.rmtree('input_samples', ignore_errors=True)
    # os.makedirs('input_samples', exist_ok=True)
    # dataset = DDFAv2_Dataset(
    #     'data/300WLP_3ddfa',
    #     transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
    # )

    # train_loader = DataLoader(dataset, batch_size=32, num_workers=0,
    #                         shuffle=True, pin_memory=True, drop_last=True)

    # for item in tqdm.tqdm(train_loader, total=len(train_loader)):
    #     pass
    from utils.face3d.face3d.face_model import FaceModel
    fm = FaceModel()

    file_list = list(Path('data/300VW-3D_cropped_closed_eyes_3ddfa').glob('**/*_0*.jpg'))
    
    file_list = ['data/300VW-3D_cropped_closed_eyes_3ddfa/001/0560_0_fliplr.jpg', 'data/300VW-3D_cropped_closed_eyes_3ddfa/001/0560_0.jpg']
    shutil.rmtree('input_samples', ignore_errors=True)
    os.makedirs('input_samples', exist_ok=True)

    for idx in tqdm.tqdm(range(len(file_list))):
        file_path = str(file_list[idx])
        file_id = file_path.split('/')[-1].split('.')[0]
        folder_id = file_path.split('/')[-2]

        img = cv2.imread(file_path)
        params = sio.loadmat(file_path.replace('jpg', 'mat'))['params'].reshape(-1,)
        pts = fm.reconstruct_vertex(img, params, de_normalize=False)[:,:2][fm.bfm.kpt_ind]

        os.makedirs(f'input_samples/{folder_id}', exist_ok=True)
        for idx in range(len(pts)):
            pt = tuple(pts[idx].astype(np.uint8))
            cv2.circle(img, pt, 2, (0,255,0), -1, 5)
        
            cv2.imwrite(f'input_samples/{folder_id}/{file_id}_{idx}.jpg', img)