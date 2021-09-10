from pathlib import Path
import scipy.io as sio
from utils.ddfa_v2 import DDFAv2_Dataset
from utils.imutils import ToTensorGjz, NormalizeGjz
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import os
from utils.compute import RunningStats
import tqdm

train_list = ['data/300VW-3D_closed_eyes_3ddfa', 'data/300WLP_3ddfa']
train_dataset = []
mean_std_path = 'utils/face3d/face3d/morphable_model/BFM/params_mean_std.mat'
if os.path.isfile(mean_std_path):
    os.remove(mean_std_path)

for idx in range(len(train_list)):
    train_path = train_list[idx]

    if '300VW' in train_path:
        if 'opened_eyes' not in train_path:
            '''
            file_list_0 is original face.
            file_list_1 is generated face based on original face.
            '''
            file_list_0 = list(Path(train_path).glob('**/*_0.jpg'))
            file_list_1 = list(Path(train_path).glob('**/*_1.jpg'))

            train_dataset.append(
                DDFAv2_Dataset(
                    root=file_list_0,
                    transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
                    aug=True,
                    hide_face_rate=0,
                    vanilla_aug_rate=0,
                    rotate_rate=1
                )
            )

            # train_dataset.append(
            #     DDFAv2_Dataset(
            #         root=file_list_1,
            #         transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
            #         aug=True,
            #         hide_face_rate=0,
            #         vanilla_aug_rate=0,
            #         rotate_rate=0
            #     )
            # )
        else:
            file_list_1 = list(Path(train_path).glob('**/*_1.jpg'))
            train_dataset.append(
                DDFAv2_Dataset(
                    root=file_list_1,
                    transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
                    aug=True,
                    hide_face_rate=0,
                    vanilla_aug_rate=0,
                    rotate_rate=0
                )
            )
    else:
        train_dataset.append(
            DDFAv2_Dataset(
                root=train_path,
                transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
                aug=True,
                hide_face_rate=0,
                vanilla_aug_rate=0,
                rotate_rate=0
            )
        )

concat_dataset = torch.utils.data.ConcatDataset(train_dataset)

train_loader = DataLoader(concat_dataset, batch_size=1, num_workers=16,
                            shuffle=True, pin_memory=True, drop_last=False)

rs = RunningStats()
for _ in range(3):
    for i, (input, target) in enumerate(tqdm.tqdm(train_loader, total=len(train_loader))):
        rs.push(target[0].numpy())

sio.savemat('utils/face3d/face3d/morphable_model/BFM/params_mean_std.mat', 
            {'mean': rs.mean(), 'std': rs.standard_deviation()})