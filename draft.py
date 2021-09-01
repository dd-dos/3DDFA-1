import cv2
import numpy as np
from utils.augment import n_rotate_vertex
from vdc_loss import VDCLoss

if __name__=='__main__':
    shutil.rmtree('input_samples', ignore_errors=True)
    os.makedirs('input_samples', exist_ok=True)
    dataset = DDFAv2_Dataset(
        'data',
        transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
    )

    train_loader = DataLoader(dataset, batch_size=1, num_workers=0,
                            shuffle=True, pin_memory=True, drop_last=False)

    vdc_loss = VDCLoss(opt_style='resample')

    for input, target in tqdm.tqdm(train_loader, total=len(train_loader)):
        loss = vdc_loss(target, target)
