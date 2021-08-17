from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

import os.path as osp
from pathlib import Path
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mobilenet_v1
import torch.backends.cudnn as cudnn

from utils.ddfa import ToTensorGjz, NormalizeGjz
from utils.ddfa import str2bool, AverageMeter
from utils.ddfa_v2 import DDFAv2_Dataset
from utils.io import mkdir
from utils.scheduler import CyclicCosineDecayLR
from vdc_loss import VDCLoss
from wpdc_loss import WPDCLoss
from utils.visualize import log_training_samples, RandomBullShit
import tqdm
from datetime import datetime
import os

lr = None
LOSS = 0.
arch_choices = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']

# from clearml import Task
# task = Task.init(project_name="Facial-landmark", task_name="3DDFA-Close-eyes-Adam-CyclicCosineDecayLR")
TODAY = datetime.today().strftime('%Y-%m-%d')
os.makedirs(f'snapshot/{TODAY}', exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='3DMM Fitting')
    parser.add_argument('--workers', default=6, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--train-batch-size', default=128, type=int)
    parser.add_argument('--val-batch-size', default=32, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float)
    parser.add_argument('--print-freq', '-p', default=20, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--devices-id', default='0,1', type=str)
    parser.add_argument('--train-1', default='', type=str)
    parser.add_argument('--train-2', default='', type=str)
    parser.add_argument('--val-path', default='', type=str)

    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    parser.add_argument('--size-average', default='true', type=str2bool)
    parser.add_argument('--num-classes', default=101, type=int, help = '12 pose + 60 shape + 29 expression')
    parser.add_argument('--arch', default='mobilenet_1', type=str,
                        choices=arch_choices)
    parser.add_argument('--frozen', default='false', type=str2bool)
    parser.add_argument('--opt-style', default='resample', type=str)  # resample
    parser.add_argument('--resample-num', default=132, type=int)
    parser.add_argument('--loss', default='vdc', type=str)
    parser.add_argument('--beta', default=0.7, type=float, help='vanilla joint control parameter')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-scheduler', action='store_true')
    parser.add_argument('--scheduler-init-decay-epoch', default=10, type=int)
    parser.add_argument('--scheduler-min-decay-lr', default=1e-6, type=float)
    parser.add_argument('--scheduler-restart-interval', default=3, type=float)
    parser.add_argument('--scheduler-restart-lr', default=1e-4, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)

    global args
    args = parser.parse_args()
    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')


ITER = 0
def train(train_loader, model, wpdc_loss, vdc_loss, optimizer, epoch, scaler):
    global ITER
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    random_bullshit = RandomBullShit()
    use_amp = args.use_amp
    num_samples = train_loader.dataset.__len__()

    '''
    Logging after every $(logging_step) iterations
    '''
    logging_step = np.ceil(num_samples/(10*args.train_batch_size))
    top_loss = 0
    end = time.time()

    model.train()
    for i, (input, target) in enumerate(tqdm.tqdm(train_loader, total=len(train_loader))):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            
            output = model(input)
            
            data_time.update(time.time() - end)
            wpdc_loss_value = wpdc_loss(output, target)
            # vdc_loss_value = vdc_loss(output, target)
            # total_loss = args.beta*wpdc_loss_value + (1-args.beta)*vdc_loss_value*(2e-2)
            total_loss = wpdc_loss_value
            losses.update(total_loss)

            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        random_bullshit.update(output.cuda(), target)

        '''
        Measure elapsed time
        '''
        batch_time.update(time.time() - end)
        end = time.time()

        if total_loss.item() >= top_loss:
            top_loss = total_loss.item()
            top_loss_samples = {
                'input': input,
                'target': target,
                'output': output
            }

        if i%logging_step==logging_step-1:
            logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t' 
                         f'LR: {lr:8f}\t' 
                         f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' 
                         f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' 
                         f'Loss {losses.val:.4f} ({losses.avg:.4f})')
            log_training_samples(input[:32], output[:32], target[:32], writer, ITER, 'Train/End-logging-step')
            writer.add_scalar('Loss/Train', losses.avg, ITER)
            random_bullshit.go(writer, ITER, 'Train')

            '''
            Log top-loss samples.
            '''
            input = top_loss_samples['input']
            target = top_loss_samples['target']
            output = top_loss_samples['output'] 

            log_training_samples(input[:32], output[:32], target[:32], writer, ITER, 'Train/Top-loss')

        ITER += 1


def validate(val_loader, model, epoch):
    global LOSS
    model.eval()
    use_amp = args.use_amp
    end = time.time()
    wpdc_criterion = WPDCLoss(opt_style=args.opt_style).cuda()
    vdc_criterion = VDCLoss(opt_style=args.opt_style).cuda()

    with torch.no_grad():
        vdc_losses = AverageMeter()
        wpdc_losses = AverageMeter()
        losses = AverageMeter()
        random_bullshit = RandomBullShit()
        top_loss = 0
        with torch.cuda.amp.autocast(enabled=use_amp):
            for i, (input, target) in enumerate(val_loader):
                target.requires_grad = False
                target = target.cuda(non_blocking=True)
                output = model(input)

                vdc_loss = vdc_criterion(output, target)
                vdc_losses.update(vdc_loss.item(), input.size(0))

                wpdc_loss = wpdc_criterion(output, target)
                wpdc_losses.update(wpdc_loss.item(), input.size(0))

                random_bullshit.update(output.cuda(), target)

                if args.loss.lower() == 'wpdc':
                    if wpdc_loss.item() >= top_loss:
                        top_loss = wpdc_loss.item()
                        top_loss_samples = {
                            'input': input,
                            'target': target,
                            'output': output
                        }
                else:
                    if vdc_loss.item() >= top_loss:
                        top_loss = vdc_loss.item()
                        top_loss_samples = {
                            'input': input,
                            'target': target,
                            'output': output
                        }

        elapse = time.time() - end
        logging.info(f'Val: [{epoch}][{len(val_loader)}]\t'
                     f'VDCLoss {vdc_losses.avg:.4f}\t'
                     f'WPDCLoss {wpdc_losses.avg:.4f}\t'
                     f'Time {elapse:.3f}')
        writer.add_scalar('VDC_Loss/Val', vdc_losses.avg, ITER)
        writer.add_scalar('WPDC_Loss/Val', wpdc_losses.avg, ITER)

        random_bullshit.go(writer, ITER, 'Val')

        '''
        Log top-loss samples.
        '''
        input = top_loss_samples['input']
        target = top_loss_samples['target']
        output = top_loss_samples['output']

        log_training_samples(input, output, target, writer, ITER, 'Val/Top-loss')
        writer.add_scalar('Top-loss/Val', top_loss, ITER)

        losses = vdc_losses
        if args.loss.lower() == 'wpdc':
            losses = wpdc_losses

        if epoch==0:
            LOSS=losses.avg
        else:
            if losses.avg <= LOSS:
                LOSS = losses.avg
                filename = f'snapshot/{TODAY}/best.pth.tar'
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        # 'optimizer': optimizer.state_dict()
                    },
                    filename
                )
            else:
                filename = f'snapshot/{TODAY}/last.pth.tar'
                save_checkpoint(
                    {
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        # 'optimizer': optimizer.state_dict()
                    },
                    filename
                )


def main():
    parse_args()  

    logging.basicConfig(
        format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )

    if args.resume:
        model = getattr(mobilenet_v1, args.arch)(num_classes=args.num_classes)
        if Path(args.resume).is_file():
            logging.info(f'Loading checkpoint {args.resume}')
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict']

            model_dict = model.state_dict()
            '''
            Because the model is trained by multiple gpus, prefix module should be removed
            '''
            for k in checkpoint.keys():
                model_dict[k.replace('module.', '')] = checkpoint[k]
            model.load_state_dict(model_dict)
        else:
            logging.info(f'=> No checkpoint found at {args.resume}')
    else:
        model = getattr(mobilenet_v1, args.arch)(num_classes=args.num_classes)

    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`
    model = nn.DataParallel(model, device_ids=args.devices_id).cuda()  # -> GPU

    train_dataset_1 = DDFAv2_Dataset(
        root=args.train_1,
        transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
        aug=True
    )

    train_dataset_2 = DDFAv2_Dataset(
        root=args.train_2,
        transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
        aug=True
    )
    
    val_dataset = DDFAv2_Dataset(
        root=args.val_path,
        transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]),
        aug=False
    )

    logging.info(f'Number of samples: ')
    logging.info(f'=> {args.train_1}: {len(train_dataset_1)}')
    logging.info(f'=> {args.train_2}: {len(train_dataset_2)}')
    logging.info(f'=> {args.val_path}: {len(val_dataset)}')

    concat_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])
    train_loader = DataLoader(concat_dataset, batch_size=args.train_batch_size, num_workers=args.workers,
                              shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers,
                            shuffle=False, pin_memory=True)

    cudnn.benchmark = True

    if args.optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    if args.use_scheduler:
        scheduler = CyclicCosineDecayLR(optimizer, 
                                    init_decay_epochs=args.init_decay_epochs,
                                    min_decay_lr=args.min_decay_lr,
                                    restart_interval =args.restart_interval,
                                    restart_lr=args.restart_lr)

    global lr
    lr = args.base_lr
    wpdc_loss = WPDCLoss(opt_style=args.opt_style).cuda() 
    vdc_loss = VDCLoss(opt_style=args.opt_style).cuda()

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for epoch in range(args.epochs):
        train(train_loader, model, wpdc_loss, vdc_loss, optimizer, epoch, scaler)
        validate(val_loader, model, epoch)

        if args.use_scheduler:
            scheduler.step()
            lr = scheduler.get_lr()[0]


if __name__ == '__main__':
    main()
