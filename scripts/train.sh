#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

python3 train.py --arch="mobilenet_1" \
    --loss=wpdc \
    --snapshot="snapshot/phase1_wpdc" \
    --opt-style=resample \
    --resample-num=132 \
    --train-batch-size=64 \
    --val-batch-size=128 \
    --base-lr=1e-4 \
    --epochs=9999 \
    --devices-id=0 \
    --workers=4 \
    --log-file="${LOG_FILE}" \
    --train-path 'data/300VW-3D_cropped_closed_eyes_3ddfa' \
                 'data/300VW-3D_cropped_closed_eyes_GAN_3ddfa' \
                 'data/300VW-3D_cropped_opened_eyes_3ddfa' \
                 'data/300WLP_3ddfa' \
    --val-path 'data/AFLW2000_3ddfa' \
    --use-amp \
    --resume 'snapshot/wpdc_last.pth.tar' \
    --beta=0.8