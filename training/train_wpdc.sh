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
    --workers=12 \
    --log-file="${LOG_FILE}" \
    --train-1 'data/300WLP_3ddfa' \
    --train-2 'data/300VW-3D_cropped_closed_eyes_3ddfa' \
    --val-path 'data/AFLW2000_3ddfa' \
    --use-amp \
    --resume 'snapshot/2021-08-16/last.pth.tar'