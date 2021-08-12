#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

python3 train.py --arch="mobilenet_1" \
    --loss=wpdc \
    --snapshot="snapshot/phase1_wpdc" \
    --warmup=5 \
    --opt-style=resample \
    --resample-num=132 \
    --train-batch-size=64 \
    --val-batch-size=128 \
    --base-lr=1e-4 \
    --epochs=9999 \
    --milestones=30,40 \
    --print-freq=50 \
    --devices-id=0,1 \
    --workers=12 \
    --log-file="${LOG_FILE}" \
    --resume 'snapshot/2021-08-12/best.pth.tar' \
    --train-one 'data/300VW-3D_cropped_3ddfa' \
    --train-two 'data/300WLP_3ddfa' \
    --val-path 'data/AFLW2000_3ddfa' \
    --use-amp