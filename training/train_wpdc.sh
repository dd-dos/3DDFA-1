#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

python3 train.py --arch="mobilenet_1" \
    --start-epoch=1 \
    --loss=wpdc \
    --snapshot="snapshot/phase1_wpdc" \
    --param-fp-train='train.configs/param_all_norm.pkl' \
    --param-fp-val='train.configs/param_all_norm_val.pkl' \
    --warmup=5 \
    --opt-style=resample \
    --resample-num=132 \
    --train-batch-size=64 \
    --base-lr=1e-4 \
    --epochs=9999 \
    --milestones=30,40 \
    --print-freq=50 \
    --devices-id=0 \
    --workers=16 \
    --filelists-train="train.configs/train_aug_120x120.list.train" \
    --filelists-val="train.configs/train_aug_120x120.list.val" \
    --root="train.configs/train_aug_120x120" \
    --log-file="${LOG_FILE}" \
    --resume "snapshot/2021-07-22/best.pth.tar"
