#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

python3 train.py --backbone="mobilenet_v2" \
                --arch="mobilenet_2" \
                --sub-name="finetune" \
                --snapshot="snapshot/phase1_wpdc" \
                --opt-style='resample' \
                --resample-num=0 \
                --train-batch-size=128 \
                --val-batch-size=128 \
                --base-lr=1e-4 \
                --num-classes=101 \
                --epochs=9999 \
                --devices-id=0 \
                --workers=16 \
                --log-file="${LOG_FILE}" \
                --train-path 'data/300VW-3D_closed_eyes_3ddfa' \
                            'data/300WLP_3ddfa' \
                --val-path 'data/AFLW2000_3ddfa' \
                --use-amp \
                --num-log-samples=32 \
                --loss 'vdc' \
                --optimizer 'sgd' \
                --use-scheduler \
                --clearml \
                --task-name '3DDFA-300WLP-300VW-closed-eyes-finetune' \
                --resume 'snapshot/2021-09-09/mobilenet_v2_mobilenet_2_finetune_wpdc_best.pth.tar'