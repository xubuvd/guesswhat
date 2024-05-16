#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH} 
export CPU_NUM=2
python src/guesswhat/train/train_qgen_supervised.py \
    -data_dir /home/Source/dataset/cv/guesswhat/ \
    -img_dir /home/Source/dataset/cv/fastercnn_feat_spa/\
    -config config/qgen/config.json \
    -exp_dir out/qgen/ \
    -no_thread 2

