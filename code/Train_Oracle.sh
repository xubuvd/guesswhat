#!/bin/bash
#Train Oracle

export PYTHONPATH=src:${PYTHONPATH}
python src/guesswhat/train/train_oracle.py \
    -data_dir /home/Source/dataset/cv/guesswhat/ \
    -img_dir data/vgg16_fc8/ft_vgg_img/ \
    -crop_dir data/vgg16_fc8/ft_vgg_crop \
    -config config/oracle/config.json \
    -exp_dir out/oracle \
    -no_thread 2

