#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH}
python src/guesswhat/train/train_guesser.py \
    -data_dir /home/Source/dataset/cv/guesswhat/ \
    -img_dir data/vgg16_fc8/ft_vgg_img \
    -config config/guesser/config.json \
    -exp_dir out/guesser \
    -no_thread 2

