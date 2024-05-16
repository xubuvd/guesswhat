#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH}
#array=( img crop )
array=( crop )
for mode in "${array[@]}"; do
    python src/guesswhat/preprocess_data/extract_img_features.py \
        -img_dir ../data/raw \
        -data_dir ../data \
        -out_dir data/vgg16_pool5/ft_crop_$mode \
        -network vgg \
        -ckpt ../data/vgg_16.ckpt \
        -feature_name pool5 \
        -mode $mode
done

#-feature_name conv5/conv5_3
#-feature_name fc8

