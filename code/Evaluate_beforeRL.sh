#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH}
for (( c=1; c<=1; c++ ))
do
    echo "Welcome $c times"
    python src/guesswhat/train/evaluate_qgen_before_reinforce.py\
        -data_dir ../data \
        -exp_dir out/loop \
        -config config/looper/config.json \
        -img_dir ../data/vgg16_pool5_5k/ft_vgg_img \
        -crop_dir ../data/vgg16_pool5_5k/ft_vgg_crop \
        -networks_dir out/ \
        -oracle_identifier "567556d96c4593722286f5ad6d47466c" \
        -qgen_identifier "857098f044abae40f99b0514a6aeeb7a"\
        -guesser_identifier "f3b46e9e4a6b1974925b65528dffbdd7" \
        -looper_identifier "968eedb6956a17f838036d3a75857689"\
        -evaluate_all false \
        -store_games false \
        -no_thread 2
    if [ $? == 0 ]; then
        echo "Successed at $c times"
        break
    fi
done

#5K GuessWhat?! dataset
#bs:8,lr:
#QGenV4.3, 081d264a2e27d95bb1e73cf6314ae2bd
#QGenV3.0 46c46b00fb5b51ee4f592860fc7e222c

#QGen-v5.2,bs:8,lr:4e-5, a791acc5aed9d74f4972a78a9912e925
#QGen-v5.2,bs:8,lr:8e-5, 857098f044abae40f99b0514a6aeeb7a
#QGen-v5.2,bs:8,lr:6e-5, 28e7d24d224f73f94b11a8a5c824b08a

#QGen-v5.2,bs:8,glimse:1,multi-head:2, lr:8e-5, f9bd8336dce94ef3b6c58fe7b85a01e1

#QGen-Looper-v5.2, bs:8,lr:8e-5, 968eedb6956a17f838036d3a75857689

#=========================================
#bs:8,lr:1e-4
#Oracle: 567556d96c4593722286f5ad6d47466c
#Guesser: f3b46e9e4a6b1974925b65528dffbdd7

#bs:64,lr:1e-4
#Oracle: 156cb3d352b97ba12ffd6cf547281ae2
#Guesser: e2c11b1757337d7969dc223c334756a9

