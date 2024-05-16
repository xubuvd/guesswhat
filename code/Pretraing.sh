#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH}
for (( c=1; c<=1; c++ ))
do
    echo "Welcome $c times"
    python src/guesswhat/train/pretrain.py\
        -data_dir /home/Source/dataset/cv/guesswhat/ \
        -exp_dir out/loop \
        -config config/looper/config.json \
        -img_dir /home/Source/dataset/cv/fastercnn_feat_spa/\
        -crop_dir data/vgg16_pool5_5k/ft_vgg_crop \
        -networks_dir out/ \
        -oracle_identifier "156cb3d352b97ba12ffd6cf547281ae2" \
        -qgen_identifier "1a7bf577c08ea15c99b9b83ab01f848e"\
        -guesser_identifier "e2c11b1757337d7969dc223c334756a9" \
        -evaluate_all false \
        -store_games false \
        -no_thread 2
    if [ $? == 0 ]; then
        echo "Successed at $c times"
        break
    fi
done

#=========================================
#Oracle: bs64, 156cb3d352b97ba12ffd6cf547281ae2
#Guesser: bs64, e2c11b1757337d7969dc223c334756a9

#=====q_v5.2_32th_alldata_OsDA_Cancel_UoOR_CMM.log, Has OsDA, Cancel UoOR and CMM model=====
#QGen, 1a7bf577c08ea15c99b9b83ab01f848e 


