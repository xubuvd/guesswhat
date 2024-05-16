#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH}
for (( c=1; c<=1; c++ ))
do
    echo "Welcome $c times"
    python src/guesswhat/train/play_game.py\
        -data_dir /home/Source/dataset/cv/guesswhat/ \
        -exp_dir out/loop \
        -config config/looper/config.json \
        -img_dir /home/Source/dataset/cv/fastercnn_feat_spa/\
        -crop_dir data/vgg16_pool5_5k/ft_vgg_crop \
        -networks_dir out/ \
        -oracle_identifier "156cb3d352b97ba12ffd6cf547281ae2" \
        -qgen_identifier "45f252096834939de6be090bc1d38e0d"\
        -guesser_identifier "66dd98b8ac125d9148ba9425b52ae5b7" \
        -looper_identifier "3de23c468246bee56ce1c0bbf5a4a6ed"\
        -evaluate_all false \
        -store_games false \
        -no_thread 2
    if [ $? == 0 ]; then
        echo "Successed at $c times"
        break
    fi
done

#==============reached human-level performance, 8Q==============
#Loop: 3de23c468246bee56ce1c0bbf5a4a6ed
#QGen: 45f252096834939de6be090bc1d38e0d
#Guesser: 66dd98b8ac125d9148ba9425b52ae5b7
#Oracle: 156cb3d352b97ba12ffd6cf547281ae2
#=============reached human-level performance, 8Q==============

#============reached human-level performance, 5Q=================
#GuesserRL: 7d27099382a0f02ab646a7b74d9e0efb
#Looper: 0ab463d19fd89dc1773e339783fb0e7c
#QGen: 45f252096834939de6be090bc1d38e0d
#Oracle: 156cb3d352b97ba12ffd6cf547281ae2
#Guesser: 4eebc6911f7b66a828b0eda910979e70
#============reached human-level performance, 5Q=================

#d8d378d0adf7215a43a11c5c42c77134
#=========================================
#Oracle: bs64, 156cb3d352b97ba12ffd6cf547281ae2
#Guesser: bs64, e2c11b1757337d7969dc223c334756a9
#QGen-v5.2: aeac9b145056eba879bf3baac3a64319

#QGen-v5.2, alldata,bs64,dual-semantic_loss canceled, c166921cfad075d7e7828591af04961f
#Loop, 8Q-500epoch, b46da7a720c5c14530d3883fd42bf9d1
#Loop.5Q-800epoch, da24d0bb115e94a20b7eb6ebf90d1a69 

#Loop.5Q-500epoch, no ShortConnections: 26b89bac7d442d72725085d48cdffefa
#Loop,5Q-350epoch, no BatchNorm: 9151ffb715d1093c563a95e487469588
#Loop.5Q-350epoch, OsDA in place of avg object representation, dd360ae2813aae38730e98d3607714d7

#==============No BN,Short and Vh=====================
#QGen, 50/50 great idea: f4f1de4167187834ed6230c773ef0fac
#Loop.5Q-350epoch, No BN,Short and Vh: 47895a136c4d199bdb42c529ebd01259

#=============Has OsDA, Cancel UoOR and CMM model===============
#Loop.5Q-350epoch: de84777c1d052f13449480c8ad33675b
#QGen: 1a7bf577c08ea15c99b9b83ab01f848e

#=============No OsDA, Has UoOR & CMM=============
#dd360ae2813aae38730e98d3607714d7

#==============New VSDT model====================
#No BN Vh and Short, AAAI-20 paper
#Loop. 5Q-500epoch, d8d378d0adf7215a43a11c5c42c77134
#-oracle_identifier "156cb3d352b97ba12ffd6cf547281ae2"
#-qgen_identifier "f4f1de4167187834ed6230c773ef0fac"
#-guesser_identifier "e2c11b1757337d7969dc223c334756a9"

#Loop. 8Q-500epoch, d03e13f598160a83bee5321a15b99f54
#-oracle_identifier "156cb3d352b97ba12ffd6cf547281ae2"
#-qgen_identifier "f4f1de4167187834ed6230c773ef0fac"
#-guesser_identifier "e2c11b1757337d7969dc223c334756a9"
