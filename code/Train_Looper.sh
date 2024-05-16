#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH}
for (( c=1; c<=1; c++ ))
do
    echo "Welcome $c times"
    python src/guesswhat/train/train_qgen_reinforce.py\
        -data_dir /home/Source/dataset/cv/guesswhat/ \
        -exp_dir out/loop \
        -exp_guesser_RL_dir out/guesser_RL \
        -config config/looper/config.json \
        -guesser_RL_config config/guesser_RL/config.json \
        -img_dir /home/Source/dataset/cv/fastercnn_feat_spa/\
        -crop_dir data/vgg16_pool5_5k/ft_vgg_crop \
        -networks_dir out/ \
        -oracle_identifier "156cb3d352b97ba12ffd6cf547281ae2" \
        -qgen_identifier "45f252096834939de6be090bc1d38e0d"\
        -guesser_identifier "4eebc6911f7b66a828b0eda910979e70" \
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
#Guesser Memory Attention : 7a2ff93b52d1265360d0330be91f1472
#Guesser, Original Version with policy gradient alogirthm without baseline function: ebce3900705440f4664ca1ac7eb2c681
#Guesser, VDST Updated of Object Representation: 4f91197e67ab65ccc36c48ae7af4a514
#Guesser, VDST, UoOR, Data Augmentation, 4D3C3L:  4e4cf726c88cbbfc8fdc1b0793d00e3b 
#Gueseer, VDST, UoOR, Data Augmentation, 4D3C3L,Simplied: 969f0071c946cfb2f4987a5087dfc63a
#Guesser, VDST, similar to QGen, 4D3C3L: a7223b93086dce1325b545a66fcdcd6a
#Guesser, VDST, 0D3C2L, 19/20 great work: e49c45a3c7980ea69d5ec4a2a4bc2e61
#Guesser, VDST, 0D3C2L, 11/20 great work,progressively: 4eebc6911f7b66a828b0eda910979e70



#QGen,v5.2,32th,4glimse-1head, ed6569a1c0a5ed836cb331f5a27a81b8
#QGen,v5.2,32th,2glimse-1head, aeac9b145056eba879bf3baac3a64319
#QGen,v5.2,32th,2glimpse-1head, 512 instead 300, 40cf658355e4517f84c9d34ffa510a11
#QGen,v5.2,33th,dualSemanticInverse, 4f8bb4d0695cd13872f27c83dfc596e4
#QGen,v5.2,32th,2glimse-2head, 5c6e559b1d7b627c8bacf7cc784d0fd9
#QGen,v5.3,32th,NoBatchNorm in VSDT: f4f1de4167187834ed6230c773ef0fac

#QGen,v5.2,32th, AblationStudy, cancel dot product of Vh in CMM. bd9810fbd31889d6872a1e787f6d6de6
#QGen,v5.2,32th, AblationStudy, cancel BN in MOR, 3c91f7f598ff2d5d711855aca5ce2675
#QGen,v5.2,32th, AblationStudy, cancel ShortConnection in MOR, 3825f2e6b64b62047680201b3c033e9f
#QGen,v6,VDST Simplied Version: cdbdf9843568e07c16e218bdfdb436c5
#QGen,v6,VDST model, adding Stop, 25f7cfb69e48b79713116c8291e83a0c
#QGen,v6,StopAsking, MLp([prev_answer;MLP(Distributions ob Objects)]) in place of MLP(Distributions on Objects): e1407989cb0d7e69d4b929fff4ba67c0
#QGen,v6,VDST Simplied Version, reproducing the results in AAAI-20 paper:66449ba5644bc894a677eff7aaaea0b2
#QGen,v7,VDST Simplied Version, Segmentaton UoOR: 48c758dcc07ad60630f92058b486d1ad
#QGen, VDST Simplied Version, prepared for Guesser with Memory Attention: 83d0a30c95c06d83a4393c304edc8497
#QGen, VDST Simplied Version, CMM v2: d8a506b3010915dcdaf83c65052f4ca8
#QGen, VDST Simplied Version, CMM v2, consistent to Guesser: 45f252096834939de6be090bc1d38e0d

