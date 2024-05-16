#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH}
python src/guesswhat/train/guesser_play.py \
    -data_dir /home/Source/dataset/cv/guesswhat/ \
    -img_dir data/vgg16_fc8/ft_vgg_img \
    -config config/guesser/config.json \
    -exp_dir out/guesser \
    -guesser_identifier "4eebc6911f7b66a828b0eda910979e70" \
    -no_thread 2

#======the guesser game in GuessWhat?!, Human-level=======
#Guesser, pretrained in SL, ES+IS: 4eebc6911f7b66a828b0eda910979e70
#Guesser, trained in RL: 66dd98b8ac125d9148ba9425b52ae5b7

#Guesser, pretrained in SL, only with ES: 5de0ced1d44406589392407c00f42d94

#5Q guesser model trained by self-play: 7d27099382a0f02ab646a7b74d9e0efb


#=====Ablation Study on SymConcat
#Guesser, 3C -> 2C: 5525cf3f0eea03b559c66d47c301708c
#Guesser, 3C order changed: a3399a625189c5520b9d94d769c1c1f7
#Guesser, [a;ab;b]->[ab]: a72ea4b0d745afa3bdab647c2a26cadb

#Guesser, alpha=0.0: 4514604bcd4df8b76533cf219a5a4a6a
#Guesser, alpha=0.1: ec0e05384e57b29e3f09f927410c9008
#Guesser, alpha=0.2: de2d8317b00f0bfc46d607b5b2617657
#Guesser, alpha=0.3: 63a4d2fa4f99244452d298dab674afce
#Guesser, alpha=0.4: a02473bb83578934136e7a8079fab73f
#Guesser, alpha=0.5: 7aa38e1feff12437d44f49ed988ca3b5
#Guesser, alpha=0.6: d0609726c94b6e2ab31c9ec770867c5c
#Guesser, alpha=0.7: 4eebc6911f7b66a828b0eda910979e70 
#Guesser, alpha=0.8: 04e99ba4e0e262e9a4e26c75785767d6
#Guesser, alpha=0.9: 9c5161ec38f0cdc9c82fa5eaebb136b5
#Guesser, alpha=1.0: 5de0ced1d44406589392407c00f42d94 


#ablation study on c
#c = 1.5, 8e36bbba7f9aa1479efd810bd9dedae4
#c = 2.0, 9e4b363c6446ecfd7ef43a05cdd33f42 
#c = 1.1, 4eebc6911f7b66a828b0eda910979e70

