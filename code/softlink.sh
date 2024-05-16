#!/bin/bash

export PYTHONPATH=src:${PYTHONPATH}
#array=( img crop )

python src/guesswhat/preprocess_data/rewire_coco_image_id.py \
    -image_dir /home/pw/VQA/download/ \
    -data_out /home/pw/word2vec/src/guesswhat/data/raw/


