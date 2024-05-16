#!/bin/bash

filename=$1
scp -r pw@10.108.211.36:/home/pw/word2vec/src/guesswhat/data/img/raw/$filename.jpg ./images/

