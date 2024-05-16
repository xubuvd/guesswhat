#!/bin/bash

wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P data/
#unzip data/img/train2014.zip -d data/img/raw
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P data/
#unzip data/img/val2014.zip -d data/img/raw

wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip -P data/

#-feature_name conv5/conv5_3
#-feature_name fc8

