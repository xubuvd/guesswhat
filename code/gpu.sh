#!/bin/bash

gpustat
nvidia-smi

#watch --color -n10 gpustat -cpu 
#nvidia-smi -l 2

#执行fuser -v /dev/nvidia* 发现僵尸进程（连号的）
fuser -v /dev/nvidia*


