#!/usr/bin/env bash

dir=$(realpath $(dirname $0)/..)
lr_simu_dir=~/Code/shuo/utils/lr-simu

docker run --gpus device=1 --rm -v $dir:$dir \
    -v $lr_simu_dir:$lr_simu_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$dir:$lr_simu_dir -w $dir/tests -t \
    ssp ./test_loss.py
