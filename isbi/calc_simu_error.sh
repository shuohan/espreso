#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/..)
simu_dir=~/Code/shuo/utils/lr-simu
proc_dir=~/Code/shuo/utils/image-processing-3d
data_dir=/data

docker run --gpus device=0 --rm \
    -v $psf_est_dir:$psf_est_dir \
    -v $simu_dir:$simu_dir \
    -v $proc_dir:$proc_dir \
    -v $data_dir:$data_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$psf_est_dir:$proc_dir:$simu_dir \
    -w $PWD -t \
    pytorch-shan ./calc_simu_error.py
