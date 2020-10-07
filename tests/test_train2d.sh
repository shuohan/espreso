#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
simu_dir=~/Code/shuo/utils/lr-simu
data_dir=/data/phantom

# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-04mm_resampled.nii
image=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-8p0_scale-0p25.nii
outdir=../tests/results_train2d/smoothness_8mm_0p25_w1

docker run --gpus device=1 --rm \
    -v $psf_est_dir:$psf_est_dir \
    -v $sssrlib_dir:$sssrlib_dir \
    -v $proc_dir:$proc_dir \
    -v $trainer_dir:$trainer_dir \
    -v $simu_dir:$simu_dir \
    -v $data_dir:$data_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
    -w $psf_est_dir/scripts -t \
    psf-est ./train2d.py -i $image -o $outdir
