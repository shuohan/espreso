#!/usr/bin/env bash

ssp_dir=$(realpath $(dirname $0)/..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/improc3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
data_dir=/data

images=(/data/phantom/data/SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-4mm-gapn2mm_resampled.nii
        /data/phantom/data/SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-4mm-gapn1mm_resampled.nii
        /data/phantom/data/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-04mm_resampled.nii)

sm=1.0
for image in ${images[@]}; do
    fwhm=$(echo $image | sed "s/.*ACQ1-[0]*\([^-]*mm\).*/\1/")
    gap=$(echo $image | sed "s/.*-0*${fwhm}-*\(.*\)_resampled.*/\1/")
    if [ -z $gap ]; then
        gap=gap0mm
    fi
    outdir=../tests/results_isbi2021_phantom_final/phantom_${fwhm}_${gap}_smooth-${sm}
    echo docker run --gpus device=0 --rm \
        -v $ssp_dir:$ssp_dir \
        -v $sssrlib_dir:$sssrlib_dir \
        -v $proc_dir:$proc_dir \
        -v $trainer_dir:$trainer_dir \
        -v $data_dir:$data_dir \
        -v $config_dir:$config_dir \
        --user $(id -u):$(id -g) \
        -e PYTHONPATH=$ssp_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir \
        -w $ssp_dir/scripts -t \
        ssp ./train2d.py -i $image -o $outdir -l 19 \
        -sw $sm -isz 1 -bs 16 -e 30000 -w 0
done | rush -j 3 {}
