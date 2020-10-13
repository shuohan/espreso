#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
simu_dir=~/Code/shuo/utils/lr-simu
data_dir=/data/phantom

# image=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-2p0_scale-0p25.nii
# outdir=../tests/results_train2d/simu_fwhm-2p0_scale-0p25
# kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-2p0_scale-0p25_kernel.npy
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -k $kernel -l 21
# 
# image=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p25.nii
# outdir=../tests/results_train2d/simu_fwhm-4p0_scale-0p25
# kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p25_kernel.npy
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -k $kernel -l 21
# 
# 
# image=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-8p0_scale-0p25.nii
# outdir=../tests/results_train2d/simu_fwhm-8p0_scale-0p25
# kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-8p0_scale-0p25_kernel.npy
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -k $kernel -l 21
# 
# 
# image=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p1.nii
# outdir=../tests/results_train2d/simu_fwhm-4p0_scale-0p1
# kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p1_kernel.npy
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -k $kernel -l 21
# 
# 
# image=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p125.nii
# outdir=../tests/results_train2d/simu_fwhm-4p0_scale-0p125
# kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p125_kernel.npy
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $config_dir:$config_dir \
#     -v $data_dir:$data_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -k $kernel -l 21
# 
# 
# image=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p5.nii
# outdir=../tests/results_train2d/simu_fwhm-4p0_scale-0p5
# kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p5_kernel.npy
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -k $kernel -l 21
# 
# 
# image=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-1p0.nii
# outdir=../tests/results_train2d/simu_fwhm-4p0_scale-1p0
# kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-1p0_kernel.npy
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -k $kernel -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-02mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-02_gap-0
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-04mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-04_gap-0
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-08mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-08_gap-0
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $config_dir:$config_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-10mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-10_gap-0
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-01mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-01_gap-0
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-2mm-gap1mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-02_gap-01
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-4mm-gap1mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-04_gap-01
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-2mm-gapn0p5mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-02_gap-n0p5
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-4mm-gap2mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-04_gap-02
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
# 
# 
# image=/data/phantom/data/SUPERRES-ADNIPHANTOM_20200830_PHANTOM-T2-TSE-2D-CORONAL-PRE-ACQ1-4mm-gapn2mm_resampled.nii
# outdir=../tests/results_train2d/real_thick-04_gap-n02
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $sssrlib_dir:$sssrlib_dir \
#     -v $proc_dir:$proc_dir \
#     -v $trainer_dir:$trainer_dir \
#     -v $simu_dir:$simu_dir \
#     -v $data_dir:$data_dir \
#     -v $config_dir:$config_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./train2d.py -i $image -o $outdir -l 21
