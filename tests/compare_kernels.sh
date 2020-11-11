#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/..)
data_dir=/data/phantom

name=avg_epoch-10000.npy

# in_kernel=../tests/results_train2d/simu_fwhm-2p0_scale-0p25/kernel/$name
# ref_kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-2p0_scale-0p25_kernel.npy
# output=../tests/results_compare-kernels/simu_fwhm-2p0_scale-0p25.png
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $data_dir:$data_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./compare_kernel.py -i $in_kernel -r $ref_kernel -o $output
# 
# in_kernel=../tests/results_train2d/simu_fwhm-4p0_scale-0p25/kernel/$name
# ref_kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p25_kernel.npy
# output=../tests/results_compare-kernels/simu_fwhm-4p0_scale-0p25.png
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $data_dir:$data_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./compare_kernel.py -i $in_kernel -r $ref_kernel -o $output
# 
# in_kernel=../tests/results_train2d/simu_fwhm-8p0_scale-0p25/kernel/$name
# ref_kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-8p0_scale-0p25_kernel.npy
# output=../tests/results_compare-kernels/simu_fwhm-8p0_scale-0p25.png
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $data_dir:$data_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./compare_kernel.py -i $in_kernel -r $ref_kernel -o $output
# 
# in_kernel=../tests/results_train2d/simu_fwhm-4p0_scale-0p1/kernel/$name
# ref_kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p1_kernel.npy
# output=../tests/results_compare-kernels/simu_fwhm-4p0_scale-0p1.png
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $data_dir:$data_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./compare_kernel.py -i $in_kernel -r $ref_kernel -o $output
# 
# in_kernel=../tests/results_train2d/simu_fwhm-4p0_scale-0p125/kernel/$name
# ref_kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p125_kernel.npy
# output=../tests/results_compare-kernels/simu_fwhm-4p0_scale-0p125.png
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $data_dir:$data_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./compare_kernel.py -i $in_kernel -r $ref_kernel -o $output
# 
# in_kernel=../tests/results_train2d/simu_fwhm-4p0_scale-0p5/kernel/$name
# ref_kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-0p5_kernel.npy
# output=../tests/results_compare-kernels/simu_fwhm-4p0_scale-0p5.png
# 
# docker run --gpus device=1 --rm \
#     -v $psf_est_dir:$psf_est_dir \
#     -v $data_dir:$data_dir \
#     --user $(id -u):$(id -g) \
#     -e PYTHONPATH=$psf_est_dir \
#     -w $psf_est_dir/scripts -t \
#     psf-est ./compare_kernel.py -i $in_kernel -r $ref_kernel -o $output

in_kernel=../tests/results_train2d/simu_fwhm-4p0_scale-1p0/kernel/$name
ref_kernel=/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_fwhm-4p0_scale-1p0_kernel.npy
output=../tests/results_compare-kernels/simu_fwhm-4p0_scale-1p0.png

docker run --gpus device=1 --rm \
    -v $psf_est_dir:$psf_est_dir \
    -v $data_dir:$data_dir \
    --user $(id -u):$(id -g) \
    -e PYTHONPATH=$psf_est_dir \
    -w $psf_est_dir/scripts -t \
    psf-est ./compare_kernel.py -i $in_kernel -r $ref_kernel -o $output

