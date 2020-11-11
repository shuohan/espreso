#!/usr/bin/env bash

psf_est_dir=$(realpath $(dirname $0)/..)
sssrlib_dir=~/Code/shuo/deep-networks/sssrlib
proc_dir=~/Code/shuo/utils/image-processing-3d
config_dir=~/Code/shuo/utils/singleton-config
trainer_dir=~/Code/shuo/deep-networks/pytorch-trainer
simu_dir=~/Code/shuo/utils/lr-simu
data_dir=/data

# images=(/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-8p0_scale-0p25_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-4p0_scale-0p25_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-2p0_scale-0p25_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-2p0_scale-0p125_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-4p0_scale-0p125_len-13.nii
#         /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-8p0_scale-0p125_len-13.nii)
images=(/data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-8p0_scale-0p5_len-13.nii
        /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-4p0_scale-0p5_len-13.nii
        /data/phantom/simu/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-2p0_scale-0p5_len-13.nii)

smooth=(0.3)
for image in ${images[@]}; do
    for sm in ${smooth[@]}; do
        fwhm=$(echo $image | sed "s/.*\(fwhm-.*\)_scale.*/\1/")
        scale=$(echo $image | sed "s/.*\(scale-.*\)_len.*/\1/")
        kernel=$(echo $image | sed "s/.*\(type-.*\)_fw.*/\1/")
        len=$(echo $image | sed "s/.*\(len-.*\)\.nii/\1/")
        outdir=../tests/results_isbi2021_l2-smooth_boundary-10_arch_all_test/simu_${kernel}_${fwhm}_${scale}_${len}_smooth-${sm}_boundary-50
        kernel=$(echo $image | sed "s/\.nii/_kernel.npy/")
        echo docker run --gpus device=0 --rm \
            -v $psf_est_dir:$psf_est_dir \
            -v $sssrlib_dir:$sssrlib_dir \
            -v $proc_dir:$proc_dir \
            -v $trainer_dir:$trainer_dir \
            -v $simu_dir:$simu_dir \
            -v $data_dir:$data_dir \
            -v $config_dir:$config_dir \
            --user $(id -u):$(id -g) \
            -e PYTHONPATH=$psf_est_dir:$sssrlib_dir:$proc_dir:$trainer_dir:$config_dir:$simu_dir \
            -w $psf_est_dir/scripts -t \
            psf-est ./train2d.py -i $image -o $outdir -k $kernel -l 19 \
            -sw $sm -isz 1 -bs 16 -e 20000 -w 0 -bw 50
    done
done | rush -j 3 {}