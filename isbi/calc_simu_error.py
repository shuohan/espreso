#!/usr/bin/env python

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
import matplotlib.pyplot as plt
from collections import OrderedDict

from psf_est.utils import calc_fwhm
from lr_simu.simu import ThroughPlaneSimulatorCPU


est_dirname =  '../tests/results_isbi2021_simu_final'
ref_dirname = '/data/phantom/simu'
est_pattern = 'simu_type-%s_fwhm-%s_scale-%s_len-13_smooth-1.0/kernel/avg_epoch-30000.npy'
ref_pattern = 'SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-%s_fwhm-%s_scale-%s_len-13_kernel.npy'
orig_filename = '/data/phantom/data/SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled.nii'
orig_image = nib.load(orig_filename).get_fdata(dtype=np.float32)
data_range = np.max(orig_image) - np.min(orig_image)

types = ['gauss', 'rect']
fwhm = {'gauss': ['2p0', '4p0', '8p0'], 'rect': ['3p0', '5p0', '9p0']}
scale = ['0p5', '0p25', '0p125']

df = list()

for t in types:
    for f in fwhm[t]:
        for s in scale:
            est_filename = Path(est_dirname, est_pattern % (t, f, s))
            ref_filename = Path(ref_dirname, ref_pattern % (t, f, s))
            est = np.load(est_filename).squeeze()
            ref = np.load(ref_filename).squeeze()

            left = (len(est) - len(ref)) // 2
            right = len(est) - len(ref) - left
            ref = np.pad(ref, (left, right))

            est_fwhm = calc_fwhm(est)[0]
            ref_fwhm = calc_fwhm(ref)[0]
            ss = float(s.replace('p', '.'))
            est_simulator = ThroughPlaneSimulatorCPU(est, scale_factor=ss)
            ref_simulator = ThroughPlaneSimulatorCPU(ref, scale_factor=ss)
            est_im = est_simulator.simulate(orig_image)
            ref_im = ref_simulator.simulate(orig_image)
            fwhm_error = np.abs(est_fwhm - ref_fwhm)
            prof_error = np.sum(np.abs(est - ref))

            # data_range = np.max(ref_im) - np.min(ref_im)
            psnr = calc_psnr(ref_im, est_im, data_range=data_range)
            ssim = calc_ssim(ref_im, est_im)

            # plt.subplot(2, 3, 1)
            # plt.imshow(est_im[:, :, est_im.shape[2]//2], cmap='gray')
            # plt.subplot(2, 3, 2)
            # plt.imshow(est_im[:, est_im.shape[1]//2, :], cmap='gray')
            # plt.subplot(2, 3, 3)
            # plt.imshow(est_im[est_im.shape[0]//2, :, :], cmap='gray')
            #           
            # plt.subplot(2, 3, 4)
            # plt.imshow(ref_im[:, :, ref_im.shape[2]//2], cmap='gray')
            # plt.subplot(2, 3, 5)
            # plt.imshow(ref_im[:, ref_im.shape[1]//2, :], cmap='gray')
            # plt.subplot(2, 3, 6)
            # plt.imshow(ref_im[ref_im.shape[0]//2, :, :], cmap='gray')

            # plt.gcf().savefig('%s_%s_%s_error.png' % (t, s, f))

            tab = OrderedDict([('type', t),
                              ('fwhm', f.replace('p', '.')),
                              ('scale', s.replace('p', '.')),
                              ('fwhm error', '%.2f' % fwhm_error),
                              ('profile error', '%.2f' % prof_error),
                              ('psnr', '%.1f' % psnr),
                              ('ssim', '%.2f' % ssim)])
            df.append(tab)

df = pd.DataFrame(df).T
print(df)
