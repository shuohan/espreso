#!/usr/bin/env python 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib 

from psf_est.utils import calc_fwhm


def plot(est, ref, filename):

    left = (len(est) - len(ref)) // 2
    right = len(est) - len(ref) - left
    ref = np.pad(ref, (left, right))

    font = {'size': 8}
    matplotlib.rc('font', **font)

    est_hm = np.max(est) / 2
    est_fwhm, est_left, est_right = calc_fwhm(est)

    ref_hm = np.max(ref) / 2
    ref_fwhm, ref_left, ref_right = calc_fwhm(ref)

    dpi = 100
    figx = 168
    figy = 120

    figl = 0.24
    figr = 0.01
    figb = 0.20
    figt = 0.05
    position = [figl, figb, 1 - figl - figr, 1 - figb - figt]

    fig = plt.figure(figsize=(figx/dpi, figy/dpi), dpi=dpi)
    ax = fig.add_subplot(111, position=position)

    plt.plot(ref, '-', color='tab:red')
    plt.plot([ref_left, ref_right], [ref_hm] * 2, '--o', color='tab:red',
             markersize=5)

    plt.plot(est, '-', color='tab:blue')
    plt.plot([est_left, est_right], [est_hm] * 2, '--o', color='tab:blue',
             markersize=5)

    ylim = ax.get_ylim()
    print(np.diff(ylim)[0] / ((1 - figt - figb) * figy))
    offset = np.diff(ylim)[0] / ((1 - figt - figb) * figy) * 12
    print(offset)
    tl = np.max((est_right, ref_right)) + 1.5
    est_tv = (est_hm + ref_hm) * 0.5 + offset / 2
    ref_tv = (est_hm + ref_hm) * 0.5 - offset / 2

    plt.text(tl, ref_tv, '%.2f' % ref_fwhm, color='tab:red',
             va='center')
    plt.text(tl, est_tv, '%.2f' % est_fwhm, color='tab:blue',
             va='center')

    plt.xticks(np.arange(0, len(est), 4))
    plt.yticks([0, 0.05, 0.10])

    plt.savefig(filename)


if __name__ == '__main__':
    
    est_dirname =  '../tests/results_isbi2021_simu_final'
    est_basename = 'simu_type-gauss_fwhm-8p0_scale-0p25_len-13_smooth-1.0/kernel/avg_epoch-30000.npy'
    est_filename = Path(est_dirname, est_basename)
    est = np.load(est_filename).squeeze()

    ref_dirname = '/data/phantom/simu'
    ref_basename = 'SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-8p0_scale-0p25_len-13_kernel.npy'
    ref_filename = Path(ref_dirname, ref_basename)
    ref = np.load(ref_filename).squeeze()
    
    plot(est, ref, 'gauss_kernel.pdf')

    est_basename = 'simu_type-rect_fwhm-9p0_scale-0p25_len-13_smooth-1.0/kernel/avg_epoch-30000.npy'
    est_filename = Path(est_dirname, est_basename)
    est = np.load(est_filename).squeeze()

    ref_dirname = '/data/phantom/simu'
    ref_basename = 'SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-rect_fwhm-9p0_scale-0p25_len-13_kernel.npy'
    ref_filename = Path(ref_dirname, ref_basename)
    ref = np.load(ref_filename).squeeze()

    plot(est, ref, 'rect_kernel.pdf')
