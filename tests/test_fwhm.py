#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from psf_est.utils import calc_fwhm
from lr_simu.kernel import create_gaussian_kernel


def test_fwhm():
    dirname = Path('results_fwhm')
    dirname.mkdir(exist_ok=True)

    kernel = create_gaussian_kernel(1 / 8)
    fwhm, left, right = calc_fwhm(kernel)
    print(fwhm, left, right)
    assert np.round(fwhm) == 8

    plt.plot(kernel)
    plt.plot([left] * 2, [0, np.max(kernel)], 'k')
    plt.plot([right] * 2, [0, np.max(kernel)], 'k')
    plt.plot([0, len(kernel) - 1], [np.max(kernel) / 2] * 2, 'k')
    plt.gcf().savefig(dirname.joinpath('kernel.png'))

if __name__ == '__main__':
    test_fwhm()
