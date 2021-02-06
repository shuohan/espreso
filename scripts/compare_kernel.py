#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-kernel', help='Input image.')
parser.add_argument('-r', '--reference-kernel', help='Input image.')
parser.add_argument('-o', '--output-filename', help='Output directory.')
args = parser.parse_args()


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ssp.utils import calc_fwhm


def plot_kernel(ax, kernel, fwhm, left, right, vp=0.5, color='b'):
    ax.plot(kernel, color, marker='o')
    ax.plot([left] * 2, [0, np.max(kernel)], color + '--')
    ax.plot([right] * 2, [0, np.max(kernel)], color + '--')
    # ax.plot([0, len(kernel)-1], [np.max(kernel)/2] * 2, color + '--')
    ax.text((left + right) / 2, vp * np.max(kernel), '%.4f' % fwhm,
            ha='center', color=color)


Path(args.output_filename).parent.mkdir(exist_ok=True, parents=True)

in_kernel = np.load(args.input_kernel).squeeze()

ref_kernel = np.load(args.reference_kernel).squeeze()
left_padding = (len(in_kernel) - len(ref_kernel)) // 2
right_padding = len(in_kernel) - len(ref_kernel) - left_padding
ref_kernel = np.pad(ref_kernel, (left_padding, right_padding))

in_fwhm, in_left, in_right = calc_fwhm(in_kernel)
ref_fwhm, ref_left, ref_right = calc_fwhm(ref_kernel)

kernel_diff = np.sum(np.abs(in_kernel - ref_kernel))
fwhm_diff = np.abs(in_fwhm - ref_fwhm)
title = 'kernel abs diff %.4f, fwhm abs diff %.4f' % (kernel_diff, fwhm_diff)

fig = plt.figure()
ax = fig.add_subplot(111)
plot_kernel(ax, in_kernel, in_fwhm, in_left, in_right, vp=0.5, color='b')
plot_kernel(ax, ref_kernel, ref_fwhm, ref_left, ref_right, vp=0.25, color='r')
plt.title(title)

plt.grid(True)
plt.tight_layout()
fig.savefig(args.output_filename)
