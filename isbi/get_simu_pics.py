#!/usr/bin/env python

import nibabel as nib
import numpy as np
from pathlib import Path
from PIL import Image
from image_processing_3d import quantile_scale
from scipy.ndimage import zoom


dirname = '/data/phantom/simu'
basename = 'SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-gauss_fwhm-8p0_scale-0p25_len-13.nii'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

th_slice_ind = 128
in_slice_ind = 22

factor = 4

ysize = 119
xsize = 60
xstart = 60
ystart = 40

in_im = data[:, :, in_slice_ind].astype(np.uint8).T
in_im = in_im[xstart : xstart + xsize, ystart : ystart + ysize]

xstart = 80
ystart = 40
th_im = data[th_slice_ind, :, :].astype(np.uint8).T
th_im = zoom(th_im, (factor, 1), order=0, prefilter=False)
th_im = th_im[xstart : xstart + xsize, ystart : ystart + ysize]

in_im = Image.fromarray(in_im)
in_im.save('gauss_in.png')
th_im = Image.fromarray(th_im)
th_im.save('gauss_th.png')

dirname = '/data/phantom/simu'
basename = 'SUPERRES-ADNIPHANTOM_20200711_PHANTOM-T2-TSE-3D-CORONAL-PRE-ACQ1-01mm_resampled_type-rect_fwhm-9p0_scale-0p5_len-13.nii'
filename = Path(dirname, basename)
obj = nib.load(filename)
data = obj.get_fdata(dtype=np.float32)
data = quantile_scale(data, upper_pct=0.99, upper_th=1) * 255

factor = 2
in_slice_ind = 45
xstart = 80

xstart = 60
ystart = 40
in_im = data[:, :, in_slice_ind].astype(np.uint8).T
in_im = in_im[xstart : xstart + xsize, ystart : ystart + ysize]

xstart = 80
ystart = 40
th_im = data[th_slice_ind, :, :].astype(np.uint8).T
th_im = zoom(th_im, (factor, 1), order=0, prefilter=False)
th_im = th_im[xstart : xstart + xsize, ystart : ystart + ysize]

in_im = Image.fromarray(in_im)
in_im.save('rect_in.png')
th_im = Image.fromarray(th_im)
th_im.save('rect_th.png')
