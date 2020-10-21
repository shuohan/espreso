#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input image.')
parser.add_argument('-o', '--output', help='Output directory.')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='The number of samples per mini-batch.')
parser.add_argument('-s', '--scale-factor', default=None, type=float,
                    help='Super resolution scale factor.')
parser.add_argument('-e', '--num-epochs', default=10000, type=int,
                    help='The number of epochs (iterations).')
parser.add_argument('-iss', '--image-save-step', default=100, type=int,
                    help='The image saving step.')
parser.add_argument('-k', '--true-kernel', default=None)
parser.add_argument('-l', '--kernel-length', default=21, type=int)
parser.add_argument('-na', '--no-aug', action='store_true')
parser.add_argument('-w', '--num-workers', default=0, type=int)
parser.add_argument('-z', '--z-axis', default=2, type=int)
parser.add_argument('-isz', '--image-save-zoom', default=1, type=int)
args = parser.parse_args()


import os
import nibabel as nib
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler
import warnings

from sssrlib.patches import Patches, PatchesOr
from sssrlib.transform import create_rot_flip
from psf_est.config import Config
from psf_est.train import TrainerHRtoLR, KernelSaver, KernelEvaluator
from psf_est.network import KernelNet2d, LowResDiscriminator2d
from psf_est.utils import calc_patch_size

from pytorch_trainer.log import DataQueue, EpochPrinter, EpochLogger
from pytorch_trainer.save import ImageSaver


warnings.filterwarnings("ignore")

args.output = Path(args.output)
args.output.mkdir(parents=True, exist_ok=True)
im_output = args.output.joinpath('patches')
kernel_output = args.output.joinpath('kernel')
log_output = args.output.joinpath('loss.csv')
eval_log_output = args.output.joinpath('eval_loss.csv')
config_output = args.output.joinpath('config.json')

xy = [0, 1, 2]
xy.remove(args.z_axis)
obj = nib.load(args.input)
image = obj.get_fdata(dtype=np.float32)
if args.scale_factor is None:
    zooms = obj.header.get_zooms()
    args.scale_factor = float(zooms[args.z_axis] / zooms[xy[0]])
    if zooms[xy[0]] != zooms[xy[1]] and not args.no_aug:
        raise RuntimeError('The resolutions of x and y are different.')
if args.scale_factor < 1:
    raise RuntimeError('Scale factor should be greater or equal to 1.')

config = Config()
nz = image.shape[args.z_axis]
hr_ps, lr_ps= calc_patch_size(config.patch_size, args.scale_factor, nz)

for key, value in args.__dict__.items():
    if hasattr(config, key):
        setattr(config, key, value)
config.add_config('input_image', os.path.abspath(str(args.input)))
config.add_config('output_dirname', os.path.abspath(str(args.output)))

kn = KernelNet2d().cuda()
lrd = LowResDiscriminator2d().cuda()
kn_optim = Adam(kn.parameters(), lr=2e-4, betas=(0.5, 0.999),
                weight_decay=config.weight_decay)
lrd_optim = Adam(lrd.parameters(), lr=2e-4, betas=(0.5, 0.999))

hr_ps = hr_ps + kn.input_size_reduced
config.add_config('hr_patch_size', hr_ps)
config.add_config('lr_patch_size', lr_ps)

print(config)
config.save_json(config_output)

print(kn)
print(lrd)
print(kn_optim)
print(lrd_optim)

transforms = [] if args.no_aug else create_rot_flip()
hr_patches = Patches(image, (hr_ps, hr_ps, 1), transforms=transforms,
                     x=xy[0], y=xy[1], z=args.z_axis).cuda()
hr_loader = hr_patches.get_dataloader(config.batch_size, args.num_workers)

if args.no_aug:
    lr_patches = Patches(image, (lr_ps, hr_ps, 1), x=args.z_axis, y=xy[1], z=xy[0]).cuda()
else:
    lr_patches_xy = Patches(image, (lr_ps, hr_ps, 1), x=args.z_axis, y=xy[1], z=xy[0]).cuda()
    lr_patches_yx = Patches(image, (lr_ps, hr_ps, 1), x=args.z_axis, y=xy[0], z=xy[1]).cuda()
    lr_patches = PatchesOr(lr_patches_xy, lr_patches_yx)
lr_loader = lr_patches.get_dataloader(config.batch_size, args.num_workers)

print('HR patches')
print('----------')
print(hr_patches)
print()
print('LR patches')
print('----------')
print(lr_patches)

trainer = TrainerHRtoLR(kn, lrd, kn_optim, lrd_optim, hr_loader, lr_loader)
queue = DataQueue(['kn_gan_loss', 'smoothness_loss', 'center_loss',
                   'boundary_loss', 'kn_tot_loss', 'lrd_tot_loss'])
printer = EpochPrinter(print_sep=False)
logger = EpochLogger(log_output)

attrs = ['lr', 'hr', 'blur', 'alias']
im_saver = ImageSaver(im_output, attrs=attrs, step=config.image_save_step,
                      file_struct='epoch/sample', save_type='png_norm',
                      save_init=False, prefix='patch',
                      zoom=config.image_save_zoom, ordered=True)

attrs = ['lrd_pred_real', 'lrd_pred_fake', 'lrd_pred_kn']
pred_saver = ImageSaver(im_output, attrs=attrs, step=config.image_save_step,
                        file_struct='epoch/sample', save_type='png',
                        image_type='sigmoid', save_init=False, prefix='lrd',
                        zoom=config.image_save_zoom, ordered=True)

kernel_saver = KernelSaver(kernel_output, step=config.image_save_step,
                           save_init=True)

if args.true_kernel is not None:
    true_kernel = np.load(args.true_kernel)
    evaluator = KernelEvaluator(true_kernel, config.kernel_length).cuda()
    eval_queue = DataQueue(['mae'])
    eval_printer = EpochPrinter(print_sep=False)
    eval_logger = EpochLogger(eval_log_output)
    eval_queue.register(eval_printer)
    eval_queue.register(eval_logger)
    evaluator.register(eval_queue)
    trainer.register(evaluator)

queue.register(printer)
queue.register(logger)
trainer.register(queue)
trainer.register(im_saver)
trainer.register(pred_saver)
trainer.register(kernel_saver)
trainer.train()
