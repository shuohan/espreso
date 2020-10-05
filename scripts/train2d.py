#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input image.')
parser.add_argument('-o', '--output', help='Output directory.')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='The number of samples per mini-batch.')
parser.add_argument('-s', '--scale-factor', default=None, type=float,
                    help='Super resolution scale factor.')
parser.add_argument('-e', '--num-epochs', default=1000, type=int,
                    help='The number of epochs (iterations).')
args = parser.parse_args()


import nibabel as nib
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler                  

from sssrlib.patches import Patches
from sssrlib.transform import create_rot_flip
from psf_est.config import Config
from psf_est.train import TrainerHRtoLR
from psf_est.network import KernelNet2d, LowResDiscriminator2d
from psf_est.utils import pad_patch_size

from pytorch_trainer.log import DataQueue, EpochPrinter
from pytorch_trainer.save import ImageSaver


args.output = Path(args.output)
args.output.mkdir(parents=True, exist_ok=True)
im_output = args.output.joinpath('image')

obj = nib.load(args.input)
image = obj.get_fdata(dtype=np.float32)
if args.scale_factor is None:
    zooms = obj.header.get_zooms()
    args.scale_factor = zooms[2] / zooms[0]

config = Config()
for key, value in args.__dict__.items():
    if hasattr(config, key):
        setattr(config, key, value)
print(config)

kn = KernelNet2d().cuda()
lrd = LowResDiscriminator2d().cuda()
kn_optim = Adam(kn.parameters(), lr=5e-4)
lrd_optim = Adam(lrd.parameters(), lr=5e-4)

hr_patch_size = pad_patch_size(config.patch_size, kn.calc_input_size_reduce())
hr_patches = Patches(image, hr_patch_size)
hr_loader = hr_patches.get_dataloader(config.batch_size)
lr_patches = Patches(image, config.patch_size, x=2, y=1, z=0,
                     scale_factor=config.scale_factor)
lr_loader = lr_patches.get_dataloader(config.batch_size)

trainer = TrainerHRtoLR(kn, lrd, kn_optim, lrd_optim, hr_loader, lr_loader)
queue = DataQueue(['kn_gan_loss', 'sum_loss', 'kn_tot_loss', 'lrd_tot_loss'])
trainer.register(queue)
printer = EpochPrinter(print_sep=False)
queue.register(printer)
im_saver = ImageSaver(im_output, attrs=['lr', 'hr', 'blur', 'alias'],
                      step=10, file_struct='epoch/sample', save_type='png')
trainer.register(im_saver)

trainer.train()
