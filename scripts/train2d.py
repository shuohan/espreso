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

from pytorch_trainer.log import DataQueue, EpochPrinter


args.output = Path(args.output)
args.output.mkdir(parents=True, exist_ok=True)

obj = nib.load(args.input)
image = obj.get_fdata(dtype=np.float32)
if args.scale_factor is None:
    zooms = obj.header.get_zooms()
    args.scale_factor = zooms[2] / zooms[0]

config = Config()
for key, value in args.__dict__.items():
    if hasattr(config, key):
        setattr(config, key, value)

kn = KernelNet2d().cuda()
lrd = LowResDiscriminator2d().cuda()
kn_optim = Adam(kn.parameters(), lr=5e-4)
lrd_optim = Adam(lrd.parameters(), lr=5e-4)

reduce = kn.calc_input_size_reduce() 
hr_patch_size = [config.patch_size[0] + reduce,
                 config.patch_size[1] + reduce, 1]
hr_patches = Patches(image, hr_patch_size)
hr_weights = np.ones(len(hr_patches))
hr_sampler = WeightedRandomSampler(hr_weights, config.batch_size)
hr_loader = DataLoader(hr_patches, batch_size=config.batch_size,
                       sampler=hr_sampler)

lr_patches = Patches(image, config.patch_size, x=2, y=1, z=0,
                     scale_factor=config.scale_factor)
lr_weights = np.ones(len(lr_patches))
lr_sampler = WeightedRandomSampler(lr_weights, config.batch_size)
lr_loader = DataLoader(lr_patches, batch_size=config.batch_size,
                       sampler=lr_sampler)

print(config)

print(kn.calc_kernel().squeeze())

trainer = TrainerHRtoLR(kn, lrd, kn_optim, lrd_optim, hr_loader, lr_loader)
queue = DataQueue(['kn_gan_loss', 'sum_loss', 'kn_tot_loss', 'lrd_tot_loss'])
trainer.register(queue)
printer = EpochPrinter(print_sep=False)
queue.register(printer)

trainer.train()

print(kn.calc_kernel().squeeze())
