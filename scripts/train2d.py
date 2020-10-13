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
args = parser.parse_args()


import nibabel as nib
import numpy as np
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler                  
import warnings

from sssrlib.patches import Patches
from sssrlib.transform import create_rot_flip
from psf_est.config import Config
from psf_est.train import TrainerHRtoLR, KernelSaver, KernelEvaluator
from psf_est.network import KernelNet2d, LowResDiscriminator2d
from psf_est.utils import pad_patch_size

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

obj = nib.load(args.input)
image = obj.get_fdata(dtype=np.float32)
if args.scale_factor is None:
    zooms = obj.header.get_zooms()
    args.scale_factor = float(zooms[2] / zooms[0])

config = Config()
for key, value in args.__dict__.items():
    if hasattr(config, key):
        setattr(config, key, value)
print(config)
config.save_json(config_output)

kn = KernelNet2d().cuda()
lrd = LowResDiscriminator2d().cuda()
kn_optim = Adam(kn.parameters(), lr=2e-4, betas=(0.5, 0.999),
                weight_decay=config.weight_decay)
lrd_optim = Adam(lrd.parameters(), lr=2e-4, betas=(0.5, 0.999))

print(kn)
print(lrd)
print(kn_optim)
print(lrd_optim)

hr_patch_size = pad_patch_size(config.patch_size, kn.input_size_reduced)
hr_patches = Patches(image, hr_patch_size).cuda()
hr_loader = hr_patches.get_dataloader(config.batch_size)
lr_patches = Patches(image, config.patch_size, x=2, y=1, z=0,
                     scale_factor=config.scale_factor).cuda()
lr_loader = lr_patches.get_dataloader(config.batch_size)

trainer = TrainerHRtoLR(kn, lrd, kn_optim, lrd_optim, hr_loader, lr_loader)
queue = DataQueue(['kn_gan_loss', 'smoothness_loss', 'center_loss',
                   'boundary_loss', 'kn_tot_loss', 'lrd_tot_loss'])
printer = EpochPrinter(print_sep=False)
logger = EpochLogger(log_output)
attrs = ['lr', 'hr', 'blur', 'alias', 'lrd_pred_real', 'lrd_pred_fake',
         'lrd_pred_kn']
im_saver = ImageSaver(im_output, attrs=attrs, step=config.image_save_step,
                      file_struct='epoch/sample', save_type='png',
                      save_init=False)
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
trainer.register(kernel_saver)
trainer.train()
