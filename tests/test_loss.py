#!/usr/bin/env python

import numpy as np
import torch
from lr_simu.kernel import create_gaussian_kernel

from psf_est.loss import GANLoss, SumLoss, SumLossMSE
from psf_est.loss import GaussInitLoss, GaussInitLoss1d, GaussInitLoss2d
from psf_est.network import KernelNet1d, KernelNet2d


def test_loss():
    x = np.ones([10, 1, 1]) * (-1)
    gan_loss_func = GANLoss().cuda()
    gan_loss = gan_loss_func(torch.tensor(x).cuda(), False)
    ref = -np.log(1 - 1 / (1 + np.exp(-x))).mean()
    assert np.allclose(ref, gan_loss.cpu().detach().numpy())

    gan_loss = gan_loss_func(torch.tensor(x).cuda(), True)
    ref = -np.log(1 / (1 + np.exp(-x))).mean()
    assert np.allclose(ref, gan_loss.cpu().detach().numpy())

    kernel = torch.ones([1, 1, 13]) * 0.5
    sum_loss_func = SumLoss()
    sum_loss = sum_loss_func(kernel)
    assert torch.allclose(sum_loss, torch.tensor(5.5))

    sum_loss_func = SumLossMSE()
    sum_loss = sum_loss_func(kernel)
    assert torch.allclose(sum_loss, torch.tensor(5.5 ** 2))

    scale = 11.2
    kernel_size = 13
    gauss = create_gaussian_kernel(1 / scale, length=kernel_size//2)
    gauss_loss_func = GaussInitLoss(scale, kernel_size).cuda()
    assert torch.equal(torch.tensor(gauss)[None, None, ...].float(),
                       gauss_loss_func.gauss_kernel.cpu())
    diff = np.mean((gauss - kernel.numpy()) ** 2)
    gauss_loss = gauss_loss_func(kernel.cuda())
    assert np.isclose(diff, gauss_loss.item())

    im1d = torch.rand(10, 1, 64).cuda()
    net1d = KernelNet1d().cuda()
    gauss_loss_func = GaussInitLoss1d(scale, kernel_size).cuda()
    gauss_loss = gauss_loss_func(net1d, im1d)
    print(gauss_loss)

    im2d = torch.rand(10, 1, 64, 64).cuda()
    net2d = KernelNet2d().cuda()
    gauss_loss_func = GaussInitLoss2d(scale, kernel_size).cuda()
    assert gauss_loss_func.gauss_kernel.shape == (1, 1, kernel_size, 1) 
    gauss_loss = gauss_loss_func(net2d, im2d)
    print(gauss_loss)


if __name__ == '__main__':
    test_loss()
