#!/usr/bin/env python

import numpy as np
import torch

from psf_est.loss import GANLoss, SmoothnessLoss, CenterLoss


def test_loss():
    x = np.ones([10, 1, 1]) * (-1)
    gan_loss_func = GANLoss().cuda()
    gan_loss = gan_loss_func(torch.tensor(x).cuda(), False)
    ref = -np.log(1 - 1 / (1 + np.exp(-x))).mean()
    assert np.allclose(ref, gan_loss.cpu().detach().numpy())

    gan_loss = gan_loss_func(torch.tensor(x).cuda(), True)
    ref = -np.log(1 / (1 + np.exp(-x))).mean()
    assert np.allclose(ref, gan_loss.cpu().detach().numpy())

    kernel = torch.tensor([1, 0, 2, -1, 4, 3], dtype=torch.float32)
    kernel = kernel[None, None, ..., None]
    smoothness_loss_func = SmoothnessLoss()
    smoothness_loss = smoothness_loss_func(kernel)
    norm = 1 ** 2 + 2 ** 2  +3 ** 2 + 5 ** 2 + 1 ** 2
    assert smoothness_loss == norm

    kernel = torch.tensor([0.6, 0.1, 0.3], dtype=torch.float32)
    center_loss_func = CenterLoss(len(kernel))
    center_loss = center_loss_func(kernel)
    assert np.allclose(center_loss.item(), 0.09)


if __name__ == '__main__':
    test_loss()
