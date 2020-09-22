#!/usr/bin/env python

import torch
from pathlib import Path
from pytorchviz import make_dot

from psf_est.network import KernelNet1d, KernelNet2d 
from psf_est.network import LowResDiscriminator1d, LowResDiscriminator2d


def test_networks():
    dirname = Path('results_networks')
    dirname.mkdir(exist_ok=True)
    patch1d = torch.rand(2, 1, 64).cuda()
    patch2d = torch.rand(2, 1, 64, 64).cuda()

    kn = KernelNet1d().cuda()
    print(kn)
    assert kn(patch1d).shape == (2, 1, 52)
    assert kn.calc_input_size_reduce() == 12
    kn_dot = make_dot(patch1d, kn)
    kn_dot.render(dirname.joinpath('kn1d'))

    kn = KernelNet2d().cuda()
    print(kn)
    assert kn(patch2d).shape == (2, 1, 52, 64)
    assert kn.calc_input_size_reduce() == 12
    kn_dot = make_dot(patch2d, kn)
    kn_dot.render(dirname.joinpath('kn2d'))

    lrd = LowResDiscriminator1d().cuda()
    print(lrd)
    assert lrd(patch1d).shape == (2, 1, 1)
    lrd_dot = make_dot(patch1d, lrd)
    lrd_dot.render(dirname.joinpath('lrd1d'))

    lrd = LowResDiscriminator2d().cuda()
    print(lrd)
    assert lrd(patch2d).shape == (2, 1, 1, 1)
    lrd_dot = make_dot(patch2d, lrd)
    lrd_dot.render(dirname.joinpath('lrd2d'))


if __name__ == '__main__':
    test_networks()
