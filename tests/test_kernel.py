#!/usr/bin/env python

import matplotlib.pyplot as plt
from pathlib import Path

from psf_est.network import KernelNet1d


def test_kernel():
    dirname = Path('results_kernel')
    dirname.mkdir(exist_ok=True)
    filename = dirname.joinpath('kernel.png')

    net = KernelNet1d().cuda()
    assert net.impulse.shape == (1, 1, 25)
    assert str(net.impulse.device) == 'cuda:0'
    kernel = net.calc_kernel().kernel
    assert kernel.shape == (1, 1, 13)
    kernel = kernel.cpu().detach().numpy().squeeze()
    fig = plt.figure()
    plt.plot(kernel)
    plt.grid(True)
    fig.savefig(filename)


if __name__ == '__main__':
    test_kernel()
