"""Handles the configuration of this algorithm.

"""
from singleton_config import Config as Config_


class Config(Config_):
    """The algorithm configuration.

    Attributes:
        kn_num_channels (int): The number of channels for
            :class:`psf_est.network.KernelNet`.
        kn_kernel_sizes (list[int]): The kernel sizes for each of
            :class:`psf_est.network.KernelNet` conv weights.
        lrd_num_convs (int): The number of convolutions in
            :class:`psf_est.network.LowResDiscriminator`.
        lrd_num_channels (int): The number of channels in the first conv of
            :class:`psf_est.network.LowResDiscriminator`.

    """
    def __init__(self):
        super().__init__()
        self.add_config('kn_num_channels', 64)
        self.add_config('kn_kernel_sizes', [7, 5, 3, 1, 1, 1])
        self.add_config('lrd_num_convs', 5)
        self.add_config('lrd_num_channels', 64)
        self.add_config('lrelu_neg_slope', 0.2)
