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
        self.add_config('kn_num_channels', 256)
        self.add_config('kn_kernel_sizes', [7, 5, 3, 1, 1, 1])
        self.add_config('lrd_num_convs', 5)
        self.add_config('lrd_num_channels', 64)
        self.add_config('lrelu_neg_slope', 0.2)
        self.add_config('patch_size', (64, 64, 1))
        self.add_config('scale_factor', 1)
        self.add_config('batch_size', 32)
        self.add_config('num_epochs', 1000)
        self.add_config('sum_loss_weight', 10)
        self.add_config('smoothness_loss_weight', 0.5)
        self.add_config('center_loss_weight', 1)
        self.add_config('weight_decay', 1e-5)
        self.add_config('image_save_step', 100)
        self.add_config('num_init_epochs', 100)
