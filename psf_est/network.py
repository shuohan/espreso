import torch
import numpy as np
from torch import nn

from .config import Config


class KernelNet1d(nn.Sequential):
    """The network to learn a 1D blur point-spread function (PSF) from 1D data.

    Attributes:
        impulse (torch.Tensor): The impulse function to calculate the PSF.

    """
    def __init__(self):
        super().__init__()
        self.register_buffer('impulse', self._get_impulse())
        self._kernel_cuda = None
        in_ch = 1
        for i, ks in enumerate(Config().kn_kernel_sizes):
            out_ch = self._calc_out_ch(i)
            conv = self._create_conv(in_ch, out_ch, ks)
            self.add_module('conv%d' % i, conv)
            in_ch = out_ch

    def _get_impulse(self):
        """Returns the impulse function as the input."""
        impulse_shape = 2 * self.calc_kernel_size() - 1
        impulse = torch.zeros([1, 1, impulse_shape]).float()
        impulse[:, :, impulse_shape // 2] = 1
        return impulse

    def _calc_out_ch(self, i):
        """Calculates the number of output channels for a conv."""
        config = Config()
        if i < len(config.kn_kernel_sizes) - 1 :
            return config.kn_num_channels
        else:
            return 1

    def _create_conv(self, in_ch, out_ch, ks):
        """Creates a conv layer."""
        return nn.Conv1d(in_ch, out_ch, ks, bias=False)

    def calc_input_size_reduce(self):
        """Calculates the number of pixels reduced from the input image.

        Returns:
            int: The number of reduced pixels.

        """
        return np.sum(np.array(Config().kn_kernel_sizes) - 1)

    def calc_kernel_size(self):
        """Calculates the size of the PSF according to the size of conv weights.

        The equation is from
        https://distill.pub/2019/computing-receptive-fields/

        Returns:
            int: The size of the PSF.

        """
        return self.calc_input_size_reduce() + 1

    def calc_kernel(self):
        """Calculates the current kernel.

        Note:
            Call :meth:`kernel` to return the calculated kernel.

        Returns:
            KernelNet1d: The instance itself.

        """
        self._kernel_cuda = self(self.impulse)
        return self

    @property
    def kernel_cuda(self):
        """Returns the current kernel on CUDA."""
        if self._kernel_cuda is None:
            self.calc_kernel()
        return self._kernel_cuda

    @property
    def kernel(self):
        """Returns the current kernel on CPU."""
        return self.kernel_cuda.detach().cpu()


class KernelNet2d(KernelNet1d):
    """The network to learn a 1D blur point-spread function (PSF) from 2D data.

    """
    def _create_conv(self, in_ch, out_ch, ks):
        return nn.Conv2d(in_ch, out_ch, (ks, 1), bias=False)

    def _get_impulse(self):
        impulse = super()._get_impulse()[..., None]
        return impulse


class LowResDiscriminator1d(nn.Sequential):
    """Discriminator for 1D low resolution patches.

    """
    def __init__(self):
        super().__init__()
        config = Config()
        in_ch = 1
        out_ch = config.lrd_num_channels
        for i in range(config.lrd_num_convs):
            if i < config.lrd_num_convs - 1:
                conv = self._create_early_conv(in_ch, out_ch)
                conv = nn.utils.spectral_norm(conv)
                self.add_module('conv%d' % i, conv)
                relu = nn.LeakyReLU(config.lrelu_neg_slope)
                self.add_module('relu%d' % i, relu)
            else:
                conv = self._create_final_conv(in_ch)
                conv = nn.utils.spectral_norm(conv)
                self.add_module('conv%d' % i, conv)
            in_ch = out_ch
            out_ch = in_ch * 2

    def _create_early_conv(self, in_ch, out_ch):
        """Creates a conv layer which is not the final one."""
        return nn.Conv1d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)

    def _create_final_conv(self, in_ch):
        """Creates the final conv layer."""
        return nn.Conv1d(in_ch, 1, 4, stride=1, padding=0, bias=False)


class LowResDiscriminator2d(LowResDiscriminator1d):
    """Discriminator for 2D low resolution patches.

    """
    def _create_early_conv(self, in_ch, out_ch):
        """Creates a conv layer which is not the final one."""
        return nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)

    def _create_final_conv(self, in_ch):
        """Creates the final conv layer."""
        return nn.Conv2d(in_ch, 1, 4, stride=1, padding=0, bias=False)
