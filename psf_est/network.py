import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from .config import Config


class KernelNet2d(nn.Sequential):
    """The network to learn a 1D blur point-spread function (PSF) from 1D data.

    Attributes:
        impulse (torch.Tensor): The impulse function to calculate the PSF.

    """
    def __init__(self):
        super().__init__()

        num_ch = 1024
        num_linears = 3

        self.input_tensor = torch.zeros(1, num_ch, dtype=torch.float32)
        self.input_tensor = nn.Parameter(self.input_tensor)
        self.kernel_size = 21

        for i in range(num_linears):
            linear = nn.Linear(num_ch, num_ch)
            self.add_module('linear%d' % i, linear)
            # self.add_module('relu%d' % i, nn.ReLU6())

        linear = nn.Linear(num_ch, self.kernel_size)
        self.add_module('linear%d' % num_linears, linear)
        self.softmax = nn.Softmax(dim=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.input_tensor, a=np.sqrt(5))

    def calc_input_size_reduce(self):
        """Calculates the number of pixels reduced from the input image.

        Returns:
            int: The number of reduced pixels.

        """
        return self.kernel_size - 1

    @property
    def kernel_cuda(self):
        kernel = super().forward(self.input_tensor) 
        kernel = kernel.view(1, 1, -1, 1)
        return kernel

    @property
    def kernel(self):
        """Returns the current kernel on CPU."""
        return self.kernel_cuda.detach().cpu()

    def forward(self, x):
        kernel = self.kernel_cuda
        return F.conv2d(x, kernel)


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
        self.sigmoid = nn.Sigmoid()

    def _create_early_conv(self, in_ch, out_ch):
        """Creates a conv layer which is not the final one."""
        return nn.Conv1d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)

    def _create_final_conv(self, in_ch):
        """Creates the final conv layer."""
        return nn.Conv1d(in_ch, 1, 1, stride=1, padding=0, bias=False)


class LowResDiscriminator2d(LowResDiscriminator1d):
    """Discriminator for 2D low resolution patches.

    """
    def _create_early_conv(self, in_ch, out_ch):
        """Creates a conv layer which is not the final one."""
        return nn.Conv2d(in_ch, out_ch, (4, 1), stride=(1, 2), padding=0, bias=True)

    def _create_final_conv(self, in_ch):
        """Creates the final conv layer."""
        return nn.Conv2d(in_ch, 1, (4, 1), stride=(1, 2), padding=0, bias=True)

    def forward(self, x):
        output = super().forward(x)
        return output
