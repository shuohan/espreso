import torch
import torch.nn.functional as F

from lr_simu.kernel import create_gaussian_kernel


class GANLoss(torch.nn.Module):
    r"""Loss of the original GAN with cross entropy.

    For the discriminator :math:`D`, this loss minimizes the binary cross
    entropy with logits:

    .. math::

        l = - \mathrm{mean}_x (y \ln(\sigma(D(x)))
          + (1 - y) \ln(1 - \sigma(D(x)))),

    where :math:`\sigma` is the sigmoid function. If
    :math:`x \in \mathrm{\{truth\}}`, we have :math:`y = 1` and

    .. math::

        l = - \mathrm{mean}_x \ln(\sigma(D(x))).

    If :math:`x \in \mathrm{\{generated\}}`, i.e. :math:`\exists z` s.t.
    :math:`x = G(z)` where `G` is the generator, we have :math:`y = 0` and

    .. math::

        l = - \mathrm{mean}_x \ln(1 - \sigma(D(G(z)))).

    Combine the above two terms together, we can get the loss for the
    discriminator.

    For the generator :math:`G`, this loss minimizes the binary cross entropy
    with the same form and :math:`y = 1`, which is to minimize

    .. math::

        l = - \mathrm{mean}_x \ln(\sigma(D(G(z)))).

    This is the modified GAN loss which minimizes
    :math:`l = \mathrm{mean}_x \ln(1 - \sigma(D(G(z))))`.

    """
    def __init__(self):
        super().__init__()

    def forward(self, x, is_real):
        target = torch.ones_like(x) if is_real else torch.zeros_like(x)
        loss = F.binary_cross_entropy_with_logits(x, target)
        return loss


class GaussInitLoss(torch.nn.Module):
    """Trains the kernel network to learn a Gaussian kernel at the beginning.

    This class directly measures the difference between the kernel from 
    by the kernel network and a Gaussian kernel.

    Attributes:
        scale_factor (float): The estimated upsampling factor.
        gauss_kernel (torch.Tensor): The target Gaussian kernel to learn.
        kernel_size (int): The size of the kernel to estimate.

    """
    def __init__(self, scale_factor, kernel_size):
        super().__init__()
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        assert self.kernel_size % 2 == 1
        self.register_buffer('gauss_kernel', self._get_gauss_kernel())

    def forward(self, kernel):
        """Calculates the loss.

        Args:
            kernel (torch.Tensor): The kernel calculated by
                :class:`pst_est.network.KernelNet1d`.

        Returns:
            torch.Tensor: The calculated loss.
        
        """
        return F.mse_loss(kernel, self.gauss_kernel)

    def _get_gauss_kernel(self):
        """Returns the reference Gaussian kernel."""
        scale = 1 / self.scale_factor
        length = self.kernel_size // 2
        kernel = create_gaussian_kernel(scale, length=length)
        kernel = torch.tensor(kernel).float()[None, None, ...]
        return kernel


class GaussInitLoss1d(GaussInitLoss):
    """Trains the kernel network to learn a Gaussian kernel at the beginning.

    In contrast to :class:`GaussInitLoss`, this loss uses high-resolution
    patches and compares the output of :class:`psf_est.network.KernelNet1d` with
    the output of a Gaussian kernel.

    """
    def forward(self, kernel_net, x):
        out_net = kernel_net(x)
        out_gauss = self._gauss_kernel_conv(x)
        loss = F.mse_loss(out_net, out_gauss)
        return loss

    def _gauss_kernel_conv(self, x):
        return F.conv1d(x, self.gauss_kernel)


class GaussInitLoss2d(GaussInitLoss1d):
    """Trains the kernel network to learn a Gaussian kernel at the beginning.

    In contrast to :class:`GaussInitLoss1d`, this loss takes a 2D image as input
    to "initialize" :class:`psf_est.network.KernelNet2d`.

    """
    def _get_gauss_kernel(self):
        kernel = super()._get_gauss_kernel()
        kernel = kernel[..., None]
        return kernel

    def _gauss_kernel_conv(self, x):
        return F.conv2d(x, self.gauss_kernel)


class SumLoss(torch.nn.Module):
    """Encourages the sum of the kernel is equal to 1.

    """
    def forward(self, kernel):
        sum = torch.sum(kernel)
        one = torch.ones_like(sum)
        loss = self._calc_loss(sum, one)
        return loss

    def _calc_loss(self, sum, one):
        return F.l1_loss(sum, one)


class SumLossMSE(SumLoss):
    """Uses MSE to encourages the sum of the kernel is equal to 1.

    """
    def _calc_loss(self, sum, one):
        return F.mse_loss(sum, one)
