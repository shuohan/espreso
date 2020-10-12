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


class SmoothnessLoss(torch.nn.Module):
    """L2 norm of derivative."""
    def forward(self, kernel):
        device = kernel.device
        operator = torch.tensor([1, -1], dtype=torch.float32, device=device)
        operator = operator[None, None, ..., None]
        derivative = F.conv2d(kernel, operator)
        loss = torch.sum(derivative ** 2)
        return loss


class CenterLoss(torch.nn.Module):
    """Penalizes off-center."""
    def __init__(self, kernel_length):
        super().__init__()
        self.kernel_length = kernel_length
        center = torch.tensor(self.kernel_length // 2, dtype=torch.float32)
        self.register_buffer('center', center)
        locs = torch.arange(self.kernel_length, dtype=torch.float32)
        self.register_buffer('locs', locs)

    def forward(self, kernel):
        kernel_center = torch.sum(kernel.squeeze() * self.locs)
        loss = F.mse_loss(kernel_center, self.center)
        return loss
