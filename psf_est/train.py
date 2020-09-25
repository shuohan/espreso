"""Functions and classes to train the algorithm.

"""
import torch
import torch.nn.functional as F

from pytorch_trainer.train import Trainer, Validator, Evaluator
from pytorch_trainer.utils import NamedData
from sssrlib.patches import Patches
from sssrlib.transform import create_rot_flip

from .config import Config
from .loss import GANLoss, SumLoss


class MixinHRtoLR:
    """Mixin class for :class:`TrainerHRtoLR` and :class:`ValidatorHRtoLR`.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_ind = -1

        self._hr = None
        self._hr_cuda = None
        self._hr_names = None

        self._lr = None
        self._lr_cuda = None
        self._lr_names = None

        self._blur_cuda = None
        self._alias_cuda = None

    @property
    def num_batches(self):
        return len(self.dataloader)

    @property
    def batch_size(self):
        return self.dataloader.batch_size

    @property
    def batch_ind(self):
        return self._batch_ind + 1

    @property
    def lr(self):
        """Returns the current named low-resolution patches on CPU."""
        return NamedData(name=self._lr_names, data=self._lr)

    @property
    def hr(self):
        """Returns the current named high-resolution patches on CPU."""
        return NamedData(name=self._hr_names, data=self._hr)

    @property
    def blur(self):
        """Returns the current estimated blurred patches on CPU."""
        return self._blur_cuda.detach().cpu()

    @property
    def alias(self):
        """Returns the current estimated aliased patches on CPU."""
        return self._alias_cuda.detach().cpu()

    def _create_aliasing(self, patches):
        """Creates aliasing on patches."""
        mode = 'linear' if patches.dim() == 3 else 'bilinear'
        results = F.interpolate(patches, self.scale_factor, mode=mode)
        return results

    def __getattr__(self, name):
        stripped_name = name.strip('_')
        if 'loss' in name and hasattr(self, stripped_name):
            return super().__getattr__(stripped_name).item()
        else:
            return super().__getattr__(name)

    def _parse_batch(self, batch):
        self._hr_names = batch[0].name
        self._hr = batch[0].data
        self._lr_names = batch[1].name
        self._lr = batch[1].data

    def _calc_reg(self):
        """Calculates kernel regularization."""
        kernel = self.kernel_net.calc_kernel()
        self._sum_loss = self._sum_loss_func(kernel)
        sum_loss_weight = Config().sum_loss_weight
        return sum_loss_weight * self._sum_loss


class TrainerHRtoLR(MixinHRtoLR, Trainer):
    """Trains point spread function (PSF) estimation.

    Attributes:
        kernel_net (psf_est.network.KernelNet1d): The kernel estimation network.
        lr_disc (psf_est.network.LowResDiscriminator1d): The low-resolution
            patches discriminator.
        kn_optim (torch.optim.Optimizer): The :attr:`kernel_net` optimizer.
        lrd_optim (torch.optim.Optimizer): The :attr:`lr_dics` optimizer.
        dataloader (torch.nn.data.DataLoader): Yields HR and LR patches.
        scale_factor (float): The upsampling scaling factor. It should be
            greater than 1.
    
    """
    def __init__(self, kernel_net, lr_disc, kn_optim, lrd_optim, dataloader,
                 scale_factor, *args, **kwargs):
        super().__init__(Config().num_epochs, *args, **kwargs)
        self.kernel_net = kernel_net
        self.lr_disc = lr_disc
        self.kn_optim = kn_optim
        self.lrd_optim = lrd_optim
        self.dataloader = dataloader
        self.scale_factor = scale_factor

        self._gan_loss_func = GANLoss()
        self._sum_loss_func = SumLoss()

    def get_model_state_dict(self):
        return {'kernel_net': self.kernel_net.state_dict()}

    def get_optim_state_dict(self):
        return {'kn_optim': self.kn_optim.state_dict()}

    def train(self):
        """Trains the algorithm."""
        self.notify_observers_on_train_start()
        for self._epoch_ind in range(self.num_epochs):
            self.notify_observers_on_epoch_start()
            for self._batch_ind, batch in enumerate(self.dataloader):
                self.notify_observers_on_batch_start()
                self._parse_batch(batch)
                self._train_kernel_net()
                self._train_lr_disc()
                self.notify_observers_on_batch_end()
            self.notify_observers_on_epoch_end()
        self.notify_observers_on_train_end()

    def _train_kernel_net(self):
        """Trains the generator :attr:`kernel_net`."""
        self.kn_optim.zero_grad()
        self._blur_cuda = self.kernel_net(self._hr_cuda)
        self._alias_cuda = self._create_aliasing(self._blur_cuda)
        lrd_pred_fake = self.lr_disc.forward(self._alias_cuda)
        self._kn_gan_loss = self._gan_loss_func(lrd_pred_fake, True)
        self._kn_tot_loss = self._kn_loss + self._calc_reg()
        self._kn_tot_loss.backward()
        self.kn_optim.step()

    def _train_lr_disc(self):
        """Trains the low-resolution discriminator :attr:`lr_disc`."""
        self.lrd_optim.zero_grad()
        lrd_pred_real = self.lr_disc(self._lr_cuda)
        lrd_pred_fake = self.lr_disc(self._alias_cuda.detach())
        self._lrd_real_loss = self._gan_loss_func(lrd_pred_real, True)
        self._lrd_fake_loss = self._gan_loss_func(lrd_pred_fake, False)
        self._lrd_tot_loss = self._lrd_real_loss + self._lrd_fake_loss
        self._lrd_tot_loss.backward()
        self.lrd_optim.step()
