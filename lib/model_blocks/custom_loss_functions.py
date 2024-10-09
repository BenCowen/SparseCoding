"""
Custom loss functions

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""

import torch
from abc import ABC, abstractmethod
from lib.UTILS.path_support import import_class_kwargs


class YamlLoss(torch.nn.Module, ABC):
    def __init__(self, input_keys, device="cuda:0", weight=1, batch_norm=False):
        super(YamlLoss, self).__init__()
        # These keys will be taken from an input dictionary and passed as
        # ARGS to the forward function.
        self.input_keys = input_keys
        self.weight = weight
        self.do_batch_norm = batch_norm

    @abstractmethod
    def loss(self, input_dict):
        pass

    def forward(self, input_dict):
        """
        Unpack only the desired input_dict elements as args into the loss.
        """
        # Divide weight by batchsize if necessary...
        bsz = input_dict[self.input_keys[0]].shape[0]
        weight = self.weight / bsz if self.do_batch_norm else self.weight
        return weight * self.loss(*[input_dict[key] for key in self.input_keys])


class MSE(YamlLoss):

    def __init__(self, input_keys, **kwargs):
        super(MSE, self).__init__(input_keys, **kwargs)
        self.fcn = torch.nn.MSELoss()

    def loss(self, recon, target):
        return self.fcn(recon, target)


class L1(YamlLoss):

    def __init__(self, input_keys, **kwargs):
        super(L1, self).__init__(input_keys, **kwargs)

    def loss(self, code):
        return code.abs().sum()


class TV(L1):
    def __init__(self, input_keys, **kwargs):
        super(TV, self).__init__(input_keys, **kwargs)

    def forward(self, inputs):
        # only do last 2 dims...
        cdim = self.super(inputs[:, :, :-1, :] - inputs[:, :, 1:, :])
        rdim = self.super(inputs[:, :, :, :-1] - inputs[:, :, :, 1:])
        return cdim + rdim


class Kld2N01(YamlLoss):
    """
    https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    Computes KL Divergence to N(0, 1)
    KL(N(\mu, \sigma), N(0, 1))"""

    def __init__(self, *args, **kwargs):
        super(Kld2N01, self).__init__()

    def forward(self, log_var, mu):
        kld_loss = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(),
            dim=1), dim=0)
        return kld_loss
