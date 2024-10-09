#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FISTA class and subclass ISTA. Algorithms provide encoder for given decoder.

@author: Benjamin Cowen
@date: Feb 24 2018 (updated Feb 2023)
@contact: benjamin.cowen.math@gmail.com
"""

from lib.model_blocks.AlgorithmBlock import AlgorithmBlock
from lib.model_blocks.dictionary import Dictionary
import torch
import torch.nn.functional as th_fcn
import copy


class FISTA(AlgorithmBlock):
    """
    FISTA for vector valued data
      minimizes F with respect to x:
         F(x) = .5||y-Ax||_2^2 + sparsity_weight*||x||_1
      where A is a sparse transform matrix.
    """

    def __init__(self,
                 data_len: int,
                 code_len: int,
                 n_iter: int,
                 sparsity_weight: float,
                 trainable: bool = False,
                 init_dict: AlgorithmBlock = None,
                 non_blocking: bool = True,
                 device: str = 'cpu'):
        """
        data_len: scalar, length of one data sample
        code_len: scalar, length of the code of one data sample
        n_iters: number of iterations to run FISTA
        sparsity_weight: balancing weight for L1
        """
        super(FISTA, self).__init__()

        # Logistics
        self.data_len = data_len
        self.code_len = code_len
        self.trainable = trainable

        # Initialize
        self.n_iters = n_iter
        self.sparsity_weight = sparsity_weight
        self.loss_hist = torch.full((self.n_iters,), torch.nan)

        # Dictionary Initialization
        self._device = device
        if init_dict is not None:
            self.decoder = copy.deepcopy(init_dict)
        else:
            self.decoder = Dictionary(code_len, data_len, device, non_blocking)
            self.decoder.normalize_columns()

        # Encoder Initialization
        self.encoder = torch.nn.Linear(self.data_len, self.code_len, bias=False, device=self._device)
        self.encoder.weight.data = self.decoder.We()
        self._max_eig = self.decoder.get_max_eig_val()
        self.set_grad()

    def set_grad(self):
        if self.trainable:
            self.turn_grad_on()
        else:
            self.turn_grad_off()

    def update_encoder_with_dict(self, decoder):
        self.decoder = decoder
        self.encoder.weight.data = self.decoder.We()
        self._max_eig = self.decoder.get_max_eig_val()

    def Wd(self):
        return self.decoder.Wd()

    def We(self):
        return self.encoder.weight.data.detach()

    def forward(self, data, n_iters=None):
        if n_iters is None:
            n_iters = self.n_iters
        # Initializations.
        self.loss_hist = torch.full((n_iters,), torch.nan)
        yk = self.encoder(data)
        xprev = torch.zeros(yk.shape).to(self._device)
        t = 1
        # Iterations.
        for it in range(n_iters):
            residual = self.decoder(yk) - data
            # ISTA step
            tmp = yk - self.encoder(residual) / self._max_eig
            xk = th_fcn.softshrink(tmp, lambd=self.sparsity_weight / self._max_eig)
            # FISTA stepsize update:
            tnext = (1 + (1 + 4 * (t ** 2)) ** .5) / 2
            fact = (t - 1) / tnext
            # Use momentum to update code estimate.
            yk = xk + (xk - xprev) * fact
            # Keep track for the next iter stuff
            xprev = xk
            t = tnext
            self.recordLoss(data, xk, it)
        return yk

    def get_copy(self, n_iters=None, trainable=False):
        """
        Return a new initialization of this class that implements the same algorithm (same weights),
            but with a different number of iterations and may/may not be trainable.
        """
        new_encoder = copy.deepcopy(self)
        if n_iters is not None:
            new_encoder.n_iters = n_iters
        new_encoder.trainable = trainable
        new_encoder.set_grad()

        return new_encoder

    @torch.no_grad()
    def recordLoss(self, data, xk, it):
        """ MSE between data,xk + L1 norm of xk, averaged over batchsize"""
        self.loss_hist[it] = ((0.5 * (data - self.decoder(xk)).pow(2).sum().item() +
                                   xk.abs().sum().detach().item())) / data.shape[0]


class ISTA(FISTA):
    """ ISTA """

    def __init__(self, **kwargs):
        super(ISTA, self).__init__(**kwargs)
        # Same init as FISTA except some normalization, and S matrix.
        self.S = torch.nn.Linear(self.code_len, self.code_len, bias=False)
        with torch.no_grad():
            self.S.weight.data = torch.eye(self.code_len) - torch.mm(self.We(), self.Wd()) / self._max_eig
            self.encoder.weight.data /= self._max_eig
        self.set_grad()

    def forward(self, data):
        # Initializations.
        self.loss_hist = torch.full((self.n_iters,), torch.nan)
        noisy_code = self.encoder(data)
        xk = torch.zeros(noisy_code.shape)
        # Iterations.
        for it in range(self.n_iters):
            xk = th_fcn.softshrink(noisy_code + self.S(xk),
                                   lambd=self.sparsity_weight / (2*self._max_eig))
            self.recordLoss(data, xk, it)
        return xk
