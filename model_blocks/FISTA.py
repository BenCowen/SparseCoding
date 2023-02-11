#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Cowen
@date: Feb 24 2018 (updated Feb 2023)
@contact: benjamin.cowen.math@gmail.com
"""

from model_blocks.AlgorithmBlock import AlgorithmBlock
import torch
import torch.nn.functional as th_fcn


class FISTA(AlgorithmBlock):
    """
    Fast ISTA for vector valued data
      minimizes F with respect to x:
         F(x) = .5||y-Ax||_2^2 + sparsity_weight*||x||_1
      where A is a sparse transform matrix.
    """

    def __init__(self, data_len, code_len, n_iters, sparsity_weight,
                 trainable=False, init_dict=None, device='cpu'):
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

        # Initialize
        self.nIter = n_iters
        self.sparsity_weight = sparsity_weight
        self.loss_hist = torch.full((n_iters,), torch.nan)

        # Dictionary Initialization
        self._device = device
        self.We = torch.nn.Linear(data_len, code_len, bias=False, device=self._device)

        if init_dict is not None:
            self.replace_dictionary(init_dict)
        else:
            self.normalizeColumns()
            self.max_eig = self.est_max_eig_val()

        if not trainable:
            self.turn_grad_off()

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False

    def get_copy(self, n_iters, trainable=False):
        """
        Return a new initialization of this class with the same weights but a different number of iterations
        """
        return FISTA(self.data_len, self.code_len, n_iters,
                     self.sparsity_weight, init_dict=self.We.weight.data.detach(),
                     device=self._device, trainable=trainable)
    
    def WeT(self, inputs):
        """ Transpose propagation through We"""
        return th_fcn.linear(inputs, self.We.weight.t())

    def recordLoss(self, y0, xk, it):
        """ MSE between y0,xk + L1 norm of xk"""
        with torch.no_grad():
            self.loss_hist[it] = ((0.5 * (y0 - self.WeT(xk)).pow(2).sum() +
                                  xk.abs().sum().detach().item())) / y0.shape[0]

    def forward(self, y0):
        # Initializations.
        self.loss_hist = torch.full((self.nIter,), torch.nan)
        yk = self.We(y0)
        xprev = torch.zeros(yk.shape).to(self._device)
        t = 1
        # Iterations.
        for it in range(self.nIter):
            residual = self.WeT(yk) - y0
            # ISTA step
            tmp = yk - self.We(residual) / self.max_eig
            xk = th_fcn.softshrink(tmp, lambd=self.sparsity_weight / self.max_eig)
            # FISTA stepsize update:
            tnext = (1 + (1 + 4 * (t ** 2)) ** .5) / 2
            fact = (t - 1) / tnext
            # Use momentum to update code estimate.
            yk = xk + (xk - xprev) * fact
            # Keep track for the next iter stuff
            xprev = xk
            t = tnext
            self.recordLoss(y0, xk, it)
        return yk

    def replace_dictionary(self, weight):
        self.We.weight.data = weight
        self.max_eig = self.est_max_eig_val()

    def normalizeColumns(self):
        with torch.no_grad():
            self.replace_dictionary(
                self.We.weight.data / self.We.weight.data.max(dim=0).values)

    def est_max_eig_val(self, iters=50):
        """
        Use the Power Method method to estimate the
            maximum EigenValue of the layer specified
            by index `moduleIdx`
        """
        with torch.no_grad():
            return torch.lobpcg(torch.nn.functional.linear(self.We.weight.data, self.We.weight.data))[0].item()
