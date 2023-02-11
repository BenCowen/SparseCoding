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

    def __init__(self, data_len, code_len, n_iters,
                 sparsity_weight, init_dict=None, device='cpu'):
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
        self.L1w = sparsity_weight
        self.lossHist = torch.full((n_iters,), torch.nan)

        # Dictionary Initialization
        self._device = device
        self.We = torch.nn.Linear(data_len, code_len, bias=False, device=self._device)

        if init_dict is not None:
            self.replaceDictionary(init_dict)
        else:
            self.normalizeColumns()
            self.estMaxEigVal()

    def WeT(self, inputs):
        """ Transpose propagation through We"""
        return th_fcn.linear(inputs, self.We.weight.t())

    def recordLoss(self, y0, xk, it):
        """ MSE between y0,xk + L1 norm of xk"""
        with torch.no_grad():
            self.lossHist[it] = ((0.5 * (y0 - self.WeT(xk)).pow(2).sum() +
                                  xk.abs().sum().detach().item())) / y0.shape[0]

    def forward(self, y0):
        # Initializations.
        self.lossHist = torch.full((self.nIter,), torch.nan)
        yk = self.We(y0)
        xprev = torch.zeros(yk.shape).to(self._device)
        t = 1
        # Iterations.
        for it in range(self.nIter):
            residual = self.WeT(yk) - y0
            # ISTA step
            tmp = yk - self.We(residual) / self.maxEig
            xk = th_fcn.softshrink(tmp, lambd=self.L1w / self.maxEig)
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

    def replaceDictionary(self, weight):
        self.We.weight.data = weight
        self.estMaxEigVal()

    def normalizeColumns(self):
        with torch.no_grad():
            self.replaceDictionary(
                self.We.weight.data / self.We.weight.data.max(dim=0).values)

    def estMaxEigVal(self, iters=50):
        """
        Use the Power Method method to estimate the
            maximum EigenValue of the layer specified
            by index `moduleIdx`
        """
        with torch.no_grad():
            bk = torch.ones(1, self.code_len).to(self._device)

            for n in range(0, iters):
                f = bk.abs().max()
                bk = bk / f
                bk = self.We(self.WeT(bk))
        self.maxEig = bk.abs().max().item()
