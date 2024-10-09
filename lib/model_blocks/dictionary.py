#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Class. TO-DO: subclass multi-dict.

@author: Benjamin Cowen, Feb 24 2018 (updated Feb 2023)
@contact: benjamin.cowen.math@gmail.com
"""
import torch
from typing import Dict
from lib.model_blocks.AlgorithmBlock import AlgorithmBlock
from lib.model_blocks.custom_batchnorm import InvertibleBatchNorm1d


class Dictionary(AlgorithmBlock):
    """
    Class that wraps a matrix (aka fully connected linear layer)
        for use with unrolled algorithm classes.
    Forward converts codes to signals (decoding).
    Points to dataset and encoding algorithm used to generate it.
    """

    def __init__(self,
                 code_dim: int = 0,
                 data_dim: int = 0,
                 device: str = 'cpu',
                 non_blocking: bool = True,
                 **kwargs):
        super(Dictionary, self).__init__()
        self.code_dim = code_dim
        self.data_dim = data_dim
        self.device = device
        self.non_blocking = non_blocking
        self.decoder = torch.nn.Linear(self.code_dim, self.data_dim, bias=False).to(device)

    def forward(self, codes):
        """ convert codes to data """
        return self.decoder(codes)

    def decode(self, codes):
        """ convenience function / same as forward """
        return self.forward(codes)

    def encode(self, data):
        """ Transpose matrix multiplication """
        return torch.nn.functional.linear(data, self.decoder.weight.transpose(-2, 1))

    @property
    def Wd(self):
        """ Get decoder matrix"""
        return self.decoder.weight.data.detach()

    @property
    def We(self):
        """ Get encoder matrix"""
        return self.Wd.transpose(-2, -1)

    def get_encoder(self):
        encoder = torch.nn.Linear(self.data_dim, self.code_dim,
                                  bias=False, device=self.device)
        encoder.weight.data = self.We  # Detach?
        return encoder

    def scale_weights(self, value):
        if value == 'eig':
            value = 1 / self.get_max_eig_val()
        self.decoder.weights.data *= value

    @torch.no_grad()
    def normalize_columns(self):
        # Dictionary is transpose of encoder weights.
        w = self.Wd
        # w.max(dim=0).values
        self.decoder.weight.data = torch.div(w, w.norm(2, dim=0) + 1e-8)

    @torch.no_grad()
    def get_max_eig_val(self, dict_weights=None):
        """
        Estimate the maximum EigenValue of a linear dictionary.
        """
        return torch.lobpcg(torch.mm(self.We, self.Wd))[0].item()


class BatchNormDictionary(Dictionary):
    """
    Same as Dictionary but with a prepended batchnorm.
    """

    def __init__(self, **kwargs):
        super(BatchNormDictionary, self).__init__(**kwargs)

        self.batchnorm = InvertibleBatchNorm1d(self.code_dim)

    def decode(self, codes):
        # Apply BatchNorm and then Linear layer
        return self.batchnorm(self.decoder(codes))

    def encode(self, data):
        # Inverse BatchNorm for encoding
        return self.batchnorm.inverse(torch.nn.functional.linear(
            data, self.decoder.weight.transpose(-2, 1)
        ))

    @torch.no_grad()
    def normalize_columns(self):
        """
        No longer explicitly normalizing.
        """
        pass


class Conv2Dictionary(Dictionary):
    """
    Same as Dictionary but Conv2d instead of Linear.
    """

    def __init__(self,
                 kernel_size: int,
                 conv2d_kwargs: Dict = None,
                 **kwargs):
        super(Conv2Dictionary, self).__init__(**kwargs)

        # Assume square if not specified
        self.kernel_size = kernel_size
        if len(self.kernel_size) == 1:
            self.kernel_size = (self.kernel_size, self.kernel_size)

        # Create Conv2d decoder
        self.conv_config = conv2d_kwargs if conv2d_kwargs else {}
        self.decoder = torch.nn.Conv2d(self.code_dim,
                                       self.data_dim,
                                       self.kernel_size,
                                       bias=False,
                                       **self.conv_config).to(self.device)

    def get_encoder(self):
        encoder = torch.nn.ConvTranspose2d(self.code_dim,
                                           self.data_dim,
                                           self.kernel_size,
                                           bias=False,
                                           **self.conv_config).to(self.device)
        encoder.weight.data = self.We
        return encoder
