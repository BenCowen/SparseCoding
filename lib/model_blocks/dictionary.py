#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Class. TO-DO: subclass multi-dict.

@author: Benjamin Cowen, Feb 24 2018 (updated Feb 2023)
@contact: benjamin.cowen.math@gmail.com
"""
import torch
from lib.model_blocks.AlgorithmBlock import AlgorithmBlock


class Dictionary(AlgorithmBlock):
    """
    Class that wraps a matrix (aka fully connected linear layer)
        for use with unrolled algorithm classes.
    Forward converts codes to signals (decoding).
    Points to dataset and encoding algorithm used to generate it.
    """

    def __init__(self,
                 code_dim: int,
                 data_dim: int,
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
        return torch.nn.functional.linear(data, self.decoder.weight.t())

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


class Conv2Dictionary(Dictionary):
    """
    Same as Dictionary but efficiently implements 1 convolutional layer.
    """

    def __init__(self, **kwargs):
        super(Conv2Dictionary, self).__init__(config)
        self.kernel_size = config['kernel-size']
        self.conv_config = config['conv-config']
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
