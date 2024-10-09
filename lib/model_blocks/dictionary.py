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

    def __init__(self, config, non_blocking=True):
        super(Dictionary, self).__init__()
        code_len = config['code-len']
        data_len = config['data-len']
        self.decoder = torch.nn.Linear(code_len, data_len, bias=False).to(config['device'])
        self._device = config['device']
        self.non_blocking = non_blocking

    def forward(self, codes):
        """ convert codes to data """
        return self.decoder(self.add_to_device(codes))

    def decode(self, codes):
        """ convenience function / same as forward """
        return self.forward(codes)

    def encode(self, data):
        """ Transpose matrix multiplication """
        return torch.nn.functional.linear(data, self.decoder.weight.t())

    def Wd(self):
        """ Get decoder matrix"""
        return self.decoder.weight.data.detach()

    def We(self):
        """ Get encoder matrix"""
        return self.Wd().t()

    def scale_weights(self, value):
        if value == 'eig':
            value = 1 / self.get_max_eig_val()
        self.decoder.weights.data *= value

    @torch.no_grad()
    def normalize_columns(self):
        # Dictionary is transpose of encoder weights.
        w = self.Wd()
        # w.max(dim=0).values
        self.decoder.weight.data = torch.div(w, w.norm(2, dim=0) + 1e-8)

    @torch.no_grad()
    def get_max_eig_val(self, dict_weights=None):
        """
        Estimate the maximum EigenValue of a linear dictionary.
        """
        return torch.lobpcg(torch.mm(self.We(), self.Wd()))[0].item()
