#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 09:56:58 2018

@author: benub
"""
from torch.nn.modules.loss import _Loss

class L1Penalty(_Loss):
    r"""
    Penalizes sparsity of the networks output via L1 norm
    """
    
    def __init__(self, size_average=True, reduce=True):
        super(L1Penalty,self).__init__(size_average)
        self.size_average = size_average
        self.reduce=reduce
        
    def forward(self,input):
        l1 = input.abs().sum()
        if self.size_average:
            l1 = l1/input.size(0)
        return l1