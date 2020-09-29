#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 15:15:05 2018

@author: benub
"""
import torch

def extractPatches(batch):
    """
    input: size (batchsize, number-patches, patch-len)
    output: size (batchsize * number-patches, patch-len)
    """
    if batch.dim()<3:
        return batch
    
    bsz = batch.size(0)
    Np = batch.size(1)
    out = torch.Tensor(bsz*Np, batch.size(2))
    n=-1
    for b in range( bsz ):
        for p in range(Np):
            n +=1
            out[n] = batch[b][p]
    return out
