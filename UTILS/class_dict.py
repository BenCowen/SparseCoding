#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Class

@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com/lsalsa
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class dictionary(nn.Module):
    """
    Class which defines an mxn linear dictionary.
    """
    def __init__(self, out_size, in_size,
                   datName = 'noname', use_cuda=True, useBias=False):
        super(dictionary, self).__init__()
        self.atoms = nn.Linear(in_size, out_size, bias = useBias)
        if use_cuda:
          self.atoms = self.atoms.cuda()
        self.m        = out_size
        self.n        = in_size
        self.datName  = datName
        self.use_cuda = use_cuda
        
# Set Dictionary Weights:
    def setWeights(self, weights):
        self.atoms.weight.data = weights

# Scale Dictionary Weights:
    def scaleWeights(self, num):
        self.atoms.weight.data *= num

#######################
## BASIC OPERATIONS
#######################
# Forward Pass (decoding)
    def forward(self,inputs):
        return self.atoms(inputs)
# Transpose Pass ([roughly] encoding)
    def encode(self,inputs):
        return  F.linear(self.atoms.weight.t(), inputs).t()
# This worked:
#        return torch.matmul(self.atoms.weight.t(), input.t()).t()
    
# Normalize each column (a.k.a. atom) for the dictionary    
    def normalizeAtoms(self):
      """
      Normalize each column to ||a||=1.
      """
      for a in range(0,self.n):
        atom = self.atoms.weight.data[:,a]
        aNorm = atom.norm()
        atom /= (aNorm+1e-8)
        self.atoms.weight.data[:,a]=atom
            
# Find Maximum Eigenvalue using Power Method
    def getMaxEigVal(self, iters=20):
      """
      Find Maximum Eigenvalue using Power Method
      """
      with torch.no_grad():
        bk = torch.ones(1,self.n)
        if self.use_cuda:
          bk = bk.cuda()
      
        for n in range(0,iters):
          f = bk.abs().max()
          bk = bk/f
          bk = self.encode(self.forward(bk))
        self.maxEig = bk.abs().max().item()

# Return copies of the weights
    def getDecWeights(self):
        return self.atoms.weight.data.clone()

    def getEncWeights(self):
        return self.atoms.weight.data.t().clone()

#######################
## VISUALIZATION
#######################        
# Print the weight values
    def printWeightVals(self):
      print(self.getDecWeights())

    def printAtomImage(self, filename):
        imsize = int(np.sqrt(float(self.m)))
        # Normalize.
        Z = self.getDecWeights()
#        Z = Z = Z - Z.min()
#        Z = Z/(Z.abs().max())
        W = torch.Tensor(self.n, 1, imsize, imsize)
        for a in range(self.n):
            W[a][0] = Z[:,a].clone().resize_(imsize,imsize)
        # Number of atom images per row.
        nr = int(np.sqrt(float(self.n)))
        torchvision.utils.save_image(W, filename, nrow=nr,
                                     normalize=True, pad_value=255)

    
    
    
    
        
        
