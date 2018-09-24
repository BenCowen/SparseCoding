#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Class

@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com/lsalsa
"""
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision


def computeMaxEigVal(Dict):
    """
    Find Maximum Eigenvalue using Power Method
    """
    if Dict.use_cuda:
        bk = Variable(torch.ones(1,Dict.n).cuda())
    else:
        bk = Variable(torch.ones(1,Dict.n))
    
    f = 1
    iters= 23
    for n in range(0,iters):
        bk = bk/f
        bk = Dict.enc(Dict.dec(bk))
        f = bk.abs().max()
    Dict.maxEig = float(bk.abs().max())

class dictionary(nn.Module):
    """
    Class which defines an mxn linear dictionary.
    """
    def __init__(self, in_size,out_size,datName,use_cuda):
        super(dictionary, self).__init__()
        if use_cuda:
            self.atoms = nn.Linear(in_size, out_size,False).cuda()
        else:
            self.atoms = nn.Linear(in_size, out_size,False)
        self.m = out_size
        self.n = in_size
        self.datName = datName
        self.use_cuda = use_cuda
        
# Set Dictionary Weights:
    def setWeights(self, weights):
        self.atoms.weight.data = weights
# Scale Dictionary Weights:
    def scaleWeights(self, num):
        self.atoms.weight.data = self.atoms.weight.data*num
#######################
## BASIC OPERATIONS
#######################
# Forward Pass (decoding)
    def dec(self,input):
        """
        I think it's better to call it dec but maybe need
         'forward' method for autograd?...
        """
        return self.atoms(input)
    def forward(self,input):
        return self.dec(input)
# Transpose Pass ([roughly] encoding)
    def enc(self,input):
        return torch.matmul(self.atoms.weight.t(), input.t()).t()
    
# Normalize each column (a.k.a. atom) for the dictionary    
    def normalizeAtoms(self):
        for a in range(0,self.n):
            atom = self.atoms.weight.data[:,a]
            atom = atom/atom.norm()
            self.atoms.weight.data[:,a]=atom
            
# Find Maximum Eigenvalue using Power Method
    def getMaxEigVal(self):
        computeMaxEigVal(self)
        
    def getDecWeights(self):
        return self.atoms.weight.data
    def getEncWeights(self):
        return self.atoms.weight.data.t()
#######################
## VISUALIZATION
#######################        
    def printAtomImage(self, filename):
        imsize = int(np.sqrt(float(self.m)))
        Z = self.atoms.weight.data.clone()
        Z = Z = Z - Z.min()
        Z = Z/(Z.abs().max())
        W = torch.Tensor(self.n,1,imsize,imsize)
        for a in range(self.n):
            W[a][0] = Z[:,a].clone().resize_(imsize,imsize)
        nr = int(np.sqrt(float(self.n)))
        torchvision.utils.save_image(W,filename,nrow=nr,normalize=True,pad_value=255)
        
class superDictionary(nn.Module):
    """
    does not have 'forward' method. just helpful tool.
    """
    def __init__(self):
        super(superDictionary, self).__init__()
        self.ms = []
        self.ns = []
        self.dicts = []
        self.m  = 0
        self.n  = 0
        
    def add_dict(self,Dict):
    # LOGISTICS:
        if len(self.dicts) ==0:
            self.datName = Dict.datName
            self.m = Dict.m
            if Dict.use_cuda:
                self.use_cuda = True
        else:
            self.datName = self.datName + '_' + Dict.datName
            if not (Dict.m == self.m):
                print('INCOMPATIBLE DICTIONARIES!')
                error()

        #input size:
        self.ns.append(Dict.n)
        self.n += Dict.n
            
        # APPEND THE WEIGHTS:
        self.dicts.append(Dict)
        
    # PIECEWISE DECODING :
    def pw_dec(self,input):
        """
        decoding and returning each chunk
        """
        outputs = []
        beg = 0
        n_d = 0
        for d,Dict in enumerate(self.dicts):
            beg += n_d
            n_d = self.ns[d]
            outputs.append( Dict.dec(input.narrow(1,beg,n_d)) )
        return outputs
    # STANDARD MATRIX MULTIPLICATIONS:
    # DECODING:
    #    [A1,..,Ap]*[x1; x2;...; xp] = sum_i {Ai*xi}
    def dec(self, input):
        """
        equivalent to summing outputs from pw_dec
        """
        if self.dicts[0].use_cuda:
            output = Variable(torch.zeros(input.size(0), self.m).cuda())
        else:
            output = Variable(torch.zeros(input.size(0), self.m))
        beg = 0
        n_d = 0
        for d,Dict in enumerate(self.dicts):
            beg += n_d
            n_d = self.ns[d]
            output += Dict.dec(input.narrow(1,beg,n_d))
        return output
    # ENCODING:
    #    [A1';...;Ap'] y  = [A1'y ; A2'y ; ... ; Ap'y]
    def enc(self, input):
        """
        A_i^T*y for each i, concatenated.
        """
        output = []
        for d,Dict in enumerate(self.dicts):
            y_d = torch.matmul(Dict.atoms.weight.t(), input.t()).t()
            output.append(y_d)
        return torch.cat(output,1)
    
    def getDecWeights(self):
        A = []
        for d,Dict in enumerate(self.dicts):
            A.append(Dict.atoms.weight.data)
        return torch.cat(A,1)
    def getEncWeights(self):
        A = self.getDecWeights()
        return A.t()
# Find Maximum Eigenvalue using Power Method
    def getMaxEigVal(self):
        computeMaxEigVal(self)
    
    
    
    
    
    
    
    
    
    
    
        
        
