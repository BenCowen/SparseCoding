#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classic algorithm implementations

@author: Benjamin Cowen, March 6 2018
@contact: bc1947@nyu.edu, bencowen.com/lsalsa
"""
import torch
from AUX.class_nonlinearity import soft_thresh
from AUX.class_nonlinearity import hard_thresh
from AUX.class_nonlinearity import makeSALSAthreshes

def getWeightedL1norms(l1w,x,ns=[]):
    if type(l1w)==float:
        return l1w*x.norm(1)
    elif type(l1w)==list:
       out = 0.0
       beg = 0
       n_d = 0
       for t,alpha in enumerate(l1w):
           beg += n_d
           n_d = ns[t]
           out += alpha*(x.narrow(1,beg,n_d)).norm(1)
       return out
    
class  optCode():
    def __init__(self, codes):
        """
        codes: tensor of size (batch x codeLen)
        """
        self.codes = codes
        self.N     = codes.size(0)
    def compMSEs(self, estCodes):
        return (estCodes - self.codes).pow(2).sum(1)
    def compTotalMSE(self, estCodes):
        return (estCodes - self.codes).pow(2).sum()/self.N
        
class SALSA():
    def __init__(self, A, mu, options):
        super(SALSA, self).__init__()
        
        self.m = A.m
        self.n = A.n
        self.A = A
        self.mu = mu
        
        
##########################
## S matrix: [ mu*I + A^T*A]^{-1}
##########################
        # TAKE OUT OF CUDA FOR INVERSE!!
        AA = torch.mm(A.getEncWeights(),A.getDecWeights()).cpu()
        S_weights = (mu*torch.eye(A.n) + AA).inverse()
        self.S = torch.nn.Linear(A.n, A.n, False)
        self.S.weight.data = S_weights
        if A.use_cuda:
            self.S=self.S.cuda()
        
##########################
## Per-iteration stuff
##########################
        # cost fcn hist
        # compHistories thing
        # outputs
    #def iterUpdate(self, it,x,options):
    def cstfcn(self,y,x,l1w):
        return (y-self.A.dec(x)).norm(2).data[0]**2 + getWeightedL1norms(l1w,x,self.A.ns).data[0]
##########################
## Go-time!
##########################
    def inference(self, y, l1w, maxIter, options):
        
        cost = []
        shrink,finThrsh = makeSALSAthreshes(l1w,self.mu,self.A.ns,self.A.use_cuda)

        AHy = self.A.enc(y)
        x = AHy.clone()
        d = 0*x.clone()
        for it in range(maxIter):
#          print(x.norm(2).data[0])
      # 'u' update (nonlinearity: shrinkage/soft-thresholding)
          u  = shrink( x+d )
      # *NOT* USING THE MATRIX INVERSE LEMMA:
          x  = self.S( AHy + (u-d)*self.mu ) 
      # 'd' update (Bregman Penalty)
          d += x - u 
      # compute any updates desired:
          cost.append( self.cstfcn(y,x,l1w) )
     #     self.iterUpdate(it, x, options)
      
#        print(shrink(x).norm(2).data[0])
        return finThrsh(x),cost#self.outputs(x,options)

        
        
        
class ISTA():
    """
    ISTA should really just be a method of dictionary.
    But let's make each algorithm a class for consistency.
    """
    def __init__(self, A):
        super(ISTA, self).__init__()
        
        self.m = A.m
        self.n = A.n
        if not hasattr(A,'maxEig'):
            A.getMaxEigVal()
        self.A = A
        
    def inference(self, y, l1w, maxIter, options):
        
        cost = []
        L = self.A.maxEig
        shrink   = soft_thresh(l1w/L,[],self.A.use_cuda)
        finThrsh = hard_thresh(l1w/L,[],self.A.use_cuda)
       
        AHy_L = self.A.enc(y)/L
        x = AHy_L.clone()*0
        for it in range(maxIter):
            z = x - self.A.enc(self.A.dec(x))/L + AHy_L
            x = shrink(z)
            cost.append( ((y-self.A.dec(x)).norm(2)**2 + l1w*x.norm(1)).data[0] )
     #     self.iterUpdate(it, x, options)
      
        return finThrsh(x),cost
    
        
        
        
        
        
        