#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FISTA:
    minimizes F (w.r.t x):
    F(x) = .5||y-Ax||_2^2 + alpha*||x||_1


Inputs:
    A: dictionary object
    y: each row is a data vector 
    alpha: scalar L1 parameter
    options: structure ...
Output:
  A structure with the following components:
    codes: each row is an inferred sparse code
    time: timed execution
    fidErr: the matrix valued residual "y-Ax"
    costHist: cost function history
    
@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com/lsalsa
"""
import torch
from torch.autograd import Variable
from UTILS.class_nonlinearity import soft_thresh

def FISTA(y0, A, alpha, maxIter,
          returnCodes = True, returnCost = False, returnResidual = False):
  if not hasattr(A,'maxEig'):
      A.getMaxEigVal()
  shrink = soft_thresh(alpha/A.maxEig,[], A.cuda)
  returnTab = {}

  # INITIALIZE:
  batchSize = y0.size(0)
  yk        = Variable(torch.zeros(y0.size()))
  xprev     = Variable(torch.zeros(y0.size()))
  if A.use_cuda:
      yk    = yk.cuda()
      xprev = xprev.cuda()
  t     = 1
  residual = A.forward(yk) - y0;
  
  # TMP:
  cost = torch.zeros(maxIter)
  
  for it in range(0, maxIter):
  #ista step:
    tmp = yk - A.encode(residual)/A.maxEig 
    xk  = shrink.forward(tmp)
    
  #fista stepsize update:
    tnext = (1 + (1+4*(t**2))**.5)/2 
    fact  = (t-1)/tnext
    yk    = xk + (xk-xprev)*fact
    
  #copies for next iter 
    xprev    = xk;
    t        = tnext
    residual = A.forward(yk) - y0

  # compute any updates desired: (cost hist, code est err, etc.)
   # comphistories(it,yk, params, options, returntab)
    if returnCost:
        fidErr = residual.data[0].norm()**2
        L1err = yk.data[0].abs().sum()
        cost[it] = float(fidErr + alpha * L1err)/batchSize

  if maxIter == 0:
    yk = shrink( A.enc(y0) )

  if returnCodes:
      returnTab["codes"] = yk
  if returnResidual:
      returnTab["residual"] = residual
  if returnCost:
      returnTab["costHist"] = cost.numpy()
  #TODO: time it!
  #if timeTrials:
     # returnTab["time"] = TIME
  return returnTab
  
