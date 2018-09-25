#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Learning

Minimizes F w.r.t. A, using stochastic gradient descent:
    F(A) = 1/P * sum_{p=1}^P   1/2*|| y(p)-A*x(p)||_2^2 + alpha*||x(p)||_1
    where x(p) = FISTA(y(p), A[previous iter],alpha,...)
        is the sparse code of y(p) w.r.t. to the previous dictionary guess
        
@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com/lsalsa
"""
# NUMPY
import numpy as np
# TORCH
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# DICTIONARY
from AUX.class_dict import dictionary
# L1 LOSS/PENALTY
from AUX.class_l1penalty import L1Penalty
# ALGORITHMS
from FISTA import FISTA
# PLOTTING
import matplotlib.pyplot as plt
# DATA
from fcn_patchExtract import extractPatches
# MISC
from AUX.class_nonlinearity import soft_thresh
import gc
  

def trainDictionary(train_loader, test_loader, sigLen, codeLen, datName,
                    maxEpoch = 2,
                    learnRate = 1,
                    learnRateDecay = 1,
                    fistaIters = 50,
                    useL1Loss = True,
                    l1w = 0.5,
                    useCUDA = False,
                    imSaveName = "dictAtoms.png",
                    printFreq = 10,
                    saveFreq  = 100,
                    **kwargs):
    
    # MISC SETUP
    fistaOptions = {"returnCodes"  : True,
                    "returnCost"   : False,
                    "returnFidErr" : True}
    
    # Recordbooks:
    LossHist  = np.zeros(maxEpoch)
    ErrHist   = np.zeros(maxEpoch)
    SpstyHist = np.zeros(maxEpoch)
    
    # INITIALIZE DICTIONARY
    Dict = dictionary(codeLen, sigLen, datName, useCUDA)
    if "dictInitWeights" in kwargs:
        Dict.setWeights(dictInitWeights)
    Dict.normalizeAtoms()
    Dict.zero_grad()
    
    # Loss Function:  .5 ||y-Ax||_2^2 + alpha||x||_1
    if useL1Loss:
        loss1 = nn.MSELoss()
        loss1.size_average=True
        loss2 = L1Penalty()
        loss2.size_average=True
    else:
        loss = nn.MSELoss()
        loss.size_average=True
        
    # Optimizer
    SGD = torch.optim.SGD(Dict.parameters(), lr=learnRate)
  
###########################
### DICTIONARY LEARNING ###
###########################
    for it in range(maxEpoch):
        epoch_LR        = learnRate*(learnRateDecay**(it-1))
        epoch_loss      = 0
        epoch_sparsity  = 0
        epoch_rec_error = 0
    
#================================================
    ## TRAINING
        numBatch = 0
        for batch_idx, (batch, labels) in enumerate(train_loader):
          gc.collect()
          numBatch += 1
        
          if useCUDA:
              Y = Variable(extractPatches(batch).cuda())
          else:
              Y = Variable(extractPatches(batch))
              
      ## CODE INFERENCE
          fistaOut = FISTA(Y, Dict, l1w, fistaIters, fistaOptions)

          X        = fistaOut["codes"]
          residual = fistaOut["fidErr"]
          gc.collect()
         
     ## FORWARD PASS
          Y_est     = Dict.forward(X)   #try decoding the optimal codes
          
          if useL1Loss:
              rec_err   = loss1(Y_est,Y)
              spsty_er  = loss2(Y_est)
          else:
              rec_err   = loss(Y_est,Y)
              spsty_er  = 0
         
     ## BACKWARD PASS
          (rec_err+l1w*spsty_er).backward()
          SGD.step()
          Dict.zero_grad()
          Dict.normalizeAtoms()
          del Dict.maxEig
         
     ## Housekeeping
          sample_loss      = (residual.pow(2).sum() + l1w*X.norm(1)) / X.size(1)
          epoch_loss      +=   sample_loss.data[0]
          
          sample_rec_error = residual.pow(2).sum(1).sqrt().mean()
          epoch_rec_error += sample_rec_error.data[0]
          
          sample_sparsity = ((X.data==0).sum())/X.numel()
          epoch_sparsity  +=  sample_sparsity
          
          if batch_idx % printFreq == 0:
    # PLOT FISTA COST FUNCTION HISTORY
#              plt.plot(fistaOut.costHist)
#              plt.xlabel('Iterations')
#              plt.ylabel('Cost (averaged over samples)')
#              plt.title('Cost Function History')
#              plt.show()
    # PLOT CODE EXAMPLE
#              plt.plot(np.array(X.data[5]))
#              plt.title('Code visualization')
#              plt.show()
              print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                it, batch_idx * len(batch), len(train_loader.dataset),
                100* batch_idx / len(train_loader)))
              tmp0 = (rec_err+l1w*spsty_er).data[0]
              tmp1 = epoch_rec_error/(batch_idx+1)
              tmp2 = epoch_sparsity/(batch_idx+1)
              print('Loss: {:.6f} \tRecon Err: {:.6f} \tSparsity: {:.6f} '.format(
                     tmp0,tmp1,tmp2))
          
          if batch_idx % saveFreq == 0:
              Dict.printAtomImage(imSaveName)
      ## end "TRAINING" batch-loop
#================================================
    
    ## need one for training, one for testing
        epoch_average_loss = epoch_loss/numBatch
        epoch_avg_recErr   = epoch_rec_error/numBatch
        epoch_avg_sparsity = epoch_sparsity/numBatch
    
        LossHist[it]  = epoch_average_loss
        ErrHist[it]   = epoch_avg_recErr
        SpstyHist[it] = epoch_avg_sparsity
        
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('EPOCH ', it + 1,'/',maxEpoch, " STATS")
        print('LOSS = ', epoch_average_loss)
        print('RECON ERR = ',epoch_avg_recErr)
        print('SPARSITY = ',epoch_avg_sparsity)
  ## end "EPOCH" loop
    
    
#    torch.save(save_dir..'decoder_'..datName..'psz'..pSz..'op'..outPlane..'.t7', decoder) 
    Dict.printAtomImage(imSaveName)
    return Dict,LossHist,ErrHist,SpstyHist
    
    
    
