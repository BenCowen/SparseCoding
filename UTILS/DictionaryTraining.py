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
# ALGORITHMS
from AUX.FISTA import FISTA
# PLOTTING
import matplotlib.pyplot as plt
from AUX.writeProgressFigs import printProgressFigs
# DATA
from AUX.fcn_patchExtract import extractPatches
# MISC
from AUX.class_nonlinearity import soft_thresh
import gc
  

def trainDictionary(train_loader, test_loader, sigLen, codeLen, datName,
                    maxEpoch = 2,
                    learnRate = 1,
		                lrdFreq   = 30,
                    learnRateDecay = 1,
                    fistaIters = 50,
                    useL1Loss = False,
                    l1w = 0.5,
                    useCUDA = False,
		                imSave = True,
                    imSavePath = "./",
                    extension = ".png",
                    printFreq = 10,
                    saveFreq  = 100,
                    **kwargs):
    """
    Inputs:
      maxEpoch : number of training epochs (1 epoch = run thru all data)
      learnRate : weight multiplied onto gradient update
      lrdFreq : number of batches between learnRate scaling
      learnRateDecay: scales learnRate every lrdFreq batches, i.e.
        epoch_LR = learnRate*(learnRateDecay**(epoch-1))
      fistaIters : number of iterations to run FISTA when generating epoch codes
      useL1Loss : set to True to penalize sparsity of network output (BUGGY)
      l1w : the L1-norm weight, balances data fidelity with sparsity
      useCUDA : set to true for GPU acceleration
      imSave : boolean determines whether to save images
      dataset : name of the dataset (used in saved image files)
      imSavePath : directory in which multiple images will be saved
      extension : type of image to save (e.g. ".png", ".pdf")
      printFreq : the number of batches between print statements during training
      saveFreq : the number of batches between image saves during training
      kwargs: optional arguments
	      atomImName : name for dictionary atom image other than "dictAtoms"
	      dictInitWeights: initial weights (instead of normal distribution)
    Outputs:
      Dict: the trained dictionary / decoder
      lossHist : loss function history (per batch)
      errHist : reconstruction error history (per batch)
      spstyHist : sparsity history (per batch). i.e. the percent zeros
          achieved during encoding with the dictionary. Encoding is
          performed using FISTA.
    """

    
    # MISC SETUP
    # FISTA:
    fistaOptions = {"returnCodes"  : True,
                    "returnCost"   : False,
                    "returnFidErr" : False}
    # Saving dictionary atom images:
    if "atomImName" in kwargs:
        dictAtoms = kwargs["atomImName"]
    else:
        dictAtoms = datName + "dictAtoms"
    dictAtomImgName = imSavePath + dictAtoms + extension

    # Recordbooks:
    lossHist  = []
    errHist   = []
    spstyHist = []
    
    # INITIALIZE DICTIONARY
    Dict = dictionary(sigLen, codeLen, datName, useCUDA)
    if "dictInitWeights" in kwargs:
        Dict.setWeights(dictInitWeights)
    Dict.normalizeAtoms()
    Dict.zero_grad()
    
    # Loss Function:  .5 ||y-Ax||_2^2 + alpha||x||_1
    mseLoss = nn.MSELoss()
    mseLoss.size_average=True
    if useL1Loss:
      l1Loss = Variable( torch.FloatTensor(1), requires_grad=True)
    else:
      l1Loss = 0

    # Optimizer
    OPT = torch.optim.SGD(Dict.parameters(), lr=learnRate)
    # For learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(OPT, step_size = lrdFreq,
                                                gamma = learnRateDecay)
###########################
### DICTIONARY LEARNING ###
###########################
    for it in range(maxEpoch):
        epochLoss      = 0
        epoch_sparsity  = 0
        epoch_rec_error = 0
    
#================================================
    # TRAINING
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
          X = fistaOut["codes"]
          gc.collect()
         
      ## FORWARD PASS
          Y_est     = Dict.forward(X)   #try decoding the optimal codes
          
          # loss
          reconErr   = mseLoss(Y_est,Y)
          if useL1Loss:
            l1Loss  = Y_est.norm(1) / X.size(0)
         
      ## BACKWARD PASS
          batchLoss = reconErr + l1w * l1Loss
          batchLoss.backward()
          OPT.step()
          scheduler.step()
          Dict.zero_grad()
          Dict.normalizeAtoms()
          del Dict.maxEig
         
      ## Housekeeping
          sampleLoss      = batchLoss.data[0]
          epochLoss      +=   sampleLoss
          lossHist.append( epochLoss/numBatch )
          
          sample_rec_error = reconErr.data[0]
          epoch_rec_error += sample_rec_error
          errHist.append( epoch_rec_error/numBatch )

          sample_sparsity = ((X.data==0).sum())/X.numel()
          epoch_sparsity  +=  sample_sparsity
          spstyHist.append( epoch_sparsity/ numBatch )

      ## SANITY CHECK:
          # If the codes are practically all-zero, stop fast.
          # TODO: something smarter to do here? Lower L1 weight?
          if np.abs(sample_sparsity - 1.0) < 0.001 :
            print("CODES NEARLY ALL ZERO. SKIP TO NEXT EPOCH.")
            break
 
     ## Print stuff.
     # You may wish to print some figures here too. See bottom of page.
          if batch_idx % printFreq == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                it, batch_idx * len(batch), len(train_loader.dataset),
                100* batch_idx / len(train_loader)))
              print('Loss: {:.6f} \tRecon Err: {:.6f} \tSparsity: {:.6f} '.format(
                     sampleLoss,sample_rec_error,sample_sparsity))
          
          if batch_idx % saveFreq == 0:
              Dict.printAtomImage(dictAtomImgName)
              printProgressFigs(imSavePath, extension, lossHist, errHist, spstyHist)

      ## end "TRAINING" batch-loop
#================================================
    
    ## need one for training, one for testing
        epoch_average_loss = epochLoss/numBatch
        epoch_avg_recErr   = epoch_rec_error/numBatch
        epoch_avg_sparsity = epoch_sparsity/numBatch
    
#        lossHist[it]  = epoch_average_loss
#        errHist[it]   = epoch_avg_recErr
#        spstyHist[it] = epoch_avg_sparsity
        
        print('- - - - - - - - - - - - - - - - - - - - -')
        print('EPOCH ', it + 1,'/',maxEpoch, " STATS")
        print('LOSS = ', epoch_average_loss)
        print('RECON ERR = ',epoch_avg_recErr)
        print('SPARSITY = ',epoch_avg_sparsity)
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
  ## end "EPOCH" loop

    # Convert recordbooks to numpy arrays:
    lossHist  = np.asarray(lossHist)
    errHist   = np.asarray(errHist)
    spstyHist = np.asarray(spstyHist)

    # Save the dictionary/decoder:
#    torch.save(save_dir..'decoder_'..datName..'psz'..pSz..'op'..outPlane..'.t7', decoder) 
    Dict.printAtomImage(dictAtomImgName)
    printProgressFigs(imSavePath, extension, lossHist, errHist, spstyHist)
    return Dict,lossHist,errHist,spstyHist

##########################################
## Plotting examples.
#-------------------
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
##########################################
