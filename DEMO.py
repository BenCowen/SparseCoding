#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Training Demo
After (0) importing modules we (1) designate training and data
  model parameters. Then we (2) construct dataloader and
  execute dictionary training. Finally we (3) visualize
  the training procedure and resulting dictionary.

@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com
"""

#######################################################
# (0) Import modules.
#######################################################
import numpy as np
# DATA
from DATA.loadImDat import loadData
# TRAINING
from AUX.DictionaryTraining import trainDictionary
# PLOTTING
import matplotlib.pyplot as plt

#######################################################
# (1) Define experiment.
# TODO: put this in a .json configuration file.
#######################################################
# Cost function parameters.
dataset    = "MNIST"
patchSize  = 32
datName    = dataset + str(patchSize)
sigLen     = patchSize**2
codeLen    = sigLen              # "1x overcomplete"
L1_weight  = 0.2

# Optimization parameters.
maxEpoch  = 10
batchSize = 10
learnRate = 2e2
learnRateDecay = 0.999
fistaIters = 200

# Logistics.
USE_CUDA = True
savePath = 'results/'

#######################################################
# (2) Set up data loader and train dictionary.
#######################################################
trainSet, testSet = loadData(dataset, patchSize, batchSize)

atomImName = dataset + str(patchSize) + '_demoDict'

print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('DICTIONARY TRAINING XXXXXXXXXXXXXXXXXXXX')
Dict,lossHist,errHist,spstyHist = trainDictionary(trainSet, testSet, sigLen,
                                                  codeLen, datName,
                                                  maxEpoch = maxEpoch,
                                                  fistaIters =fistaIters,
                                                  l1w = L1_weight,
                                                  batchSize = batchSize,
                                                  learnRate = learnRate,
                                                  learnRateDecay = learnRateDecay,
                                                  useCUDA = USE_CUDA,
                                                  imSavePath = savePath,
                                                  daSaveName = atomImName)
print("done!")


############################################################
# Now run FISTA with the new dictionary to test convergence.
############################################################





#benjamin@anna-devbox02:~/LSALSApy$ CUDA_VISIBLE_DEVICES=3 python dict_pTest0.py






