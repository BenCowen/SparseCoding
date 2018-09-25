#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brute-force parameter search for dictionary learning.
See Demo.py for a simple example.
This runs something like Demo.py for a large number of
parameters.

@author: Benjamin Cowen, September 25, 2018
@contact: bc1947@nyu.edu, bencowen.com
"""
#######################################################
# (0) Import modules.
#######################################################
# DATA
from loadImDat import loadData
# TRAINING
from DictionaryTraining import trainDictionary
# PLOTTING
import matplotlib.pyplot as plt

#######################################################
# (1) Define experiment.
# TODO: put this in a .json configuration file.
#######################################################
# Cost function parameters.
dataset    = "MNIST"
patchSize  = 32
sigLen     = patchSize**2
codeLen    = sigLen              # "1x overcomplete"
L1_weightList  = [.1, 0.5, 1]

# OPTIMIZATION PARAMETERS:
maxEpoch   = 5
batchSizeList = [100, 1000]
learnRateList = [5e-1, 1, 5, 5e2, 5e3]
LRDecayList = [1, 0.999, 0.9]

# LOGISTICS:
USE_CUDA = True
savePath = 'parameterSearch/'

#######################################################
# (2) Set up data loader and train dictionary.
#######################################################

print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('DICTIONARY TRAINING XXXXXXXXXXXXXXXXXXXX')

count = 0;
total = len(batchSizeList) * len(L1_weightList) * len(learnRateList) * len(LRDecayList)

for j,bsz in  enumerate(batchSizeList):
  trainSet, testSet = loadData(dataset, patchSize, bsz)
  for i,l1w in enumerate(L1_weightList):
    for k,lr in enumerate(learnRateList):
      for l, lrd in enumerate(LRDecayList):
        count += 1
        paramID = str(i) + str(j) + str(k) + str(l)
        print("Training " + dataset + " dictionary with parameter set " + paramID)
        print("(" + str(count) + "/" + str(total) + ")")
        atomImName = savePath + dataset + '_' + paramID + '.png'
        Dict,LossHist,ErrHist,SpstyHist = trainDictionary(trainSet, testSet, sigLen,
                                                  codeLen, dataset,
                                                  maxEpoch = maxEpoch,
                                                  useCUDA = USE_CUDA,
                                                  fistaIters = 100,
                                                  printFreq = 100,
                                                  saveFreq = 100,
                                                  # looped parameters:
                                                  l1w = l1w,
                                                  batchSize = bsz,
                                                  learnRate = lr,
                                                  learnRateDecay = lrd,
                                                  imSaveName = atomImName)
















