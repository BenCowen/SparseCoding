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
patchSize = 32
sigLen     = patchSize**2
codeLen    = sigLen              # "1x overcomplete"
L1_weight  = 0.5

# OPTIMIZATION PARAMETERS:
maxEpoch   = 5
batchSize = 100
learnRate = 5000

# LOGISTICS:
USE_CUDA = True
savePath = 'results/'

#######################################################
# (2) Set up data loader and train dictionary.
#######################################################
trainSet, testSet = loadData(dataset, patchSize, batchSize)

atomImName = savePath + dataset + '_demoDict.png'

print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('DICTIONARY TRAINING XXXXXXXXXXXXXXXXXXXX')
Dict,LossHist,ErrHist,SpstyHist = trainDictionary(trainSet, testSet, sigLen,
                                                  codeLen, dataset,
                                                  l1w = L1_weight,
                                                  batchSize = batchSize,
                                                  learnRate = learnRate,
                                                  useCUDA = USE_CUDA,
                                                  imSaveName = atomImName)


#######################################################
# Visualize loss, reconstruction error, and sparsity
#  histories.
#######################################################

############################################################
# Now run FISTA with the new dictionary to test convergence.
############################################################





#benjamin@anna-devbox02:~/LSALSApy$ CUDA_VISIBLE_DEVICES=3 python dict_pTest0.py






