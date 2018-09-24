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
# MISC
from AUX.class_pArray import pArray

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
nppi = 1 #number of patches per image

# OPTIMIZATION PARAMETERS:
maxEpoch   = 5
batchSize = 100
learnRate = 5000

# LOGISTICS:
USE_CUDA = True
z = 'dictAtoms/da'
q = '_'

#######################################################
# (2) Set up data loader and train dictionary.
#######################################################
trainSet, testSet = loadData(dataset, patchSize, batchSize)

savePath = 'dictAtoms/'+dataset+'_demoDict.png'
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('DICTIONARY TRAINING XXXXXXXXXXXXXXXXXXXX')
Dict,LossHist,ErrHist,SpstyHist = trainDictionary(trainSet, testSet, sigLen,
                                                  codeLen, dataset,
                                                  batchSize = batchSize,
                                                  learnRate = learnRate,
                                                  useCUDA = USE_CUDA,
                                                  imSaveName = savePath)


#######################################################
# NOW RUN FISTA, SALSA WITH NEW DICTIONARY AS A TEST:
#######################################################





#benjamin@anna-devbox02:~/LSALSApy$ CUDA_VISIBLE_DEVICES=3 python dict_pTest0.py






