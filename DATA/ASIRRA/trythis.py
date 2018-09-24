#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:12:42 2018

@author: benub
"""

from create_data_dict import defineTrainValSets
from create_data_dict import asirraDataSet


split = 0.8

partition,labels = defineTrainValSets(1)
trainPath = 'train/'
trainSet  = asirraDataSet(partition['train'],labels, trainPath)
valSet    = asirraDataSet(partition['validation'],labels, trainPath)

k = -1
for epoch in range(1):
  for batch,labels in trainSet:
    k+=1
    print(batch.size())
    print(labels.size())
    if k>1:
      break



#mixedMNISTdataset(trainSet,numMixIms)




