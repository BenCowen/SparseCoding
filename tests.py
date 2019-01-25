#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for unofficial CI. Try to run these before commits.

@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com
"""
# scientific computing
import numpy as np
import torch
from torch.autograd import Variable
import random
# classes
from AUX.class_dict import dictionary
from AUX.FISTA import FISTA

testTol = 1e-5
def testEq(a, b, errorMessage):
  if not (np.abs(a-b) <= testTol):
    print(errorMessage)
    return False
  else:
    return True

############### DICTIONARY TESTS
# Create a trivial dictionary and double check 
# all its methods.
print("Testing dictionary methods ", end = "... ")
PASS = True

## Basic multiplication.
M = 40
N = 50
x = Variable(torch.ones(1,N))
decoder = dictionary(M, N, "testDict", False)
decoder.setWeights( torch.ones(M, N) )

SDx = torch.sum(decoder(x)).item()
PASS = PASS and testEq(SDx, M*N, "\nError: dictionary not multiplying right")

# Maximum eigenvalue and scaling.
decoder.setWeights( torch.eye(M, N) )
decoder.atoms.weight.data[0,0] = np.sqrt(23.0/2)
decoder.scaleWeights(np.sqrt(2))
decoder.getMaxEigVal()
PASS = PASS and testEq( decoder.maxEig, 23, "\nError: maximum eigenvalue computed incorrectly")

# Normalization of atoms/ columns.
decoder.setWeights(torch.rand(M, N))
decoder.normalizeAtoms()
W = decoder.getDecWeights()
for atomID in range(0,N):
  atomNorm = torch.norm( W[:,atomID], 2)
  PASS = PASS and testEq( atomNorm, 1,
              "\nError: Normalization failing")

if PASS:
  print("Passed!")
else:
  print("Failed!")

############### FISTA TESTS
print("Testing FISTA", end = "... ")

# Simple missing-data / denoising problem.
random.seed(23)
torch.manual_seed(23)
N = 256
optCode  = Variable(torch.zeros(1, N))
optCode.data[0, int(.2 * N)] = 1
optCode.data[0, int(.4 * N)] = 0.5
optCode.data[0, int(.5 * N)] = -2
optCode.data[0, int(.6 * N)] = -1.75
optCode.data[0, int(.8 * N)] = 0.25

decoder  = dictionary( N, N, "decimation", False)
# Create noisy observation
noisyDat = decoder(optCode) + Variable(0.21*(torch.rand(1,N) - 0.5))
alpha = 0.15
maxIter = 200
options = {"returnCodes" : True, "returnCost" : True, "returnResidual" : True}
fistaOut = FISTA(noisyDat, decoder, alpha, maxIter, *options)

code = fistaOut["codes"]
loss = fistaOut["costHist"]
residual   = fistaOut["residual"]

# Sort of a golden-value test isn't great but...
codeErr = ((code - optCode).norm(2)**2).sum().item()
PASS = PASS and testEq(codeErr, 0.8006578, "Error: FISTA no longer computing code correctly")

## Test convergence:
shouldBeTiny = np.diff(loss)[-10:-1].sum()
PASS = PASS and testEq(shouldBeTiny, 0, "Error: FISTA no longer converging on toy problem")

if PASS:
  print("Passed!")
else:
  print("Failed!")























