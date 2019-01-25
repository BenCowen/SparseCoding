#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Training Demo
Executable version of DEMO.py script.

@author: Benjamin Cowen, 23 Jan 2019
@contact: ben.cowen@nyu.edu, bencowen.com
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
# PARAMETERS
from getParams import fetchVizParams
# ARGUMENTS
import argparse
from AUX.utils import ddict

#######################################################
# (0.5) Parse user inputs.
#######################################################
parser = argparse.ArgumentParser(description='Sparse-Dictionary-Learning')

########### Model and Data arguments
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='"MNIST", "CIFAR10", ...')
parser.add_argument('--valid-size',type=int, default=-1, metavar='N',
                    help='Number of samples removed for validation. If N<=0, test set is used.')
parser.add_argument('--patch-size',type=int, default=10, metavar='p',
                    help='Breaks each image into pxp subimages.')
parser.add_argument('--overComplete', type=float, default=1, metavar='d',
                    help='Defines dictionary as d-overcomplete (number of times overcomplete)')
parser.add_argument('--L1-weight', type=float, default=0.2, metavar='lambda',
                    help='Non-negative scalar weight for L1 (sparsity) term.')
# Encoding Parameters:
parser.add_argument('--encode-alg', type=str, default='fista',
                    help='Encoding algorithm ("fista" or "salsa").')
parser.add_argument('--fista-iters', type=int, default=200, metavar='M',
                    help='During encoding step, perform M steps of encoding algorithm.')

########### Optimization arguments
parser.add_argument('--max-epochs', type=int, default=200,
                    help='Number of training epochs.')
parser.add_argument('--batch-size', type=int, default=10,
                    help='Number of training patches per batch.')
parser.add_argument('--opt-method', type=str, default='sgd',
                    help='Learning algorithm ("sgd" or "adam").')
parser.add_argument('--learn-rate', type=float, defualt=2e2,
                    help='Initial learning rate.')
# SGD Parameters:
parser.add_argument('--learn-rate-decay', type=float, default=0.999,
                    help='Epoch-wise learning rate decay.')
parser.add_argument('--momentum', type=float, default=0,
                    help='If nonzero, applies Nesterov Momentum.')

########### Logistics
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--save-filename', type=str, default='./no-name',
                    help='Path to directory where everything gets saved.')
parser.add_argument('--use-HPO-params', action='store_true', default=False,
                    help='If true, uses the hyperparameters found by grid search (stored in "paramSearchResults").')
parser.add_argument('--seed',type=int,
                    help='RNG seed.')
parser.add_argument('--data-seed',type=int,
                    help='Special seed for validation set generation.')
parser.add_argument('--save-trained-model', action='store_true', default=False,
                    help='If true, saves trained model with ddict().')
parser.add_argument('--visualize-dict-atoms', action='store_true', default=False,
                    help='If true, saves image of dictionary atoms during training.')


#######################################################
# (1) Unpack user inputs.
#######################################################
args = parser.parse_args()
# Cost function parameters.
dataset    = args.dataset
patchSize  = args.patch_size
datName    = dataset + str(patchSize)
sigLen     = patchSize**2
codeLen    = sigLen * args.overComplete
L1_weight  = args.L1-weight

# Optimization parameters.
maxEpoch  = args.max_epochs
batchSize = args.batch_size

learnRate = args.learn_rate
learnRateDecay = args.learn_rate_decay
fistaIters = 200

# First set up the data using the data-seed.

# Now set up CUDA and reset RNG's.

# If requested, use HPO results.

# **** reproduce results using this function. to play, comment it out. *****
# batchSize, L1_weight, learnRate, learnRateDecay = fetchVizParams(datName)
# ************

# Main
if __name__ == "__main__":
    # Save arguments.
    print('********************')
    print('Saving shelf to:')
    print(args.save_filename)
    print('********************')
    SH = ddict(args=args.__dict__)
    if args.save_filename:
        SH._save(args.save_filename, date=True)

    # Store training and test losses, sparsities, and reconstruction errors
    #     after each training epoch.
    SH.tr_perf = ddict(loss=[], sparsity=[], reconErr=[])
    SH.te_perf = ddict(loss=[], sparsity=[], reconErr=[])
    
    #######################################################
    # (2) Set up data loader, model, and encoder.
    #######################################################
    trainSet, testSet = loadData(dataset, patchSize, batchSize)
    model = 
    encoder = 
    #######################################################
    # (3) Train model.
    #######################################################
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print('DICTIONARY TRAINING XXXXXXXXXXXXXXXXXXXX')

    encoderOptions =  {"returnCodes"  : True,
                       "returnCost"   : False,
                       "returnFidErr" : False}


    for epoch in range(1, args.max_epochs+1):

        with torch.no_grad:
            dictionary.encode(data, encodeAlg, encodeArgs)
    
# TODO: actually look up how to save "this" file etc.
dictSavePath = "trainedModels/" + dataset + str(patchSize) + "/"
# save model
# save convergence / training progress plots and dict atoms
# save the file used to create it all (i.e. this one)

#benjamin@anna-devbox02:~/LSALSApy$ CUDA_VISIBLE_DEVICES=3 python dict_pTest0.py






