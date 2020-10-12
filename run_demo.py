#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary Training Demo
Executable version of DEMO.py script.

@author: Benjamin Cowen, 23 Jan 2019
@contact: benmoo23@gmail.com, bencowen.com
"""

#######################################################
# (0) Import modules.
#######################################################
# Scientific Computing
import numpy as np
import torch
import torch.nn.functional as F
import random as rand # needed for validation shuffle
# DATA
from loadImDat import loadData
# TRAINING
from UTILS.class_dict import dictionary
from UTILS.class_encoder import ENCODER, compute_sparsity
import gc
# PLOTTING
import matplotlib.pyplot as plt
# PARAMETERS
from getParams import fetchVizParams
# ARGUMENTS
import argparse
from UTILS.utils import ddict

#######################################################
# (0.5) Parse user inputs.
#######################################################
parser = argparse.ArgumentParser(description='Sparse-Dictionary-Learning')

########### Model and Data arguments
parser.add_argument('--dataset', type=str, default='mnist',
                    help='"mnist", "fashion_mnist", "cifar10", "ASIRRA", ...')
parser.add_argument('--valid-size',type=int, default=-1, metavar='N',
                    help='Number of samples removed for validation. If N<=0, test set is used.')
parser.add_argument('--patch-size',type=int, default=32, metavar='p',
                    help='Breaks each image into pxp subimages.')
parser.add_argument('--overComplete', type=float, default=1, metavar='d',
                    help='Defines dictionary as d-overcomplete (number of times overcomplete)')
parser.add_argument('--L1-weight', type=float, default=0.2, metavar='lambda',
                    help='Non-negative scalar weight for L1 (sparsity) term.')
# Encoding Parameters:
parser.add_argument('--encode-alg', type=str, default='fista',
                    help='Encoding algorithm ("ista","fista" or "salsa").')
parser.add_argument('--encode-iters', type=int, default=200, metavar='M',
                    help='During encoding step, perform M steps of encoding algorithm.')
parser.add_argument('--mu', type=float, default=None, metavar='M',
                    help='"mu" parameter for SALSA-based encoding.')

########### Optimization arguments
parser.add_argument('--max-epochs', type=int, default=200,
                    help='Number of training epochs.')
parser.add_argument('--batch-size', type=int, default=10,
                    help='Number of training patches per batch.')
parser.add_argument('--opt-method', type=str, default='adam',
                    help='Learning algorithm ("sgd" or "adam").')
parser.add_argument('--learn-rate', type=float, default=2e2,
                    help='Initial learning rate.')
# SGD Parameters:
parser.add_argument('--learn-rate-decay', type=float, default=0.999,
                    help='Epoch-wise learning rate decay.')
parser.add_argument('--momentum', type=float, default=0,
                    help='If nonzero, applies Nesterov Momentum.')

########### Logistics
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--save-filename', type=str, default='results/tmp/no-name',
                    help='Path to directory where everything gets saved.')
parser.add_argument('--use-HPO-params', action='store_true', default=False,
                    help='If true, uses the hyperparameters found by grid search (stored in "paramSearchResults").')
parser.add_argument('--seed',type=int, default=23,
                    help='RNG seed.')
parser.add_argument('--data-seed',type=int, default=23,
                    help='Special seed for validation set generation.')
parser.add_argument('--save-trained-model', action='store_true', default=False,
                    help='If true, saves trained model with ddict().')
parser.add_argument('--visualize-dict-atoms', action='store_true', default=False,
                    help='If true, saves image of dictionary atoms during training.')
parser.add_argument('--print-frequency', type=int, default=4,
                    help='Number of print statements per epoch.')

#######################################################
# (1) Unpack user inputs.
#######################################################
args = parser.parse_args()

# Cost function parameters.
dataset    = args.dataset
patch_size = args.patch_size
datName    = dataset + str(patch_size)
data_size  = patch_size**2
code_size  = data_size * args.overComplete

# Optimization parameters.
maxEpoch  = args.max_epochs
batch_size = args.batch_size

learnRate = args.learn_rate
learnRateDecay = args.learn_rate_decay

# Get the official optimizer name!
if args.opt_method.lower()=='sgd':
  optName   = 'SGD'
  optParams = {'lr':args.learn_rate}
elif args.opt_method.lower()=='nest':
  optName   = 'SGD'
  optParams = {'lr':args.learn_rate, 'momentum':args.momentum, 'nesterov':True}
elif args.opt_method.lower()=='adam':
 optName   = 'Adam'
 optParams = {'lr':args.learn_rate}
else:
  raise ValueError('Unfamiliar opt-method {} requested.'.format(args.opt_method))
optimizer_module = getattr(torch.optim, optName)

# Scheduler.
scheduler = None
if optName=='SGD':
    scheduler = torch.optim.lr_scheduler.StepLR(OPT, step_size = 1,
                                                gamma = args.learn_rate_decay)
  
# First set up the data using the data-seed.
print('\n* Loading dataset {}'.format(args.dataset))
#if   args.use_validation_size>0:
#  dataName  = 'valid_'
#TODO: valid!!
rand.seed(args.data_seed)
train_loader, test_loader = loadData(dataset, patch_size, batch_size)

if hasattr(train_loader, 'numSamples'):
  numTrData = train_loader.numSamples
  numTeData = test_loader.numSamples
else:
  numTrData = len(train_loader.dataset)
  numTeData = len(test_loader.dataset)


# Now set up CUDA and reset all RNG's.
device = torch.device("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
if device.type != 'cpu':
    print('\033[93m'+'Using CUDA'+'\033[0m')
    torch.cuda.manual_seed(args.seed)
rand.seed(args.seed)

# If requested, use HPO results.

# **** reproduce results using this function. to play, comment it out. *****
# batch_size, L1_weight, learnRate, learnRateDecay = fetchVizParams(datName)
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
        SH._save(args.save_filename)

    # Store training and test losses, sparsities, and reconstruction errors
    #     after each training epoch.
    SH.tr_perf = ddict(bw_loss=[],     epoch_loss=[],
                       bw_sparsity=[], epoch_sparsity=[],
                       bw_reconErr=[], epoch_reconErr=[])
    SH.te_perf = ddict(loss=[], sparsity=[], reconErr=[])
    
    #######################################################
    # (2) Set up everything. 
    #######################################################
    # 2.b) dictionary.
    model = dictionary(data_size, code_size, use_cuda=(device!='cpu'))
    # 2.c) encoder.
    encoder = ENCODER(data_size, code_size, device=device, n_iter=args.encode_iters)
    encoder.change_encode_algorithm_(args.encode_alg)
    setup_encoder = lambda m : encoder.initialize_weights_(m, init_type=args.encode_alg, L1_weight=args.L1_weight, mu=args.mu)
    # 2.d) optimizer.
    opt = optimizer_module(model.parameters(), **optParams)

    #######################################################
    # (3) Train model.
    #######################################################
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print('DICTIONARY TRAINING XXXXXXXXXXXXXXXXXXXX')

    for epoch in range(1, args.max_epochs+1):
      print('\nEpoch {} of {}.'.format(epoch, args.max_epochs))
      epoch_loss = 0
      epoch_reconErr = 0

      for batch_idx, (true_sigs, im_labels) in enumerate(train_loader):
        gc.collect()
        true_sigs = true_sigs.to(device)

        # (3.a) Code optimization.
        with torch.no_grad():
          setup_encoder(model)           # Have to reinitialize with new weights.
          code_est = encoder(true_sigs)  # Compute locally optimal codes.
          loss_fcn_batch = encoder.lossFcn(code_est, true_sigs)  # Compute loss.

        # (3.b) Weight optimization.
        # Forward Pass.
        sig_est = model(code_est)
        loss    = F.mse_loss(sig_est, true_sigs)
        # Backward Pass.
        loss.backward()
        opt.step()
        model.normalizeAtoms()
        if scheduler is not None:
          scheduler.step()

        # (3.c) Housekeeping.
        # LOSS
        SH.tr_perf['bw_loss'] += [loss_fcn_batch]
        epoch_loss +=  loss_fcn_batch

        # SPARSITY
        SH.tr_perf['bw_sparsity'] += [compute_sparsity(code_est)]
#        epoch_sparsity +=  loss_fcn_batch

        # RECON ERROR
        SH.tr_perf['bw_reconErr'] += [loss.item()]
        epoch_reconErr +=  loss.item()


        # Outputs to terminal
        if batch_idx % int(len(train_loader)/args.print_frequency) == 0:
          print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * true_sigs.size(0), numTrData,
                  100. * batch_idx / len(train_loader), loss.item()))
      # END BATCH-WISE LOOP THRU DATA

      epoch_loss     /= batch_idx+1
      epoch_reconErr /= batch_idx+1
      #TODO get printing thing from irevnet code...
      print('EPOCH = {} xxxxxxxxxxxxxxxxxxxxxxxxxxxxx'.format(epoch))
      print('epoch loss = {}'.format(epoch_loss))
      print('epoch recon err = {}'.format(epoch_reconErr))

# TODO: actually look up how to save "this" file etc.
dictSavePath = "trainedModels/" + dataset + str(patch_size) + "/"
# save model
# save convergence / training progress plots and dict atoms
# save the file used to create it all (i.e. this one)

#benjamin@anna-devbox02:~/LSALSApy$ CUDA_VISIBLE_DEVICES=3 python dict_pTest0.py






