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
import numpy as np
import matplotlib.pyplot as plt

def printProgressFigs(savePath, extension, lossHist, errHist, spstyHist):
  """
  Prints loss, reconstruction error, and sparsity history plots.
  savePath  : a string with the path to the directory (including "/")
  extension : a string such a ".png", ".pdf", etc.
  other inputs are np.array objects containing 1D data vectors
    to be plotted.
  """

  #######################################################
  # Visualize loss, reconstruction error, and sparsity
  #  histories.
  #######################################################
  x = np.asarray([i for i in range(0, len(lossHist))])
  ################
  ## loss history
  plt.figure()
  plt.plot(x, lossHist)
  plt.xlabel("Batches")
  plt.ylabel("Loss (averaged over samples)")
  plt.title("Loss Function History")
  plt.savefig(savePath + "lossHist" + extension)
  plt.close()

  ################################
  ## reconstruction error history
  plt.figure()
  plt.plot(x, errHist)
  plt.xlabel("Batches")
  plt.ylabel("MSE")
  plt.title("Reconstruction Error History")
  plt.savefig(savePath + "reconErrHist" + extension)
  plt.close()

  ####################
  ## sparsity history
  plt.figure()
  plt.plot(x, spstyHist)
  plt.xlabel("Batches")
  plt.ylabel("% zeros")
  plt.title("Sparsity History")
  plt.savefig(savePath + "sparsityHist" + extension)
  plt.close()


