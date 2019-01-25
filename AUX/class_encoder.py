

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from class_dict import dictionary

class encoder(nn.Module):
  """
  Class of (trainable) encoders. Need to call "initialize_weights" before use!!
  """
  def __init__(self, data_size, code_size,
               encodeAlgorithm = None,
               n_iter          = 1,
               init_Dict       = None,
               datName = 'noname', device='cpu', useBias=False):
    super(encoder, self).__init__()

    # Logistics
    self.encodeAlgorithm = encodeAlgorithm
    self.device          = device
    self.code_size       = code_size
    self.data_size       = data_size

    # Linear Transforms.
    self.n_iter = n_iter
    self.We     = nn.Linear(data_size, code_size, bias=useBias)
    self.S      = nn.Linear(code_size, code_size, bias=useBias)
    self.shrink = None

    # Defaults
    self.mu = None
    self.L1_weight = None

############################################################################
# Some miscellaneous methods.
#############################
  def change_n_iter_(self, n_iter):
    """ Change number of iterations performed during forward-pass."""
    self.n_iter = n_iter

  def change_encode_algorithm_(self, new_alg):
    """ Change algorithm used during forward-pass."""
    self.encodeAlgorithm = new_alg
    print('Encoding with algorithm {}.'.format(new_alg))

  def change_L1_weight_(self, l1_weight):
    """ Change L1 weight."""
    self.L1_weight = l1_weight
    self.shrink   = nn.Softshrink(self.L1_weight/self.mu)

  def change_mu_(self, mu):
    """ Change mu."""
    self.mu     = mu
    self.shrink = nn.Softshrink(self.L1_weight/self.mu)
    if (self.encodeAlgorithm != 'salsa') and (mu!=1):
      print('WARNING: USING mu!=1 WITH ISTA-TYPE METHOD!!')

  def initialize_cvx_lossFcn_(self):
    """
    Initializes loss function based on given parameters.
    Will fail if L1 weight is not set.
    AVERAGE LOSS FCN VALUE ACROSS BATCHES!!!
    """
    if self.L1_weight is None:
      raise ValueError('Must intialize L1 weight to compute loss function.')

    # Need to counter-multiply by L if ISTA-type initialization was used.
    if self.init_type == 'ista':
      AT = lambda x : F.linear(x, self.L*self.We.weight.t())
    else:
      AT = lambda x : F.linear(x, self.We.weight.t())

    dataFidelity = lambda x,y:  (0.5*(AT(x)- y).norm(2)**2)/x.size(0)
    L1_loss      = lambda x: (self.L1_weight * x.norm(1)**2)/x.size(0)
    self.lossFcn = lambda x,y: (dataFidelity(x,y) + L1_loss(x)).item()
############################################################################
# Initialize the encoder according to some algorithm.
#####################################################
  def initialize_weights_(self, Dict = None, L1_weight = None, alg_fam='ista', mu=None):
    """
    Fully initializes the encoder using given weight matrices
      (or randomly, if none are given).
    """
    self.init_type = alg_fam
    # fix-up L1 weights and mu's.
    if self.L1_weight is None:
      if (L1_weight is None):
        print('Using default L1 weight (0.1).')
        self.L1_weight = 0.1
      else:
        self.L1_weight = L1_weight
    if self.mu is None:
      if mu is None:
        self.mu = 1
        if alg_fam=='salsa':
          print('Using default mu value (1).')
      else:
        self.mu = mu
    # If a dictionary is not provided for initialization, initialize randomly.
    if Dict is None:
      Dict = dictionary(self.data_size, self.code_size, use_cuda=False)

    #-------------------------------------
    # Initialize ISTA-style (first order).
    Wd = Dict.getDecWeights().cpu()
    if alg_fam=='ista':
      # Get the maximum eigenvalue.
      Dict.getMaxEigVal()
      self.L = Dict.maxEig
      # Initialize.
      self.We.weight.data = (1/self.L)*(Wd.detach()).t()
      self.S.weight.data = torch.eye(Dict.n) - (1/self.L)*(torch.mm(Wd.t(),Wd)).detach()
      # Set up the nonlinearity, aka soft-thresholding function.
      self.shrink = nn.Softshrink(self.L1_weight)

    #---------------------------------------
    # Initialize SALSA-style (second order).
    elif alg_fam=='salsa':
      # Initialize matrices.
      self.We.weight.data = Wd.detach().t()
      AA = torch.mm(Wd.t(), Wd).cpu()
      S_weights = (self.mu*torch.eye(Dict.n) + AA).inverse()
      self.S.weight.data = S_weights.detach()
      # Set up the nonlinearity, aka soft-thresholding function.
      self.shrink = nn.Softshrink(self.L1_weight/mu)

    else:
      raise ValueError('Encoders can only be initialized for "ista" and "salsa" like families.')

    #-------------------------------------
    # Initialize the loss function.
    self.initialize_cvx_lossFcn_()

    #-------------------------------------
    # Print status of the newly created encoder.
    print('Encoder and loss function are initialized for {}-type algorithms.'.format(alg_fam))

    #-------------------------------------
    # Finally, put to device if requested.
    self.We = self.We.to(self.device)
    self.S  = self.S.to(self.device)

############################################################################..............
# Now we must define the forward function.
#####################################################
  def forward(self, data, return_loss = False):
    """
    Forward method selects between one of the below algorithms.
    """
    return getattr(self, self.encodeAlgorithm+'_forward')(data, return_loss)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  def ista_forward(self, data, return_loss=False):
    """
    Implements ISTA.
    """
    if return_loss:
      loss = []
    # Initializations.
    We_y = self.We(data)
    x    = We_y.clone()
    # Iterations.
    for n in range(self.n_iter):
      x = self.shrink(We_y + self.S(x))
      if return_loss:
        loss += [self.lossFcn(x,data)]
    # Fin.
    if return_loss:
      return x, loss
    return x
#...............................
  def fista_forward(self, data, return_loss=False):
    """
    Implements FISTA.
    REMEMBER that self.We has been scaled by (1/L).
    """
    if return_loss:
      loss = []
    # Initializations.
    yk    = self.We(data)
    xprev = torch.zeros(yk.size())
    t  = 1
    # Iterations.
    for it in range(self.n_iter):
      # Update logistics.
      residual = F.linear(yk, self.L*self.We.weight.t()) - data;
    # ISTA step 
      tmp = yk - self.We(residual)#/self.L
      xk  = self.shrink.forward(tmp)
    # FISTA stepsize update:
      tnext = (1 + (1+4*(t**2))**.5)/2 
      fact  = (t-1)/tnext
    # Use momentum to update code estimate.
      yk    = xk + (xk-xprev)*fact
    # Update "prev" stuff.
      xprev  = xk
      t      = tnext
      if return_loss:
        loss += [self.lossFcn(x,data)]
    # Fin.
    if return_loss:
      return yk, loss
    return yk
#...............................
  def salsa_forward(self, data, return_loss=False):
    """
    Implements SALSA.
    """
    if return_loss:
      loss = []
    We_y = self.We(data)
    x    = We_y
    d    = torch.zeros(We_y.size()).to(device)
    for it in range(self.n_iter):
      u = self.shrink(x+d)
      x = self.S(We_y + self.mu*(u-d))
      d += x - u
      if return_loss:
        loss += [self.lossFcn(x,data)]
    # Fin.
    if return_loss:
      return x, loss
    return x
##########################################################################################
# Now we can add some special cases for MCA...?...








##########################################################################################
##########################################################################################
##########################################################################################
# SANITY tests!!!
#def sanity():
#  """
#  Tests functionality of encoder class.
#  """
import matplotlib.pyplot as plt

device='cpu'

data_size = 5
code_size = 10
e = encoder(data_size, code_size, device=device, n_iter=200)

batchSize=3
data = torch.ones(batchSize, data_size).to(device)

all_loss={}
comboID = 0
for D in [None, dictionary(data_size, code_size)]:
  for algFam in ['ista', 'salsa']:
    for forwardMethod in ['ista', 'fista', 'salsa']:
      comboID +=1
      print('combo {}--------------------------'.format(comboID))
      e.initialize_weights_(Dict = D, alg_fam=algFam, mu=1.0)
      e.change_encode_algorithm_(forwardMethod)
      x, loss = e(data, return_loss=True)
      all_loss[str(comboID)] = loss




for k in all_loss:
  plt.figure()
  plt.clf()
  plt.plot(all_loss[k])
  plt.savefig('./sanity_ims/combo'+k+'.png')












