import torch
import torch.nn as nn
import torch.nn.functional as F
from UTILS.class_dict import dictionary

class ENCODER(nn.Module):
  """
  Class of (trainable) encoders. Need to call "initialize_weights" before use!!
  """
  def __init__(self, data_size, code_size,
               encodeAlgorithm = None,
               n_iter          = 1,
               init_Dict       = None,
               datName = 'noname', device='cpu', useBias=False):
    super(ENCODER, self).__init__()

    # Logistics
    self.encodeAlgorithm = encodeAlgorithm
    self.device          = device
    self.code_size       = code_size
    self.data_size       = data_size

    # Linear Transforms.
    self.n_iter = n_iter
    self.We     = nn.Linear(data_size, code_size, bias=useBias)
    self.S      = nn.Linear(code_size, code_size, bias=useBias)

    # Defaults
    self.mu        = None
    self.L1_weight = None
    self.thresh    = None

############################################################################
# Some miscellaneous methods.
#############################
  def change_n_iter_(self, n_iter):
    """ Change number of iterations performed during forward-pass."""
    self.n_iter = n_iter

  def change_encode_algorithm_(self, new_alg):
    """ Change algorithm used during forward-pass."""
    self.encodeAlgorithm = new_alg
    print('Encoding with algorithm {}; NOT updating threshold!'.format(new_alg))
###############################################
## The following are for setting the threshold.
#
#  def change_L1_weight_(self, l1_weight):
#    """ Change L1 weight."""
#    self.L1_weight = l1_weight
#    self.update_threshold_()
#    print('L1 weight changed to {}; threshold updated.')
#
#  def change_mu_(self, mu):
#    """ Change mu; changing the threshold accordingly if encode_alg == salsa."""
#    self.mu     = mu
#    self.update_threshold_()
#    if (self.encodeAlgorithm == 'salsa') and (mu!=1):
#      print('WARNING: USING mu!=1 WITH ISTA-TYPE METHOD!!')
#
  def update_threshold_(self):
    """
    Implicitly updates the threshold based on self.L1_weights,
      self.mu, and self.We.
    """
    if self.encodeAlgorithm in ['ista' ,'fista']:
      self.update_L_()
      self.thresh = self.L1_weight /self.L
    elif self.encodeAlgorithm == 'salsa':
      self.thresh = self.L1_weight /self.mu

  def update_L_(self, iters=20):
    """
    Find Maximum Eigenvalue using Power Method
    """
    with torch.no_grad():
      bk = torch.ones(1 ,self.code_size).to(self.device)
    
      for n in range(0 ,iters):
        f = bk.abs().max()
        bk = bk /f
        bk = self.We(F.linear(bk, self.We.weight.t()))
      self.maxEig = bk.abs().max().item()

#################################################

  def initialize_cvx_lossFcn_(self, Dict):
    """
    Initializes loss function based on given parameters.
    Will fail if L1 weight is not set.
    AVERAGE LOSS FCN VALUE ACROSS BATCHES!!!
    """
    if self.L1_weight is None:
      raise ValueError('Must intialize L1 weight to compute loss function.')

    # Need to counter-multiply by L if ISTA-type initialization was used.

    dataFidelity = lambda x ,y:  (0.5 *(Dict(x )- y).norm(2 )**2 ) /x.size(0)
    L1_loss      = lambda x: (self.L1_weight * x.norm(1 )**2 ) /x.size(0)
    self.lossFcn = lambda x ,y: (dataFidelity(x ,y) + L1_loss(x)).item()
############################################################################
# Initialize the encoder according to some algorithm.
#####################################################
  def initialize_weights_(self, Dict = None, L1_weight = None, init_type='ista', mu=None):
    """
    Fully initializes the encoder using given weight matrices
      (or randomly, if none are given).
    """
    self.init_type = init_type
    # fix-up L1 weights and mu's.
    if self.L1_weight is None:
      if L1_weight is None:
        print('Using default L1 weight (0.1).')
        self.L1_weight = 0.1
      else:
        self.L1_weight = L1_weight
    if self.mu is None:
      if mu is None:
        self.mu = 1
        if init_type =='salsa':
          print('Using default mu value (1).')
      else:
        self.mu = mu
    # If a dictionary is not provided for initialization, initialize randomly.
    if Dict is None:
      Dict = dictionary(self.data_size, self.code_size, use_cuda=False)

    # -------------------------------------
    # Initialize the loss function.
    self.initialize_cvx_lossFcn_(Dict)

    # -------------------------------------
    # Initialize ISTA-style (first order).
    Wd = Dict.getDecWeights().cpu()
    if init_type=='ista':
      # Get the maximum eigenvalue.
      Dict.getMaxEigVal()
      self.L = Dict.maxEig
      # Initialize.
      self.We.weight.data = ( 1 /self.L ) *(Wd.detach()).t()
      self.S.weight.data = torch.eye(Dict.n) - ( 1 /self.L ) *(torch.mm(Wd.t() ,Wd)).detach()
      self.thresh = self.L1_weight /self.L
      # Set up the nonlinearity, aka soft-thresholding function.

    # -------------------------------------
    # Initialize FISTA-style (first order).
    elif init_type=='fista':
      # Get the maximum eigenvalue.
      Dict.getMaxEigVal()
      self.L = Dict.maxEig
      # Initialize.
      self.We.weight.data = Wd.detach().t()
      self.thresh = self.L1_weight /self.L
      # Set up the nonlinearity, aka soft-thresholding function.

    # ---------------------------------------
    # Initialize SALSA-style (second order).
    elif init_type =='salsa':
      # Initialize matrices.
      self.We.weight.data = Wd.detach().t()
      AA = torch.mm(Wd.t(), Wd).cpu()
      S_weights = (self.mu *torch.eye(Dict.n) + AA).inverse()
      self.S.weight.data = S_weights.detach()
      self.thresh = self.L1_weight /self.mu
      # Set up the nonlinearity, aka soft-thresholding function.

    else:
      raise ValueError('Encoders can only be initialized for "ista" and "salsa" like families.')

    # -------------------------------------
    # Print status of the newly created encoder.
#    print('Encoder, threshold, and loss functions are initialized for {}-type algorithms.'.format(init_type))

    # -------------------------------------
    # Finally, put to device if requested.
    self.We = self.We.to(self.device)
    self.S  = self.S.to(self.device)

  def printStats(self):
    print('Initialization type = ' + self.init_type)
    print('Encoding Algorithm  = ' + self.encodeAlgorithm)
############################################################################..............
# Now we must define the forward function.
#####################################################
  def forward(self, data, return_loss = False):
    """
    Forward method selects between one of the below algorithms.
    """
    if self.init_type != self.encodeAlgorithm:
      raise NotImplemented('Not yet supporting mix-and-match init and forward types.')
    return getattr(self, self.encodeAlgorithm +'_forward')(data, return_loss)

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
      x = F.softshrink(We_y + self.S(x), self.thresh)
      if return_loss:
        loss += [self.lossFcn(x ,data)]
    # Fin.
    if return_loss:
      return x, loss
    return x

  # ...............................
  def fista_forward(self, data, return_loss=False):
    """
    Implements FISTA.
    """
    if return_loss:
      loss = []
    # Initializations.
    yk    = self.We(data)
    xprev = torch.zeros(yk.size()).to(self.device)
    t  = 1
    # Iterations.
    for it in range(self.n_iter):
      # Update logistics.
      residual = F.linear(yk, self.We.weight.t()) - data;
    # ISTA step 
      tmp = yk - self.We(residual ) /self.L
      xk  = F.softshrink(tmp, lambd=self.thresh)
    # FISTA stepsize update:
      tnext = (1 + ( 1 + 4 *( t**2) )**.5 ) /2
      fact  = ( t -1 ) /tnext
    # Use momentum to update code estimate.
      yk    = xk + (xk -xprev) *fact
    # Update "prev" stuff.
      xprev  = xk
      t      = tnext
      if return_loss:
        loss += [self.lossFcn(yk ,data)]
    # Fin.
    if return_loss:
      return yk, loss
    return yk

  # ...............................
  def salsa_forward(self, data, return_loss=False):
    """
    Implements SALSA.
    """
    if return_loss:
      loss = []
    We_y   = self.We(data)
    x      = We_y
    d      = torch.zeros(We_y.size()).to(self.device)
    for it in range(self.n_iter):
      u = F.softshrink(x + d, lambd=self.thresh)
      x = self.S(We_y + self.mu *(u -d))
      d += x - u
      if return_loss:
        loss += [self.lossFcn(x ,data)]
    # Fin.
    if return_loss:
      return x, loss
    return x
##########################################################################################
# Now we can add some special cases for MCA...?...

##########################################################################################
##########################################################################################

def compute_sparsity(x):
  return (x.gt(0).sum( ) /x.numel()).item()

##########################################################################################
##########################################################################################
##########################################################################################
# SANITY tests!!!
def encoder_class_sanity():
  """
  Tests functionality of encoder class.
  """
  import matplotlib.pyplot as plt
  torch.manual_seed(3)
  device ='cpu'

  data_size = 15
  code_size = 15
  n_iter    = 75
  e = ENCODER(data_size, code_size, device=device, n_iter=n_iter)

  # Create solutions (sparse codes).
  batch_size =30
  codeHeight = 1.5
  x_true = torch.zeros(batch_size, code_size).to(device)
  for batch in range(batch_size):
    randi = torch.LongTensor(int(code_size /2)).random_(0, code_size)
    x_true[batch][randi] = codeHeight *torch.rand(x_true[batch][randi].size()).to(device)

  # Create the dictionary.
  D = dictionary(data_size, code_size, use_cuda=(device!='cpu'))

  # Create the observations based on signal model
  #    y = Dx + w
  # where w is white noise
  sigma = 1.5
  data = D(x_true) + sigma *torch.rand(batch_size, data_size).to(device)

  # Dictionary for collecting results.
  forwardMethods = ['ista', 'fista', 'salsa']
  all_loss ={}
  for f in forwardMethods:
    all_loss[f] = {}

  # Run the experiment.
  for meth in forwardMethods:
    print('{} init with {} algorithm--------------------------'.format(meth ,meth))
    e.initialize_weights_(Dict = D, init_type=meth, mu=1, L1_weight=0.05)
    e.change_encode_algorithm_(meth)
    with torch.no_grad():
      x, loss = e(data, return_loss=True)

    if meth =='ista':
      all_loss[meth]['loss'] = loss
      all_loss[meth]['x']    = x
    elif meth =='fista':
      all_loss[meth]['loss'] = loss
      all_loss[meth]['x']    = x
    elif meth =='salsa':
      all_loss[meth]['loss'] = loss
      all_loss[meth]['x']    = x


  # Visualize results.
  for k in all_loss:
    plt.figure(1)
    plt.clf()
    plt.plot(all_loss[k]['loss'])
    plt.annotate('%0.3f' % all_loss[k]['loss'][-1],
                  xy=(1 ,all_loss[k]['loss'][-1]), xytext=(8 ,0),
                  xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.title( k +' loss')
    plt.savefig('./sanity_ims/fixedEncoders/ ' + k +'_loss.png')

    plt.figure(2)
    plt.clf()
    plt.stem(all_loss[k]['x'][0].numpy())
    plt.title( k +' solution 0 ')
    plt.savefig('./sanity_ims/fixedEncoders/ ' + k +'_x.png')


  ###############################################################################
  print('xxxxxxxxxxx Now try a little learning on each architecture xxxxxxxxxxx')

  all_loss ={}
  max_epoch =150
  for meth in ['ista' ,'salsa', 'fista']:
    all_loss[meth ] ={}
    print('TRAINING w/ init = {} and alg = {}--------------------------'.format(meth ,meth))
    e.initialize_weights_(Dict = D, init_type=meth, mu=1, L1_weight=0.05)
    e.change_encode_algorithm_(meth)

    # Set up optimizer.
    opt = torch.optim.Adam(e.parameters(), lr=0.001)

    # Compute labels ("optimal codes").
    with torch.no_grad():
      e.change_n_iter_(1000)
      labels = e(data)
      e.change_n_iter_(10)

    # Loss function
    loss = lambda x : F.mse_loss(x, labels)
    loss_hist = []

    for epoch in range(1 ,max_epoch):
      opt.zero_grad()
      # Forward Pass
      x = e(data.detach())
      # Backward Pass
      err = loss(x)
      err.backward()
      opt.step()
      loss_hist += [err.item()]

    all_loss[meth]['loss'] = loss_hist
    all_loss[meth]['x']    = x.detach()


  # Visualize results.
  for k in all_loss:
    plt.figure(1)
    plt.clf()
    plt.plot(all_loss[k]['loss'])
    plt.annotate('%0.3f' % all_loss[k]['loss'][-1],
                  xy=(1 ,all_loss[k]['loss'][-1]), xytext=(8 ,0),
                  xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.title( k +' Training Loss')
    plt.savefig('./sanity_ims/trainedEncoders/ ' + k +'_loss.png')

    plt.figure(2)
    plt.clf()
    plt.stem(all_loss[k]['x'][0].numpy())
    plt.title( k +' solution after training')
    plt.savefig('./sanity_ims/trainedEncoders/ ' + k +'.png')




