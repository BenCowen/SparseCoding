# -*- coding: utf-8 -*-
"""
DRAFT: encoder
EXTRA: 

@author: BenJammin
"""
from FISTA import FISTA

class Encoder:
    # If you want to train a layer as a dictionary, i.e. treating its
    # output as a code, add the module(s) here:
    _linear_pytorch_mods = [nn.Linear, nn.Biliniear, nn.LazyLinear,
                                 nn.Con1d, nn.Conv2d, 
                                 nn.Conv3d, nn.ConvTranspose1d, 
                                 nn.ConvTraspose2d, nn.ConvTranspose3d,
                                 nn.LazyCon1d, nn.LazyConv2d, 
                                 nn.LazyConv3d, nn.LazyConvTranspose1d, 
                                 nn.LazyConvTraspose2d, nn.LazyConvTranspose3d]
    _activation_pytorch_mods = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid,
                                     nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, 
                                     nn.LogSigmoid, nn.MultiheadAttention, 
                                     nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, 
                                     nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid, 
                                     nn.SiLU, nn.Mish, nn.Softplus, 
                                     nn.Softshrink, nn.Softsign, nn.Tanh, 
                                     nn.Tanhshrink, nn.Threshold, nn.GLU, 
                                     nn.Softmin, nn.Softmax, nn.Softmax2d, 
                                     nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss]

    def __init__(self, config, model=None):
        # Need: n steps, all optimizer settings,
        
        # If no model is specified, a linear dictionary 
        #  (a.k.a. 1-layer 'fully connected' network) 
        #  is used along with FISTA for sparse encoding. So the
        #  forward model is an implementation of FISTA with the given
        #  dictionary.
        if model is None:
            self.model = FISTA(config)
            self.no_intermittant_codes = True
        
        # Analyze the encoder model for the purpose of keeping track of its
        #  intermediate codes.
        self.analyze_given_model(model)
            
    
    def prep_training(self, encoder_optimizer = None, code_optimizer = None):
        self.encoder_optimizer = encoder_optimizer
        self.code_optimizer    = code_optimizer
        
    def inference(self, batch):
        # In some cases inferences is just a forward pass:
        if self.no_intermittant_codes:
            self.codes = []
            return self.model(batch)
        else:
        # Sometimes it is desireable to keep track of intermittant codes:
            self.codes = []
            for layer in self.model:
                batch = layer(batch)
                if hasattr(layer, 'has_codes'):
                    self.codes.append(batch.clone().detach())
            return batch
            
    def optimize_codes(self, batch):
        # In some cases (such as dictionary learning), the codes
        # should not be optimized directly.
    
        # Perform requested steps of code optimization
        for step in range(self.n_code_opt_steps):
            pass
        
    def optimize_encoder(self):
        # I'm not sure why this would get called when the encoder
        #  wasn't trainable, but why not?
        if self.encoder_optiimzer is None:
            return
        
        # Perform requested steps of code optimization
        for step in range(self.n_encoder_opt_steps):
            pass
    
    def analyze_given_model(self, model, encode_preactivation = False):
        # Determine the intermediate layers whose outputs will be encoded.
        # In order to utilize the intermediate code functionality you must 
        # define the model as iterable through layers (see examples).
        if not hasattr(model, '__iter__'):
            self.no_intermittant_codes = True
        else:
            self.no_intermittant_codes = False
            # Default behavior is to treat any linear layer's output as a code
            #  (this is the implementation in Beyond Backprop paper).
            if not encode_preactivation:
                for layer in model:
                    if layer in self._linear_pytorch_mods:
                        layer.has_codes = True
            # If encode_preactivation == True, then inputs to activations are
            #  treated as codes instead.
            else:
                for layer_idx, layer in enumerate(model):
                    if layer in self._activation_pytorch_mods:
                        model[layer_idx-1].has_codes = True
        