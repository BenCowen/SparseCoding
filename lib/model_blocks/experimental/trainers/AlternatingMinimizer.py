# -*- coding: utf-8 -*-
"""
REALLY EXCITED TO DO THIS BUT GONNA FOCUS ON DICT LEARNING FOR NOW :(

@author: BenJammin
"""

class AlternatingMinimizer:
    def __init__(self, config):
        # Parse, initialize all training parameters
        
        # Intermediate logging?
        
    def train(self, model_config):
        """
        Uses the defaults specified in the config if none are given.
        """
        # Set up models based on given config
        # Setup optimizer
        
        # Setup encoder
        coder = Coder(model)
        
        # Epoch 0 statistics
        self.epoch_update(encoder, 0)
        
        # Notes to delete later:
        # dict learning: mini_epochs=1, encoder_optimizer=None,code_optimizer=None
        # beyondBackprop: decoder_optimizer=None
        # classification: code_optimizer, decoder_optimizer = None
        
        # Train
        for epoch in range(1, n_epoch):
            for batch_idx, (batch, targets) in self.train_loader:
                # Alternate between codes/weights multiple times:
                for mini_epoch in range(self.mini_epochs(epoch)):
                    
                    # Step 0: Code Inference
                    # i.e. encode the batch (forward pass)
                    codes = coder.encode(batch, A)
                    
                    # Step 1: Code Optimization
                    opt_codes = coder.optimize_codes(codes)
                    
                    # Step 2: Encoder Optimization
                    # (backward pass w.r.t. inference)
                    coder.optimize_encoder(batch, opt_codes)
                    
                # Step 3: Decoder Optimization
                coder.optimize_decoder(opt_codes, targets)
                
                # Step 3: record progress
                self.batch_update(coder)
            
            # End epoch:
            self.epoch_update(coder, epoch)
            
        # Done training
        self.finish_stats()
        
    def batch_update(self):
    def epoch_update(self):
    def finish_stats(self):