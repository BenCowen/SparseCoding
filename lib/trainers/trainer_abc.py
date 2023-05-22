"""
Class
"""

import os
import torch
from abc import ABC

class Trainer(ABC):
    def __init__(self, config):
        self.config = config
        self.max_epoch = config['max-epoch']
        self.prints_per_epoch = config['prints-per-epoch']
        self.batches_per_print = None
        if 'batches-per-epoch' in config:
            self.max_batches = config['batches-per-epoch']
        else:
            self.max_batches = torch.inf
        self.optimizer = self.scheduler = self.training_record = self.loss_fcn = None
        self.epoch = None
        self.bb_config = None

    def train(self, model, dataset, loaded_objects={}):
        pass

    def save_train_state(self, model, training_hist, optimizer, scheduler):
        # Abbrev:
        sdir = self.config['save-dir']
        torch.save(model, os.path.join(sdir, 'model.pt'))
        torch.save(training_hist, os.path.join(sdir, 'training_record.pt'))
        torch.save(optimizer, os.path.join(sdir, 'optimizer.pt'))
        torch.save(scheduler, os.path.join(sdir, 'scheduler.pt'))

    def _parse_loaded_objects(self, exp_backup):
        """ Attribute everything in the obj """
        for key, value in exp_backup.items():
            setattr(self, key, value)

    def batch_print(self, loss_value, batch_idx, n_batches, other_prints={}):
        """ other_prints is a dict containin name/value pairs to be printed"""
        if batch_idx % self.batches_per_print == 0:
            print_str = f"Epoch {self.epoch}, Batch {batch_idx}/{n_batches}, Loss {loss_value:.2E}"
            for name, val in other_prints.items():
                print_str += f", {name} {val:.2E}"
            print(print_str)