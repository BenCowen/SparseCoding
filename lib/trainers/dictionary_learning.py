"""
Dictionary learning with fixed encoder.

@author Benjamin Cowen
@date: Feb 24 2018 (update April 2023)
@contact benjamin.cowen.math@gmail.com
"""

import gc
import torch
import types
import lib.UTILS.path_support as torch_aux
from lib.UTILS.path_support import import_from_specified_class
from lib.UTILS.image_transforms import OverlappingPatches


class DictionaryLearning:
    """
    Use stochastic optimization to train a dictionary of atoms for sparse reconstruction.
    """

    def __init__(self, config):
        # First ensure consistency between Set up non-trainable encoder
        self.config = config
        self.max_epoch = config['max-epoch']
        self.batch_print_frequency = config['batch-print-frequency']
        self.optimizer = self.scheduler = self.training_record = self.loss_fcn = None
        self.epoch = -1
        self.encoder = None
        # TODO: make this optional? why tho?...
        self.Patcher = OverlappingPatches(config['OverlappingPatches'])

    def initialize_encoder(self, model_config):
        # Need to add some things to encoder config so it's consistent with the model:
        self.config['encoder-config']['data-len'] = model_config['data-len']
        self.config['encoder-config']['code-len'] = model_config['code-len']
        self.config['encoder-config']['device'] = model_config['device']
        self.encoder = import_from_specified_class(self.config, 'encoder')

    def train(self, backbone_config, model, dataset):
        """ Train the dictionary! """

        # Configure GPU usage with pinned memory dataloader
        model = model.to(self.config['device'], non_blocking=True)
        self.encoder = self.encoder.to(self.config['device'], non_blocking=True)

        # Assign loss function to self:
        loss_fcn = torch_aux.generate_loss_function(self.config['loss-config'], dataset)

        # Initialize the optimizer for our model
        self.optimizer, self.scheduler = torch_aux.setup_optimizer(self.config['optimizer-config'], model)

        # Keep record of the training process
        self.training_record = {'loss-hist': [],
                                'sparsity-hist': []}

        while self.epoch < self.max_epoch:
            self.epoch += 1
            batch_loss = 0
            batch_sparsity = 0
            for batch_idx, (batch, _) in enumerate(dataset.train_loader):
                # Reshape channel and patch dimensions into batch dimension
                batch = batch.to(self.config['device'], non_blocking=True)
                patches = self.Patcher(batch)
                model.zero_grad()
                gc.collect()

                # Code inference (encode the batch)
                with torch.no_grad():
                    model.normalize_columns()
                    self.encoder.update_encoder_with_dict(model)
                    opt_codes = self.encoder(patches.view(-1, patches.shape[-1]))

                # Encoder Optimization
                # Forward pass
                est_patches = model(opt_codes).view(patches.shape)
                batch_est = self.Patcher.reconstruct(est_patches, batch.shape[-2:])
                loss_value = loss_fcn(batch, batch_est, opt_codes)

                # Backward pass:
                loss_value.backward()
                self.optimizer.step()
                self.scheduler.step()

                # Logistics
                with torch.no_grad():
                    loss_value_item = loss_value.detach().item()
                    batch_loss += loss_value_item
                    batch_sparsity += opt_codes.abs().gt(0).sum().detach().item()/opt_codes.numel()
                    if batch_idx % self.batch_print_frequency == 0:
                        print(f"Epoch {self.epoch} > Batch {batch_idx} > Loss value {loss_value_item:.2E}")
                    # Save trainer
                    torch_aux.save_train_state(backbone_config, model, self.training_record)

            self.training_record['loss-hist'].append(batch_loss)
            self.training_record['sparsity-hist'].append(batch_sparsity / batch_idx)
            print(
                f"Epoch {self.epoch}/{self.max_epoch} Complete: loss = {batch_loss:.2E}, avg-sparsity = {100 * batch_sparsity / batch_idx:.2E}")
            self.print_dictionary_atoms(model, patches, est_patches, batch.shape[:2])

    # def print_dictionary_atoms(self, model, patches, est_patches, batch_shape[:2]):

