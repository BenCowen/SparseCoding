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
from lib.UTILS.image_transforms import OverlappingPatches, get_sparsity_ratio
from lib.UTILS.visualizer import Visualizer
from lib.trainers.trainer_abc import Trainer


class DictionaryLearning(Trainer):
    """
    Use stochastic optimization to train a dictionary of atoms for sparse reconstruction.
    """

    def __init__(self, config):
        super(DictionaryLearning, self).__init__(config)
        self.recon_example = None
        self.Patcher = None
        self.viz = None
        self.encoder = self.initialize_encoder(self.config['model-config'])

    def initialize_encoder(self, model_config):
        # Need to add some things to encoder config so it's consistent with the model:
        self.config['encoder-config']['data-len'] = model_config['data-len']
        self.config['encoder-config']['code-len'] = model_config['code-len']
        self.config['encoder-config']['device'] = model_config['device']
        self.encoder = import_from_specified_class(self.config, 'encoder')

    def train(self, model, dataset, loaded_objects={}):
        """ Train the dictionary! """
        # Initialize Visualizer
        self.viz = Visualizer(self.config['save-dir'])
        self.Patcher = OverlappingPatches(self.config['post-load-transforms']['OverlappingPatches'])
        # Configure GPU usage with pinned memory dataloader
        model = model.to(self.config['device'], non_blocking=True)
        self.encoder = self.encoder.to(self.config['device'], non_blocking=True)

        # Assign loss function to self:
        loss_fcn = torch_aux.generate_loss_function(self.config['loss-config'], dataset)
        n_batches = min(self.max_batches, len(dataset.train_loader))
        self.batches_per_print = n_batches//self.prints_per_epoch

        # Initialize the optimizer for our model
        self.optimizer, self.scheduler = torch_aux.setup_optimizer(self.config['optimizer-config'], model)

        # Keep record of the training process
        self.training_record = {'epoch': 0,
                                'loss-hist': [],
                                'sparsity-hist': []}

        self._parse_loaded_objects(loaded_objects)
        self.epoch = self.training_record['epoch']

        while self.epoch < self.max_epoch:
            # Visualizations each epoch:
            self.epoch_visualizations(model, dataset.valid_loader)
            self.training_record['epoch'] = self.epoch
            # Add an epoch to the record:
            self.training_record['loss-hist'].append([])
            self.training_record['sparsity-hist'].append([])
            for batch_idx, (batch, _) in enumerate(dataset.train_loader):
                # Reshape channel and patch dimensions into batch dimension
                batch = batch.to(self.config['device'], non_blocking=True)
                patches = self.Patcher(batch)
                # f batch_idx > 2:
                #     break

                # Code inference (encode the batch)
                with torch.no_grad():
                    model.normalize_columns()
                    self.encoder.update_encoder_with_dict(model)
                    opt_codes = self.encoder(patches.view(-1, patches.shape[-1]))

                # Encoder Optimization
                # Forward pass
                est_patches = model(opt_codes).view(patches.shape)
                est_batch = self.Patcher.reconstruct(est_patches, batch.shape[-2:])
                loss_value = loss_fcn(batch, est_batch, opt_codes)

                # Backward pass:
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()
                self.scheduler.step()
                gc.collect()

                # Logistics
                with torch.no_grad():
                    loss_value_item = loss_value.detach().item()
                    last_sparsity = get_sparsity_ratio(opt_codes)
                    self.training_record['loss-hist'][-1].append(loss_value_item)
                    self.training_record['sparsity-hist'][-1].append(last_sparsity)
                    self.batch_print(loss_value_item, batch_idx, n_batches,
                                     other_prints={'Sparsity': last_sparsity})
                if batch_idx > self.max_batches:
                    break

            # End epoch:
            print("\tEPOCH {}/{} COMPLETE: Loss = {:.2E}, AVG-Sparsity = {:.2E}%".format(
                self.epoch, self.max_epoch, sum(self.training_record['loss-hist'][-1]),
                100 * sum(self.training_record['sparsity-hist'][-1]) / batch_idx))

            # Save trainer
            torch_aux.save_train_state(self.bb_config, model, self.training_record,
                                       self.optimizer, self.scheduler)

            self.epoch += 1

    @torch.no_grad()
    def epoch_visualizations(self, model, valid_loader):
        """
        Visualize:
         - dictionary atoms
         - original batch of data
         - patches
         - codes
         - code coefficient distribution
         - code-space as a volume
         - reconstructions
         - fill in patches 1 coefficient at a time from largest to smallest
         - glitch weights around... maybe something subtle... or reverse weighting...
        """
        # Update the loss plot so far (should really use tensorboard...
        self.viz.plot_loss(self.training_record, 'loss_history', 'Loss & Sparsity History')

        # Write out the originals of some example images the first time:
        epoch = self.training_record["epoch"]
        if self.recon_example is None:
            # Get a couple official samples that will be re-used for reconstruction visualization
            for batch, _ in valid_loader:
                self.recon_example = batch[:4].to(self.config['device'], non_blocking=True)
                break
            self.viz.array_plot(self.recon_example.view(2, 2, 3,
                                                        self.recon_example.shape[-2],
                                                        self.recon_example.shape[-1]),
                                save_name=f'original-example-images',
                                title_str=f'Original images',
                                color=True)

        # Encode the data and rank the atoms in terms of heaviest usage:
        model.normalize_columns()
        self.encoder.update_encoder_with_dict(model)
        patches = self.Patcher(self.recon_example)
        codes = self.encoder(patches.view(-1, patches.shape[-1]), n_iters=100)
        # Collapse data dimension to get total coefficient power:
        sorted_code_idx = codes.pow(2).sum(0).argsort()

        # Get the top 100 atoms, as weighted by the code coefficients they produce:
        atoms = model.decoder.weight.data.detach().transpose(0, 1).unfold(-1, self.Patcher.psz, self.Patcher.psz)
        top100atoms = atoms[sorted_code_idx[-100:], :].view(10, 10, self.Patcher.psz, self.Patcher.psz)

        # Visualize top dictionary atoms:
        loss_vals = [torch.nan] if len(self.training_record['loss-hist']) == 0 else self.training_record['loss-hist'][
            -1]
        perf_summary = 'Loss {:.2E}, Sparsity {:.1E}%'.format(
            sum(loss_vals),
            100 * get_sparsity_ratio(codes))

        self.viz.array_plot(top100atoms,
                            save_name=f'top100atoms_e{epoch}',
                            title_str=f'Top 100 atoms used for reconstruction at epoch {epoch}'
                                      + '\n' + perf_summary)

        # Visualize reconstruction using varying numbers of atoms:
        # for n_atoms in [1, 2, 3, 4, 5, 10, 100, 500, atoms.shape[0]]:

        est_recons = self.Patcher.reconstruct(model(codes).view(patches.shape), self.recon_example.shape[-2:])
        self.viz.array_plot(est_recons.view(2, 2, 3, est_recons.shape[-2], est_recons.shape[-1]),
                            save_name=f'recon_examples_e{epoch}',
                            title_str=f'Reconstruction at Epoch {epoch}',
                            color=True)
