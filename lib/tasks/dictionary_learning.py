"""
Dictionary learning with fixed encoder.

@author Benjamin Cowen
@date: Feb 24 2018 (update April 2023)
@contact benjamin.cowen.math@gmail.com
"""

import gc
import torch
import lib.UTILS.path_support as torch_aux
from lib.UTILS.image_transforms import OverlappingPatches, get_sparsity_ratio
from lib.UTILS.visualizer import Visualizer
from lib.tasks.trainer_abc import Trainer
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

class DictionaryPatchTrainer(Trainer):
    """
    Use stochastic optimization to train a dictionary of atoms for sparse reconstruction.
    """

    def __init__(self,
                 post_load_transforms: Dict = None,
                 encoder_config: Dict = None,
                 decoder_config: Dict = None,
                 dataset_name: str = None,
                 save_dir: str = None,
                 loss_config: List = None,
                 **kwargs):
        super(DictionaryPatchTrainer, self).__init__(**kwargs)
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        self.recon_example = None
        self.Patcher = OverlappingPatches(post_load_transforms['OverlappingPatches'])
        self.viz = None
        self.dataset_name = dataset_name
        self.save_dir = Path(save_dir)
        self.loss_config = loss_config

    def run_task(self, loaded_objects):
        """ Train the dictionary! """

        # Initialize Visualizer
        self.viz = Visualizer(self.save_dir)

        # Extract models
        self.model = loaded_objects[self.decoder_config['stuff-name']]
        self.encoder = loaded_objects[self.encoder_config['stuff-name']]

        # Dataloader
        dataset = loaded_objects[self.dataset_name]

        # Setup optimizer
        self.optimizer, self.scheduler = self.setup_optimizer(self.decoder_config['optimizer-config'],
                                                              self.model)

        # Setup loss fcn:
        loss_fcn = self.generate_loss_function(self.loss_config)
        n_batches = min(self.max_batches,
                        len(dataset.train_loader))

        # Keep record of the training process
        self.training_record = {'epoch': 0,
                                'loss-hist': [],
                                'sparsity-hist': []}

        self._parse_loaded_objects(loaded_objects)
        self.epoch = self.training_record['epoch']

        while self.epoch < self.max_epoch:
            # Visualizations each epoch:
            self.epoch_visualizations(self.model,
                                      dataset.valid_loader)
            self.training_record['epoch'] = self.epoch
            # Add an epoch to the record:
            self.training_record['loss-hist'].append([])
            self.training_record['sparsity-hist'].append([])
            epoch_loop = tqdm(enumerate(dataset.train_loader),
                              total=n_batches,
                              desc=f"Epoch {self.epoch}")
            for batch_idx, (batch, _) in epoch_loop:
                # Reshape channel and patch dimensions into batch dimension
                batch = batch.to(self.model.device, non_blocking=True)
                patches = self.Patcher(batch)
                # f batch_idx > 2:
                #     break

                # Code inference (encode the batch)
                with torch.no_grad():
                    self.model.normalize_columns()
                    self.encoder.sync_to_decoder(self.model)
                    opt_codes = self.encoder(patches.view(-1, patches.shape[-1]))

                # Encoder Optimization
                # Forward pass
                est_patches = self.model(opt_codes).view(patches.shape)
                est_batch = self.Patcher.reconstruct(est_patches, batch.shape[-2:])
                loss_value = loss_fcn({'target': batch,
                                       'recon': est_batch,
                                       'code': opt_codes})

                # Backward pass:
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()
                self.scheduler.step()
                gc.collect()

                # Logistics
                with torch.no_grad():
                    loss_value_item = loss_value.detach().item()
                    last_sparsity = 1-get_sparsity_ratio(opt_codes)
                    self.training_record['loss-hist'][-1].append(loss_value_item)
                    self.training_record['sparsity-hist'][-1].append(last_sparsity)
                    epoch_loop.set_postfix({'loss': f"{loss_value_item:.2E}",
                                            'sparsity': f"{100*last_sparsity:.4f}%"})
                if batch_idx > self.max_batches:
                    break

            # End epoch:
            avg_sp = 100 * sum(self.training_record['sparsity-hist'][-1]) / len(self.training_record['sparsity-hist'][-1])
            print(f"EPOCH {self.epoch}/{self.max_epoch} COMPLETE: \n"
                  f"\tTotal Loss = {sum(self.training_record['loss-hist'][-1]):.2E}\n"
                  f"\tAVG-Sparsity = {avg_sp:.4f}%")

            # Save trainer
            self.save_train_state(self.model, self.training_record,
                                  self.optimizer, self.scheduler)

            self.epoch += 1
        # Done, return encoder, decoder
        self.encoder.sync_to_decoder(self.model)
        return {'trained-decoder': self.model,
                'trained-encoder': self.encoder}

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
                self.recon_example = batch[:4].to(model.device, non_blocking=True)
                break
            self.viz.array_plot(self.recon_example.view(2, 2, 3,
                                                        self.recon_example.shape[-2],
                                                        self.recon_example.shape[-1]),
                                save_name=f'original-example-images',
                                title_str=f'Original images',
                                color=True)

        # Encode the data and rank the atoms in terms of heaviest usage:
        model.normalize_columns()
        self.encoder.sync_to_decoder(model)
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
