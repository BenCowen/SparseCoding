"""
Dictionary learning with fixed encoder.

@author Benjamin Cowen
@date: 4 Feb 2024
@contact benjamin.cowen.math@gmail.com
"""

import torch
from lib.UTILS.image_transforms import OverlappingPatches, get_sparsity_ratio
from lib.UTILS.visualizer import Visualizer
from lib.tasks.trainer_abc import Trainer


class EncodeDecodeTrainer(Trainer):
    """
    Use stochastic optimization to train a dictionary of atoms for sparse reconstruction.
    """

    def __init__(self, config):
        super(EncodeDecodeTrainer, self).__init__(config)
        # Most stuff taken care of in super
        self.recon_example = None
        self.Patcher = None
        self.viz = None

    def pretraining_todos(self, stuff):
        """Train an Encoder and/or decoder."""

        # Initialize Visualizer
        self.viz = Visualizer(self.config['save-dir'])
        self.Patcher = OverlappingPatches(self.config['post-load-transforms']['OverlappingPatches'])

        for model_name, model_config in self.config["models"].items():
            self.models[model_name] = stuff[model_config['stuff-name']].to(self.config['device'], non_blocking=self.use_blocking)
            opt_class = getattr(torch.optim, model_config['optimizer']['name'])
            sch_class = getattr(torch.optim.lr_scheduler, model_config['scheduler']['name'])
            self.optimizer[model_name] = opt_class(self.models[model_name].parameters(),
                                                    **model_config['optimizer']['kwargs'])
            self.scheduler[model_name] = sch_class(self.optimizer[model_name],
                                                   **model_config['scheduler']['kwargs'])
        self._parse_loaded_objects(loaded_objects)
        self.epoch = self.training_record['epoch']

        return stuff['dataset-name'].train_loader, stuff['dataset-name'].valid_loader

    def forward(self, batch):
        # Reshape channel and patch dimensions into batch dimension
        batch = batch.to(self.config['device'],
                         non_blocking=self.use_blocking)
        patches = self.Patcher(batch)

        # Code inference (encode the batch)
        with torch.no_grad():
            self.models['decoder'].normalize_columns()
            self.models['encoder'].sync_to_decoder(self.models['decoder'])
            opt_codes = self.models['encoder'](patches.view(-1, patches.shape[-1]))

        # Encoder Optimization
        # Forward pass
        est_patches = self.models['decoder'](opt_codes).view(patches.shape)
        est_batch = self.Patcher.reconstruct(est_patches, batch.shape[-2:])
        return {"loss": self.loss_fcn(batch, est_batch, opt_codes)}

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
