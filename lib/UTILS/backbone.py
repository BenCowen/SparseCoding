"""
Config reader and wrapper for the whole system.

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""
import os, shutil
import torch
from lib.UTILS.path_support import import_from_specified_class


class BackBone:
    """ Backbone of the pipeline, controls experiment maintenence, data, etc. """

    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.model = None
        self.save_dir = config['save-dir']
        self.loaded_objects = {}

        # Identify compute device and share with all subconfigs
        self._device = config['device']

        # Set up results directory:
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            shutil.copy(self.config['config-path'],
                        os.path.join(self.save_dir, 'config-backup.yml'))

    def check_for_continuation(self):
        """
        Check if a save file already exists in the specified location. If so,
          and continuation is requested, load the corresponding model etc.
        """
        previous_experiment_exists = os.path.exists(self.save_dir) and len(os.listdir(self.save_dir)) > 0
        if previous_experiment_exists and self.config['allow-continuation']:
            return self.load()
        else:
            # Copy config
            if not os.path.exists(self.config['save-dir']):
                os.makedirs(self.config['save-dir'])
            shutil.copy(self.config['config-path'],
                        os.path.join(self.config['save-dir'],
                        os.path.basename(self.config['config-path'])))
            return self

    def load(self):
        """
        Loads in every .pt file it can find in save_dir and stores in loaded_objects.
        model treated specially.
        """
        print("Continuing from experiment results saved in "+self.save_dir)
        for filename in os.listdir(self.save_dir):
            if filename == 'saved_model.pt':
                self.model = torch.load(os.path.join(self.save_dir, filename))
            elif filename.endswith('.pt'):
                self.loaded_objects[filename.split('.')[0]] = torch.load(os.path.join(self.save_dir, filename))

    def configure_dataset(self):
        """ Retrieves dataloader from the specified class. """
        self.dataset = import_from_specified_class(self.config, 'data')
        if not hasattr(self.dataset, 'valid_loader'):
            self.dataset.valid_loader = self.dataset.train_loader

    def configure_model(self):
        """ Initializes model from the specified class. """

        # Get data-len from the dataloader, and add it to the model-config.
        self.config['model-config']['data-len'] = self.dataset.data_dim

        if self.model is not None:
            # If the model has already been loaded in, bail.
            return

        # Construct the model
        self.model = import_from_specified_class(self.config, 'model')

    def run_experiment(self):
        # Generate the trainer
        trainer = import_from_specified_class(self.config, 'trainer')

        # Save the experiment

        # Execute
        trainer.train(self.model, self.dataset, self.loaded_objects)

