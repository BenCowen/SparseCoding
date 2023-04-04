"""
Config reader and wrapper for the whole system.

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""
import os
from lib.UTILS.path_support import import_from_specified_class


class BackBone:
    """ Backbone of the pipeline, controls experiment maintenence, data, etc. """

    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.model = None
        self.save_dir = config['save-dir']

        # Identify compute device and share with all subconfigs
        if 'device' in config:
            self._device = config['device']
        else:
            self._device = 'cpu'
        # I hate this XD
        for key, _ in self.config.items():
            if key.endswith('-config'):
                self.config[key]['device'] = self._device

    def check_for_continuation(self):
        """
        Check if a save file already exists in the specified location. If so,
          and continuation is requested, load the corresponding model etc.
        """
        previous_experiment_exists = os.path.exists(self.save_dir) and len(os.listdir(self.save_dir)) > 0
        if previous_experiment_exists and self.config['allow-continuation']:
            return self.load()
        else:
            return self

    def configure_dataset(self):
        """ Retrieves dataloader from the specified class. """
        self.dataset = import_from_specified_class(self.config, 'data')

    def configure_model(self):
        """ Initializes model from the specified class. """
        if self.model is not None:
            # If the model has already been loaded in, bail.
            return

        # Get data-len from the dataloader, and add it to the model-config.
        self.config['model-config']['data-len'] = self.dataset.data_dim

        # Construct the model
        self.model = import_from_specified_class(self.config, 'model')

    def run_experiment(self):
        # Generate the trainer
        trainer = import_from_specified_class(self.config, 'trainer')
        # Need a dictionary config to init encoder:
        trainer.initialize_encoder(self.config['model-config'])

        # Save the experiment

        # Execute
        trainer.train(self.config, self.model, self.dataset)
