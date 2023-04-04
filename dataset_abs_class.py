"""
Decide whether to make this a simple function call or if it needs to be an
abstract class with methods that are worth inheriting...

@author Benjamin Cowen
@date 23 Jan 2022
@contact benjamin.cowen.math@gmail.com
"""

import os, sys
from argparse import ArgumentParser
from yaml import safe_load as yml_safeload
import importlib

# How determine module to save/load from ? given class name in config

class DATASET:
    def __init__(self, config_path, STATE = None, verbose = True):
        """
        On initialization, this constructs the specified dataset. This wrapper 
        might seem annoying but it's the price of having repeatable dataset
        separations while avoiding a middle-layer of function calls 

        """
        # Load config from .yml
        self._config = yml_safeload(config_path.read())
        
        # Import the specified dataset class
        DataloaderClass = getattr(importlib.import_module(self._config['dataloader-module']),
                                  self.config['dataloader-class'])
        
        # This will either initialize a new dataset or load existing specs.
        dataset = DataloaderClass(self._config)
        
        # If a STATE is provided, the dataset is added to it and passed back.
        if STATE is not None:
            STATE.dataset = dataset
            
        return
        
            
    # def initialize_dataset(self):
    #     """
    #     Determine the train/validation/test splits once and for all. 
    #     Save specs in provided cache location.
    #     """
        
    # def load_dataset(self):
    #     """
    #     Loads specs from provided cache location.
    #     """
        
    # def __next__(self):
    #     """
    #     Returns batch of samples (and labels if supervised)
    #     """