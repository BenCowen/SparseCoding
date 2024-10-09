"""
Model Constructor.

@author Benjamin Cowen
@date 7 Feb 2022
@contact benjamin.cowen.math@gmail.com
"""
import torch
from lib.core.task import Task
from lib.UTILS.path_support import import_class_kwargs
from collections import OrderedDict


def get_stuff_for_kwargs(config, stuff):
    """ Put requested stuff into kwargs: """
    kwargs = {}
    if 'include-stuff' in config:
        for thing in config['include-stuff']:
            kwargs |= {thing['kwarg-name']: stuff[thing['stuff-name']]}
    return kwargs


class GenerateModelFromBlocks(Task):
    """
    Parses Optimizer and Model configuration files.
    Generates model blocks and creates optimizer associations.
    Implements custom optimizer stepping.
    """

    def __init__(self, blocks, model_name=""):
        self.model_name = model_name
        self.blocks = blocks

    def run_task(self, stuff: dict) -> dict:
        # model = OrderedDict({})
        model = []
        for block_idx, block_config in enumerate(self.blocks):
            # Get name and setup Class:
            name = block_config['name'] if 'name' in block_config else block_idx
            maker, kwargs = import_class_kwargs(block_config)
            # Optionally append objects from stuff to kwargs:
            kwargs |= get_stuff_for_kwargs(block_config, stuff)
            # Build and append the model block
            model.append(maker(**kwargs))
        # Return synced model
        # TODO: for now skipping this... problem is dont have access to block shit when wrapped in Sequential
        return {self.model_name: model[-1]}
        # return {self.model_name: torch.nn.Sequential(model)}
