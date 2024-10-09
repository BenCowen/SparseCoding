
import torch
import numpy

def set_seeds(seed_config = {}):
    """ Set both random seeds """
    if 'torch-seed' in seed_config:
        torch.manual_seed(seed_config['torch-seed'])
    else:
        torch.manual_seed(1234)
    if 'numpy-seed' in seed_config:
        numpy.random.seed(seed_config['numpy-seed'])
    else:
        numpy.random.seed(1234)

def reproducibility_mode():
    """ Set seeds to default value and turn on deterministic CUDA mode"""
    set_seeds()
    torch.use_deterministic_algorithms(True)

def set_resproducibility(config):
    """
    If requested, make everything as deterministic as possible.
        Default behavior is to simply set seeds to 1234.
    """
    if 'reproducibility-mode' in config:
        if config['reproducibility-mode']:
            reproducibility_mode()
    elif 'seed-config' in config:
        set_seeds(config['seed-config'])
    else:
        set_seeds()
