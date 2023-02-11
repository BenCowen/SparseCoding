
import torch
import numpy

def set_seeds(torch_seed = 1234, numpy_seed = 1234):
    """ Set both random seeds """
    torch.manual_seed(torch_seed)
    numpy.random.seed(numpy_seed)

def reproducibility_mode():
    """ Set seeds to default value and turn on deterministic CUDA mode"""
    set_seeds()
    torch.use_deterministic_algorithms(True)
