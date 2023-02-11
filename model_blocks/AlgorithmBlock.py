#  encoder subclass that requires ReplaceDictionary, saves codes, etc.
#    see altmin-analyzer.py
import torch.nn as nn
class AlgorithmBlock(nn.Module):
    """
    Parent class of all the algorithm blocks
    """
    # think of necessary properties, e
    def __init__(self):
        super(AlgorithmBlock, self).__init__()
