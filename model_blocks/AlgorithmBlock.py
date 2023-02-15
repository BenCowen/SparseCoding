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

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True

class SparseCoder(AlgorithmBlock):
    """
    Owns an encoder and decoder.
    """
    def __init__(self):
        super(SparseCoder, self).__init__()