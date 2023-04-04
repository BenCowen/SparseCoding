"""
Model Constructor.

@author Benjamin Cowen
@date 7 Feb 2022
@contact benjamin.cowen.math@gmail.com
"""

import Blocks as blockLib
import torch.nn as nn
import torch.nn.functional as F

class ModelGenerator:
    """
    Parses Optimizer and Model configuration files.
    Generates model blocks and creates optimizer associations.
    Implements custom optimizer stepping.
    """
    def __init__(self, config):

        self.blocks = []
        for block in config['model']:
            self.blocks.append(getattr(blockLib, block['constructor'])(block))

