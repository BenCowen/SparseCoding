#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataloader with required properties

@author: Benjamin Cowen, 22 Feb 2023
@contact: benjamin.cowen.math@gmail.com
"""
import torch.utils.data


class Dataloader(torch.utils.data.DataLoader)
    """
    This will be important when managing encoder/decoder pairs?
    """
    def __init__(self):
        super(Dataloader, self).__init__()