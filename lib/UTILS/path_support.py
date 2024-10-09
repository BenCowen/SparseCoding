"""
Auxiliary functions / logistic helpers...

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""

import torch
import importlib


def import_class_kwargs(config):
    module_name = config['module']
    class_name = config['class']
    module = importlib.import_module(module_name)
    kwargs = config['kwargs']
    return getattr(module, class_name), kwargs

