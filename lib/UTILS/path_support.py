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


def setup_optimizer(config, model):
    optimizer = getattr(torch.optim, config['class'])(model.parameters(),
                                                      **config['kwargs'])
    if 'scheduler-config' in config:
        sch_config = config['scheduler-config']
        scheduler = getattr(torch.optim.lr_scheduler,
                            sch_config['class'])(optimizer,
                                                 sch_config['step-size'],
                                                 **sch_config['kwargs'])
    return optimizer, scheduler


def models_to_GPU(model_list):
    for idx in range(len(model_list)):
        model_list[idx] = torch.nn.DataParallel(model_list[idx]).cuda()
    return model_list

