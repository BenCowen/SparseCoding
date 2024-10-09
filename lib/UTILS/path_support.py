"""
Auxiliary functions / logistic helpers...

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""

import os
import shutil
import torch
import importlib
import lib.trainers.custom_loss_functions as custom_losses


def import_from_specified_class(config, keyword):
    """ Generic loader which lets you specify class in YAML config file."""
    config_string = f"{keyword}-config"
    module = importlib.import_module(config[config_string]['module'])
    class_name = config[config_string]['class']
    return getattr(module, class_name)(config=config[config_string])


def generate_loss_function(config, custom_inputs):

    if 'torch-loss' in config:
        recon_losses = []
        for loss_name, loss_config in config['torch-loss'].items():
            recon_losses.append(getattr(torch.nn, loss_name)(**config['torch-loss'][loss_name]))
    if 'custom-loss' in config:
        custom_loss_list = []
        for loss_name, loss_config in config['custom-loss'].items():
            custom_loss_list.append(getattr(custom_losses, loss_name)(custom_inputs, **config['custom-loss'][loss_name]))

    def loss_fcn(batch, inputs_dict):
        loss = sum([loss(batch, inputs_dict['recon']) for loss in recon_losses])
        for cust_loss in custom_loss_list:
            loss += cust_loss(batch, inputs_dict)
        return loss
    return loss_fcn


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

