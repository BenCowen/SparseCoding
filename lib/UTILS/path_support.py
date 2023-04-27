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
import lib.UTILS.custom_loss_functions as custom_losses


def import_from_specified_class(config, keyword):
    """ Generic loader which lets you specify class in YAML config file."""
    config_string = f"{keyword}-config"
    module = importlib.import_module(config[config_string]['module'])
    class_name = config[config_string]['class']
    return getattr(module, class_name)(config=config[config_string])


def generate_loss_function(config, dataloader=None):
    recon_losses = []
    for loss_name, loss_config in config['recon-loss'].items():
        if hasattr(torch.nn, loss_name):
            recon_losses.append(getattr(torch.nn, loss_name)(**config['recon-loss'][loss_name]))
        else:
            recon_losses.append(getattr(custom_losses, loss_name)(dataloader, **config['recon-loss'][loss_name]))

    if 'code-loss' in config:
        code_losses = []
        for loss_name, loss_config in config['code-loss'].items():
            if hasattr(torch.nn, loss_name):
                code_losses.append(getattr(torch.nn, loss_name)(**config['code-loss'][loss_name]))
            else:
                code_losses.append(getattr(custom_losses, loss_name)(dataloader, **config['code-loss'][loss_name]))
    else:
        code_losses = [lambda x: 0 * x.sum()]

    def loss_fcn(batch, batch_est, codes):
        return sum([loss(batch, batch_est) for loss in recon_losses] +
                   [loss(codes) for loss in code_losses])

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


def save_train_state(config, model, training_hist, optimizer, scheduler):
    if not os.path.exists(config['save-dir']):
        os.makedirs(config['save-dir'])
        shutil.copy(config['config-path'], os.path.join(config['save-dir'], 'config-backup.yml'))

    torch.save(model, os.path.join(config['save-dir'], 'model.pt'))
    torch.save(training_hist, os.path.join(config['save-dir'], 'training_record.pt'))
    torch.save(optimizer, os.path.join(config['save-dir'], 'optimizer.pt'))
    torch.save(scheduler, os.path.join(config['save-dir'], 'scheduler.pt'))
