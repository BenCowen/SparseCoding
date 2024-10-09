"""
Class
"""

import os
import torch
from abc import ABC
from lib.core.task import Task
from lib.UTILS.path_support import import_class_kwargs


class Trainer(Task, ABC):
    def __init__(self, **config):
        self.config = config
        self.max_epoch = config['max_epoch']
        self.batches_per_print = None
        if 'batches_per_epoch' in config:
            self.max_batches = config['batches_per_epoch']
        else:
            self.max_batches = torch.inf
        self.optimizer = self.scheduler = self.training_record = self.loss_fcn = None
        self.epoch = None
        self.bb_config = None

    def run_task(self, model, dataset, loaded_objects={}):
        """
        Train the model to fit the dataset
        """

    def save_train_state(self, model, training_hist, optimizer, scheduler):
        # Abbrev:
        sdir = self.save_dir
        torch.save(model, os.path.join(sdir, 'model.pt'))
        torch.save(training_hist, os.path.join(sdir, 'training_record.pt'))
        torch.save(optimizer, os.path.join(sdir, 'optimizer.pt'))
        torch.save(scheduler, os.path.join(sdir, 'scheduler.pt'))

    def _parse_loaded_objects(self, exp_backup):
        """ Attribute everything in the obj """
        for key, value in exp_backup.items():
            setattr(self, key, value)

    @staticmethod
    def setup_optimizer(config, model):
        optimizer = getattr(torch.optim, config['class'])(model.parameters(),
                                                          **config['kwargs'])
        if 'scheduler-config' in config:
            sch_config = config['scheduler-config']
            scheduler = getattr(torch.optim.lr_scheduler,
                                sch_config['class'])(optimizer,
                                                     sch_config['step-size'],
                                                     **sch_config['kwargs'])
        else:
            scheduler = DoNothingScheduler(optimizer)
        return optimizer, scheduler

    @staticmethod
    def models_to_GPU(model_list):
        for idx in range(len(model_list)):
            model_list[idx] = torch.nn.DataParallel(model_list[idx]).cuda()
        return model_list

    @staticmethod
    def generate_loss_function(loss_config_list):
        losses = []
        for loss_config in loss_config_list:
            loss, kwargs = import_class_kwargs(loss_config)
            losses.append(loss(**kwargs))

        def my_loss(inputs_dict):
            return sum(loss(inputs_dict) for loss in losses)

        return my_loss


class DoNothingScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(DoNothingScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Simply return the current learning rates without changing them
        return [group['lr'] for group in self.optimizer.param_groups]

