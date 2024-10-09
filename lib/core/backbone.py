"""
Config reader and wrapper for the whole system.

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""
import os
import pathlib
import shutil
from lib.core.task import Task
import torch

from lib.UTILS.path_support import import_class_kwargs


class BackBone(Task):
    """ Backbone of the pipeline, controls experiment maintenence, data, etc. """

    def __init__(self, config: dict):
        self.config = config
        self.save_dir = pathlib.Path(config['save-dir'])
        # All temporary objects are kept track of here:
        self.stuff = {'save_dir': self.save_dir}
        self.loaded_stuff = []

        # Identify compute device and share with all subconfigs
        self._device = config['device']
        self.check_defaults()

    def check_defaults(self):
        if self.save_dir.name == "ben-scratch":
            print(f"Warning: \n\tUsing deffault scratch dir:"
                  f"\n\t\t{self.save_dir}"
                  f"\n\tProbably want a custom spot.")

    @property
    def task_list(self):
        """For cleaner access"""
        return self.config['tasks']

    ############################################################################
    # Setup
    def check_for_continuation(self):
        """
        Check if a save file already exists in the specified location. If so,
          and continuation is requested, load the corresponding model etc.
        """
        previous_experiment_exists = self.save_dir.is_dir() and len(os.listdir(self.save_dir)) > 0
        if previous_experiment_exists and self.config['allow-continuation']:
            self.load()
        else:
            self.setup_results_dir()

    def load(self):
        """
        Loads in every .pt file it can find in cache_dir and stores in loaded_objects.
        """
        cache_dir = self.save_dir / "checkpoints"
        print(f"Continuing from experiment results saved in {cache_dir}")
        for filename in cache_dir.rglob("*.pt"):
            obj_name = filename.name
            self.stuff[obj_name] = torch.load(cache_dir / filename)
            self.loaded_stuff += obj_name

    def setup_results_dir(self):
        """ If results directory doesnt exist, set it up. """
        if not self.save_dir.is_dir():
            os.makedirs(self.save_dir)
            # Make a copy of the config in the results dir:
            shutil.copy(self.config['config-path'],
                        os.path.join(self.save_dir, 'config-backup.yml'))
        # TODO:this doesn't work anymore...
        # Make a copy of save_dir in every task config
        for task_config in self.task_list:
            task_config['save-dir'] = self.save_dir

    ############################################################################
    # Running tasks
    def execute(self):
        n_tasks = len(self.task_list)
        for idx, task_config in enumerate(self.task_list):
            print(f'Running task {task_config["nickname"]} ({idx+1}/{n_tasks})')
            self.run_task(idx)

    def run_task(self, task_index: int):
        """Initialize the task and run it."""
        # Get config
        task_config = self.task_list[task_index]
        # Check for overwriting:
        overwriting_allowed = self.check_for_overwrite(task_config)
        # Initialize the task
        task_class, task_kwargs = import_class_kwargs(task_config)
        taskrunner = task_class(**task_kwargs)
        # Do the task
        new_stuff = taskrunner.run_task(self.stuff)
        # Append to stuff with care
        if new_stuff:
            for key, val in new_stuff.items():
                if (key not in self.stuff) or overwriting_allowed:
                    self.stuff |= {key: val}

    @staticmethod
    def check_for_overwrite(config):
        if 'overwrite-stuff' in config:
            return config['overwrite-stuff']
        else:
            return False
