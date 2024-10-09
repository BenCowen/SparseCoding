"""
Abstract class for `task`

@author Benjamin Cowen
@date 3 April 2024
@contact benjamin.cowen.math@gmail.com
"""
import importlib

from lib.core.task import Task


class ReturnObject(Task):
    """
    This is a simple task that just initializes and object
      (such as a dataloader) and places it into backbone.stuff.
    """

    def __init__(self,
                 obj_name: str,
                 obj_class: str,
                 obj_module: str,
                 obj_config: dict):
        self.obj_init = getattr(importlib.import_module(obj_module),
                                obj_class)
        self.obj_config = obj_config
        self.obj_name = obj_name

    def run_task(self, stuff: dict) -> dict:
        # Doesn't use any Backbone information:
        return {self.obj_name: self.obj_init(self.obj_config)}
