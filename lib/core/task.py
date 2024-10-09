"""
Abstract class for `task`

@author Benjamin Cowen
@date 3 April 2024
@contact benjamin.cowen.math@gmail.com
"""

from abc import ABC, abstractmethod


class Task(ABC):

    @abstractmethod
    def __init__(self, kwargs):
        """
        * tasks are initialized in backbone.run_task
        * kwargs are input via the config file
        """
        pass

    @abstractmethod
    def run_task(self, stuff: dict) -> dict:
        """
        * Stuff is a dictionary of the stuff that is computed and/or
            loaded in prior tasks.
        * must return a dictionary of new data products that will be
            added to the backbone's `stuff`.
        """

        return {"oops! ran abstract class": 23}

