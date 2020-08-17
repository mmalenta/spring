import logging
import module.computemodule as cm

from typing import List

from module.computemodule import IqrmModule
from module.module import Module

logger = logging.getLogger(__name__)

class ModuleQueue:

    """
    Queue for pipeline modules.

    Not a queue in the FIFO sense. A wrapper around a list,
    with additional functionality added.

    Attributes:

        _queue : List[Module]
            Modules to run for post-processing
        
        _required : List[Module]
            Bare minimum modules that are required for basic
            functionality


    """

    def __init__(self, modules : List[str]):
        """
        Constructs the ModuleQueue object.

        Parameters:

            modules : List[str]
                List of optional modules to add to the module queue 

        """

        self._required = ["candmaker", "frbid"]
        self._queue = []

        modules.extend(self._required)

        for module in modules:
            # Follow the naming convention described in the
            # ComputeModule class docstring
            self._queue.append(getattr(cm, module.capitalize() + "Module")())
            self._queue.sort(key=lambda val: val.id)

    def __iter__(self):

        return self


    """

    Get the module to be run.

    """
    def __next__(self):

        try:

            x = 1

        except:

            logger.error("Could not dispatch the module")


    def add_module(self, module: Module) -> None:

        self._queue.append(module)

    def remove_module(self, module: Module) -> None:

        self._queue.remove(module)