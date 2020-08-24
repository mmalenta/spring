import logging
import spmodule.computemodule as cm

from typing import List

from spmodule.computemodule import IqrmModule
from spmodule.module import Module

logger = logging.getLogger(__name__)

class ComputeQueue:

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

    def __init__(self, modules: List[str]):
        
        """
        Constructs the ComputeQueue object.

        Parameters:

            modules : List[str]
                List of optional modules to add to the module queue 

        """

        self._required = ["candmaker", "frbid"]
        self._queue = []
        self._idx = 0

        # Just in case no pre-processing is done
        if modules == None:
            modules = self._required
        else:
            modules.extend(self._required)

        for module in modules:
            # Follow the naming convention described in the
            # ComputeModule class docstring
            self._queue.append(getattr(cm, module.capitalize() + "Module")())
            self._queue.sort(key=lambda val: val.id)

    def __iter__(self):
        
        """

        Resets the index of the list.

        If not done here, creating new iteration will start from where
        the __next__() stopped the last time.

        """

        self._idx = 0
        return self

    def __next__(self):

        """

        Get the next module in the module queue.

        As well as returning the current module to run the processing
        on, passes the data from the output of the current module to
        the input of the next module. The exception to this rule are
        the first and the last modules

            Returns:

                : Module
                    Current module to be run

            Raises:
        
                StopIteration: raised when there are no modules to
                    return. Required for the proper implementation of
                    the __next__() method

        """

        if self._idx < len(self._queue):

            if self._idx != 0:
                logger.debug("Sending the data to the next module...")
                self._queue[self._idx].set_input(self._queue[self._idx - 1].get_output())

            self._idx = self._idx + 1
            return self._queue[self._idx - 1]

        raise StopIteration

    def __contains__(self, item: str) -> bool:

        for module in self._queue:

            if isinstance(module, getattr(cm, item.capitalize() + "Module")):
                return True

        return False

    def __getitem__(self, idx: int) -> Module:

        """

        Return the module at specified index.

        Parameters:

            idx : int
                Index of the requested module

        Returns:

            : Module
                Requested module
            
        Raises:

            IndexError: raised when the index exceeds the length of the
            module queue.

        """

        if idx < len(self._queue):
            return self._queue[idx]

        raise IndexError

    def __len__(self) -> int:

        """

        Get the number of modules currently in the queue.

        Returns the length of the underlying module list

        Returns:

            : int 
            Lenght of the module queue (list)

        """

        return len(self._queue)

    def add_module(self, module: Module) -> None:

        self._queue.append(module)

    def remove_module(self, module: Module) -> None:

        self._queue.remove(module)