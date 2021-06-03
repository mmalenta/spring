import logging
import spmodule.spcompute as cm

from typing import List

from spmodule.module import Module

logger = logging.getLogger(__name__)

class ComputeQueue:

  """
  
  Queue for pipeline modules.

  Not a queue in the FIFO sense. A wrapper around a list,
  with additional functionality added.

  Parameters:

    modules : List[Tuple]
      List of optional modules and their configurations
      to be included in the module queue.

  Attributes:

    _required: List[str]
      Names of compute modules which have to be present to ensure 
      that the pipeline works correctly. This provide a minimal working
      pipeline, with no extra pre-processing steps.

    _queue: List[ComputeModule]
      Compute modules queue responsible for processing the data.

    _idx: int
      Compute module queue index of the module
      currently being processed.

  """

  def __init__(self, modules: List[str], fil_table):

    # Having FRBID in this list is a bit redundand
    # as it is a requirement to provide a valid configuration
    # for this module
    self._required = ["candmaker", "frbid"]
    self._queue = []
    self._idx = 0

    self._fil_table = fil_table

    module_names = [module[0] for module in modules]

    for module in self._required:
      if module not in module_names:
        logger.info("Adding a required %s module with an empty configuration",
                    module.capitalize())
        modules.append((module, {}))

    for module in modules:
      # Follow the naming convention described in the
      # ComputeModule class docstring
      self._queue.append(getattr(getattr(cm, module[0] + "module"), module[0].capitalize() + "Module")(module[1]))
      self._queue.sort(key=lambda val: val.id)

  def __iter__(self):

    """

    Resets the index of the list.

    If not done here, creating new iteration will start from where
    the __next__() stopped the last time.

    Parameters:

      None

    Returns:

      None

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

    Parameters:

      None

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
        logger.info("Sending the data to the next module...")
        self._queue[self._idx].set_input(self._queue[self._idx - 1].get_output())

      self._idx = self._idx + 1
      logger.info("Module %d starting processing", self._idx)
      logger.info("Module type %s", self._queue[self._idx - 1].type)

      if (self._queue[self._idx - 1].type == "M" and 
          self._queue[self._idx - 1]._data.data is None):

          logger.info("Module will now read the filterbank data")
          self._queue[self._idx - 1].read_filterbank(self._fil_table)


      return self._queue[self._idx - 1]

    raise StopIteration

  def __contains__(self, item: str) -> bool:

    for module in self._queue:
      if isinstance(module, getattr(cm, item.capitalize() + "Module")):
        return True

    return False

  def __getitem__(self, idx) -> Module:

    """

    Return the requested module.

    If the index is an integer it simply returns module at that
    position. If it is a string however, it tries to find the module
    of this type.

    Parameters:

      idx : int or str
        Index of the requested module

    Returns:

      : Module
        Requested module
        
    Raises:

      IndexError: raised when the index exceeds the length of the
      module queue.

    """
    if isinstance(idx, int):
      if idx < len(self._queue):
        return self._queue[idx]

    if isinstance(idx, str):
      for module in self._queue:
        if isinstance(module, getattr(getattr(cm, idx + "module"), idx.capitalize() + "Module")):
          print("Found module " + idx)
          return module

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

    """
    
    Dynamically add a module to the processing queue.

    Currently not properly implemented.

    Parameters:

      module: Module
        Module to be added to the processing queue.

    Returns:

      None

    """

    self._queue.append(module)

  def remove_module(self, module: Module) -> None:

    """
    
    Dynamically remove a module to the processing queue.

    Currently not properly implemented.

    Parameters:

      module: Module
        Module to be removed from the processing queue.

    Returns:

      None

    """

    self._queue.remove(module)