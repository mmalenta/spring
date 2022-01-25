import logging
import spmodule.sptransform as tm

from typing import Dict, List

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

    _queue: List[TransformModule]
      Compute modules queue responsible for processing the data.

    _idx: int
      Compute module queue index of the module
      currently being processed.

  """

  def __init__(self, modules: Dict, fil_table):

    # Having FRBID in this list is a bit redundand
    # as it is a requirement to provide a valid configuration
    # for this module
    self._required = ["candmaker", "frbid"]
    self._queue = []
    self._idx = 0

    self._data_state = None
    self._fil_table = fil_table


    for module in self._required:
      if module not in modules:
        logger.info("Adding a required %s module with an empty configuration",
                    module.capitalize())
        modules[module] = {}

    for module, config in modules.items():
      # Follow the naming convention described in the
      # TransformModule class docstring
      self._queue.append(getattr(getattr(tm, module + "module"), module.capitalize() + "Module")(config))
      
    self._queue.sort(key=lambda val: val.id)

    self._first_c_idx = self._find_module_type_idx("C")
    self._first_p_idx = self._find_module_type_idx("P")

    logger.info("First Cleaning module index: %d", self._first_c_idx)
    logger.info("First Processing module index: %d", self._first_p_idx)

  def _find_module_type_idx(self, type: str) -> int:

    for imodule in range(len(self._queue)):
      if self._queue[imodule].type == type:
        return imodule

    return -1

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

      run_idx = self._idx

      logger.debug("Sending the data to the next module...")

      # First cleaning module - need to read the data
      if self._idx == self._first_c_idx:

        logger.debug("Running the first cleaning module")

        if self._idx != 0:
          self._queue[self._idx].set_input(self._queue[self._idx - 1].get_output())

        try:
          self._data_state = self._queue[self._idx].read_filterbank(self._fil_table)
        except FileNotFoundError:
          # We cannot continue without the data
          logger.error("Can no longer process the file!")
          raise StopIteration
        else:

          # Decide which module to call next, depending on the state of
          # the filterbank data in the data table.
          # If the data is marked as cleaned, that means it was processed
          # with the previous candidate and all the cleaning has been
          # applied - skip the cleaning then and proceed to processing
          if self._data_state == "clean":
            run_idx = self._first_p_idx
            # Correctly set up the data for the first module
            # This will contain the metadata and the filterbank data
            # read above
            self._queue[run_idx].set_input(self._queue[self._idx].get_output())
          # If the data was not there and we get the origina filterbank
          # fiele, proceed as normal to cleaning
          else: 
            run_idx = self._idx
            # No need to read anything here - the metadata was passed
            # and the filterbank was read into the correct module above

      else:

        # We need to update the filterbank data to its cleaned version
        if self._idx == self._first_p_idx:
          self._fil_table.update_data(self._queue[self._idx - 1].get_output())

        # Move to the next module
        if self._idx != 0:
          self._queue[run_idx].set_input(self._queue[self._idx - 1].get_output())

      self._idx = run_idx + 1
      logger.debug("Module %d starting processing", self._idx)
      logger.debug("Module type %s", self._queue[run_idx].type)

      return self._queue[run_idx]

    self._data_state = None
    raise StopIteration

  def __contains__(self, item: str) -> bool:

    for module in self._queue:
      if isinstance(module, getattr(tm, item.capitalize() + "Module")):
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
        if isinstance(module, getattr(getattr(tm, idx + "module"), idx.capitalize() + "Module")):
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