import logging
import pika

from math import ceil
from time import perf_counter, time
from typing import Dict

import cupy as cp

from json import dumps
from numpy import append, array, clip, linspace, logical_not, mean, median
from numpy import newaxis, random, std

from FRBID_code.prediction_phase import load_candidate, FRB_prediction
from spcandidate.candidate import Candidate as Cand
from spmodule.module import Module

logger = logging.getLogger(__name__)

# Seconds in a day
DAY_SEC = 86400.0

class ComputeModule(Module):

  """

  Parent class for all the compute modules.

  This class should not be used explicitly in the code.

  We break the standard Python class naming convention here.
  To create your own module, use the CamelCase naming convention,
  with a single-word module indentifier, followed by the word 'Module'.
  If an acronym is present in the identifier, capitalise the first
  letter of the acronym only if present at the start (e.g. Rfi);
  if present somewhere else, write it all in lowercase (e.g. Zerodm).
  This is linked to how module names and their corresponding
  command-line names are processed when added to the processing queue.

  """
  def __init__(self):

    super().__init__()
    self._data = array([])
    self.id = 0

  def initialise(self, indata: Cand) -> None:

    """

    Set the data that the first module in the processing pipeline
    will work on.

    A simple wrapper around the set_input, but a different name used
    to distinguish the functionality.

    Parameters:

      indata: Cand
        All the data on the candidate being currently processed.

    Returns:

      None

    """

    self.set_input(indata)

  def set_input(self, indata: Cand) -> None:

    """

    Set the data that the given module will work on.

    Used by the compute queue to pick the output data from the
    previous module and use it as the input to the current module.

    Parameters:

      indata: Cand
        All the data on the candidate being currently processed.

    Returns:

      None

    """

    self._data = indata

  def get_output(self) -> Cand:

    """

    Provide the data that the give module finished workign on.

    Used by the compute queue to get the data from the module and pass
    it to the next one.

    Parameters:

      None

    Returns

      self._data: Cand
        All the data on the candidate being currently processed.

    """

    return self._data



class MaskModule(ComputeModule):

  """

  Module responsible for masking the data.

  Applies a static mask to the data.

  Parameters:

    None

  Attributes:

    id: int
      Module ID used to sort the modules in the processing queue.

  """

  def __init__(self):

    super().__init__()
    self.id = 20
    logger.info("Mask module initialised")


  async def process(self, metadata : Dict) -> None:

    """"

    Start the masking processing.

    Applies a static, user-defined mask to the data.

    Parameters:

      metadata["mask"] : Numpy Array
        Mask with the length of the number of channels in the
        data. Only values 0 and 1 are allowed. If not supplied,
        none of the channels will be masked.

      medatata["multiply"] : bool
        If true, then the mask is multiplicative, i.e. the data
        is multiplied with the values is the mask; if false, the
        mask is logical, i.e. 0 means not masking and 1 means
        masking. Defaults to True, i.e. multiplicative mask.

    Returns:

      None

    """
    logger.debug("Mask module starting processing")
    mask = metadata["mask"]
    # Need to revert to a multiplicative mask anyway
    if metadata["multiply"] == False:
      mask = logical_not(mask)

    self._data.data = self._data.data * mask[:, newaxis] 
    logger.debug("Mask module finished processing")

class ThresholdModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 30
    logger.info("Threshold module initialised")

