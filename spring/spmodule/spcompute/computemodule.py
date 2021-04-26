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


  We break the standard class naming convention here a bit.
  To create your own module, use the CamelCase naming convention,
  with the module indentifier, followed by the word 'Module'. If an 
  acronym is present in the identifier, capitalise the first letter of
  the acronym only if present at the start; if present somewhere else,
  write it all in lowercase. This is linked to how module names and
  their corresponding command-line names are processed when added to
  the processing queue.

  """
  def __init__(self):

    super().__init__()
    self._data = array([])
    self.id = 0

  def initialise(self, indata: Cand) -> None:

    self.set_input(indata)

  def set_input(self, indata: Cand) -> None:

    self._data = indata

  def get_output(self) -> Cand:

    return self._data



class MaskModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 20
    logger.info("Mask module initialised")


  async def process(self, metadata : Dict) -> None:

    """"

    Start the masking processing

    Applies the user-defined mask to the data.

    Parameters:

      metadata["mask"] : array
        Mask with the length of the number of channels in the
        data. Only values 0 and 1 are allowed. If not supplied,
        none of the channels will be masked.

      medatata["multiply"] : bool
        If true, then the mask is multiplicative, i.e. the data
        is multiplied with the values is the mask; if false, the
        mask is logical, i.e. 0 means not masking and 1 means
        masking. Defaults to True, i.e. multiplicative mask.

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

