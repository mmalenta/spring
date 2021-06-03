import logging

from time import perf_counter
from typing import Dict

from numpy import mean

from spmodule.spcompute.computemodule import ComputeModule

logger = logging.getLogger(__name__)

class ZerodmModule(ComputeModule):

  """
  
  Module responsible for running a zeroDM RFI removal algorithm.

  Simple zeroDM removal. Calculates mean across all the channels for
  every time sample and the subtracts that mean from a given time
  sample.

  Parameters:

    config: Dict, default None
      Currently a dummy variable, not used

  Attributes:

    id: int
      Module ID used to sort the modules in the processing queue.

  """

  def __init__(self, config: Dict = None):

    super().__init__()
    self.id = 40
    self.type = "M"
    logger.info("ZeroDM module initialised")


  async def process(self, metadata : Dict) -> None:

    """"

    Run the zeroDm RFI removal.

    In-place removes (hopefully) the zeroDM RFI.

    Parameters:

      metadata: Dict
        Dictionary with all the necessary candidate information. 
        Contains the the array with the filterbank data.

    Returns:

      None

    """

    if self._data.data is None:
      self._read_filterbank()

    logger.debug("ZeroDM module starting processing")
    zerodm_start = perf_counter()
    self._data.data = self._data.data - mean(self._data.data, axis=0)
    zerodm_end = perf_counter()
    logger.debug("ZeroDM module finished processing in %.4fs",
                 zerodm_end - zerodm_start)