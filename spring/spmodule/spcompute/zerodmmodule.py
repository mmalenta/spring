import logging

from time import perf_counter
from typing import Dict

from numpy import mean

from spmodule.spcompute.computemodule import ComputeModule

logger = logging.getLogger(__name__)

class ZerodmModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 40
    logger.info("ZeroDM module initialised")


  async def process(self, metadata : Dict) -> None:

    """"

    Start the zeroDM processing

    """

    logger.debug("ZeroDM module starting processing")
    zerodm_start = perf_counter()
    self._data.data = self._data.data - mean(self._data.data, axis=0)
    zerodm_end = perf_counter()
    logger.debug("ZeroDM module finished processing in %.4fs",
                 zerodm_end - zerodm_start)