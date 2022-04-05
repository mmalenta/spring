import logging

from time import perf_counter
from typing import Dict

from numpy import mean

from mtcutils.core import zdot
from spmodule.sptransform.transformmodule import TransformModule

logger = logging.getLogger(__name__)

class ZdotModule(TransformModule):

  """
  
  Module responsible for running a z-dot RFI removal algorithm.

  Parameters:

    config: Dict, default None
      Currently a dummy variable, not used

  Attributes:

    id: int
      Module ID used to sort the modules in the processing queue.

  """

  id = 30
  abbr = "Zd"

  def __init__(self, config: Dict = None):

    super().__init__()
    self.id = 30
    self.type = "C"
    logger.info("Z-dot module initialised")


  async def process(self) -> None:

    """"

    Run the z-dot RFI removal.

    In-place removes RFI.

    Parameters:

      None

    Returns:

      None

    """

    if self._data.data is None:
      self._read_filterbank()

    logger.debug("Z-dot module starting processing")
    zerodm_start = perf_counter()
    self._data.data = zdot(self._data.data)
    zerodm_end = perf_counter()
    logger.debug("Z-dot module finished processing in %.4fs",
                 zerodm_end - zerodm_start)