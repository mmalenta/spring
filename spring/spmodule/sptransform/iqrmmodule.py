import logging

from time import perf_counter
from typing import Dict

from mtcutils.core import normalise
from mtcutils import iqrm_mask as iqrm
from spmodule.sptransform.transformmodule import TransformModule

logger = logging.getLogger(__name__)

class IqrmModule(TransformModule):

  """

  Module responsible for running IQRM on the data.

  Really just a wrapper for the call to the relevant IQRM modules.

  Parameters:

    config: Dict, default None
      Currently a dummy variable, not used

  Attributes:

    id: int
      Module ID used to sort the modules in the processing queue.

  """

  def __init__(self, config: Dict = None):

    super().__init__()
    self.id = 10
    self.type = "C"
    logger.info("IQRM module initialised")


  async def process(self) -> None:

    """"

    Start the IQRM processing.

    Runs IQRM on the data. Scales the data and applies the mask.
    Data mean and standard deviation are updated.

    Parameters:

      None

    Returns:

      None

    """

    logger.debug("IQRM module starting processing")
    iqrm_start = perf_counter()
    scaled, norm_mean, norm_stdev = normalise(self._data.data)
    # TODO: Make maxlag properly configurable
    mask = iqrm(norm_stdev, maxlag=15)
    scaled[mask] = 0
    self._data.data = scaled
    self._data.mean = norm_mean
    self._data.stdev = norm_stdev
    iqrm_end = perf_counter()
    logger.debug("IQRM module finished processing in %.4fs",
                 iqrm_end - iqrm_start)