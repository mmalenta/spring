import logging

from spmodule.sputility.utilitymodule import UtilityModule

logger = logging.getLogger(__name__)

class OutputModule(UtilityModule):

  """
  
  Parent class for all the utility modules.

  This class should not be used explicitly in the code.
  Use it as a base class for any derived utility module classes.

  """
  def __init__(self):

    super().__init__()
    logger.info("Output module")