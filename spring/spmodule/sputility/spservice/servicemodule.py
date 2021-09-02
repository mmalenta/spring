import logging

from spmodule.sputility.utilitymodule import UtilityModule

logger = logging.getLogger(__name__)

class ServiceModule(UtilityModule):

  """
  
  Parent class for all the service modules.

  This class should not be used explicitly in the code.
  Use it as a base class for any derived service module classes.
  These modules are designed to watch available resources and the state
  of processing.

  """
  def __init__(self):

    super().__init__()
    logger.info("Utility module")