import logging

from spmodule.module import Module

logger = logging.getLogger(__name__)

class UtilityModule(Module):

  """
  
  Parent class for all the utility modules.

  This class should not be used explicitly in the code.
  Use it as a base class for any derivet utility module classes.
  """
  def __init__(self):

    super().__init__()
    logger.info("Utility module")