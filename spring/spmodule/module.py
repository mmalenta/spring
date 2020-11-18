import asyncio
import logging

from typing import Dict

logger = logging.getLogger(__name__)

class Module:

  """
  Parent class for all the modules.
  
  This class should not be used explicitly in the code.

  """
  def __init__(self):
    dummy = 1

  async def process(self, metadata: Dict) -> None:

    """

    Abstract method.

    Does nothing

    Parameters:
        metadata : Dict
            Relevant metadata for processing, i.e. DM to dedisperse
            the data to, channel mask, etc. Depends on the module
            that is currently processing the data

    """
    await asyncio.sleep(2)
    