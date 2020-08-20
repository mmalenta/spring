import asyncio
import logging

from numpy import array, logical_not, newaxis
from time import sleep
from typing import Dict

from spmodule.module import Module

logger = logging.getLogger(__name__)

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

        self._data = array([])

        self.id = 0
        super().__init__()

    def initialise(self, indata: array) -> None:
        
        self.set_input(indata)

    def set_input(self, indata: array) -> None:

        self._data = indata

    def get_output(self) -> array:

        return self._data

class IqrmModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 10
        logger.info("IQRM module initialised")


    async def process(self, metadata : Dict) -> None:

        """"

        Start the IQRM processing

        """

        logger.debug("IQRM module starting processing")
        await asyncio.sleep(2)
        logger.debug("IQRM module finished processing")
        self._data = self._data + 1


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
        if (metadata["multiply"] == False):
            mask = logical_not(mask)

        self._data = self._data * mask[:, newaxis] 
        logger.debug("Mask module finished processing")

class ThresholdModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 30
        logger.info("Threshold module initialised")

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
        await asyncio.sleep(2)
        logger.debug("ZeroDM module finished processing")
        self._data = self._data + 1

class CandmakerModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 50
        logger.info("Candmaker module initialised")

    async def process(self, metadata : Dict) -> None:

        """"

        Start the candmaker processing

        """

        logger.debug("Candmaker module starting processing")
        await asyncio.sleep(2)
        logger.debug("Candmaker module finished processing")
        self._data = self._data + 1

class FrbidModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 60
        logger.info("FRBID module initialised")

    async def process(self, metadata : Dict) -> None:

        """"

        Start the FRBID processing

        """

        logger.debug("FRBID module starting processing")
        await asyncio.sleep(2)
        logger.debug("FRBID module finished processing")
        self._data = self._data + 1

class MultibeamModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 70
        logger.info("Multibeam module initialised")

    async def process(self, metadata : Dict) -> None:

        """"

        Start the multibeam processing

        """

        logger.debug("Multibeam module starting processing")
        await asyncio.sleep(2)
        logger.debug("Multibeam module finished processing")
        self._data = self._data + 1
