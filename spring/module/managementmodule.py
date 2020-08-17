import logging

from module.module import Module

logger = logging.getLogger(__name__)

class ManagementModule(Module):

    """
    Parent class for all the management modules.

    This class should not be used explicitly in the code.
    """
    def __init__(self):

        super().__init__()
        logger.info("Management module")