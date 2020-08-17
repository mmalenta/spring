import logging

from module.module import Module

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

        self.id = 0
        super().__init__()


class IqrmModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 10
        logger.info("Hello from IQRM module %d" % (self.id))



class MaskModule(ComputeModule):
    
    def __init__(self):

        super().__init__()
        self.id = 20
        logger.info("Hello from mask module %d" % (self.id))


class ThresholdModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 30
        logger.info("Hello from threshold module")

class ZerodmModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 40
        logger.info("Hello from zeroDM module %d" % (self.id))

class CandmakerModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 50
        logger.info("Hello from candmaker module")


class FrbidModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 60
        logger.info("Hello from FRBID module")


class MultibeamModule(ComputeModule):

    def __init__(self):

        super().__init__()
        self.id = 70
        logger.info("Hello from multibeam module")
