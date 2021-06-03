import logging

from os import path
from typing import Dict

from numpy import array, float32, floor, fromfile, logical_not, newaxis, reshape

from spcandidate.candidate import Candidate as Cand
from spmodule.module import Module

logger = logging.getLogger(__name__)

# Seconds in a day
DAY_SEC = 86400.0

class ComputeModule(Module):

  """

  Parent class for all the compute modules.

  This class should not be used explicitly in the code.

  We break the standard Python class naming convention here.
  To create your own module, use the CamelCase naming convention,
  with a single-word module indentifier, followed by the word 'Module'.
  If an acronym is present in the identifier, capitalise the first
  letter of the acronym only if present at the start (e.g. Rfi);
  if present somewhere else, write it all in lowercase (e.g. Zerodm).
  This is linked to how module names and their corresponding
  command-line names are processed when added to the processing queue.

  Attributes:

    _type: str
      Type of the compute module: "V" for vetting - does some
      classification on the data and does not change the underlying
      data; "M" for mutating - actually changes the data that is sent
      to it. First module marked with "M" is responsible for reading
      in the filterbank file data. This is to prevent the pipeline
      reading in the data unnecessarily from candidates that do not
      pass the initial vetting, e.g. known source matching.

  """
  def __init__(self):

    super().__init__()
    self._data = array([])
    self.type = None
    self.id = 0

  def initialise(self, indata: Cand) -> None:

    """

    Set the data that the first module in the processing pipeline
    will work on.

    A simple wrapper around the set_input, but a different name used
    to distinguish the functionality.

    Parameters:

      indata: Cand
        All the data on the candidate being currently processed.

    Returns:

      None

    """

    self.set_input(indata)

  def read_filterbank(self, fil_table) -> None:

    """

    Read the filterbank file

    Depending on the modules used, any of them can be required to be
    the first to read the actual data from the filterbank file.

    Operates on the data already present in the pipeline.

    Parameters:

      None

    Returns:

      None

    """


    fil_metadata = self._data.metadata["fil_metadata"]
    self._data.data = fil_table.add_candidate(fil_metadata)

    """
    # Read the data now - only if the data is actually going to be
    # processed by the subsequent stages
    file_path = path.join(fil_metadata["full_dir"],
                            fil_metadata["fil_file"])

                          
    fil_data = fromfile(file_path, dtype='B')[fil_metadata["header_size"]:]
    # Just in case we do not have a full filterbank written
    # This can happen when pipeline is stopped during the write
    nchans = fil_metadata["nchans"]
    time_samples = int(floor(fil_data.size / nchans))

    fil_data = reshape(fil_data[:(time_samples * nchans)],
                        (time_samples, nchans)).astype(float32).T

    self._data.data = fil_data
    """

  def set_input(self, indata: Cand) -> None:

    """

    Set the data that the given module will work on.

    Used by the compute queue to pick the output data from the
    previous module and use it as the input to the current module.

    Parameters:

      indata: Cand
        All the data on the candidate being currently processed.

    Returns:

      None

    """

    self._data = indata

  def get_output(self) -> Cand:

    """

    Provide the data that the give module finished working on.

    Used by the compute queue to get the data from the module and pass
    it to the next one.

    Parameters:

      None

    Returns

      self._data: Cand
        All the data on the candidate being currently processed.

    """

    return self._data



class MaskModule(ComputeModule):

  """

  Module responsible for masking the data.

  Applies a static mask to the data.

  Parameters:

    None

  Attributes:

    id: int
      Module ID used to sort the modules in the processing queue.

  """

  def __init__(self):

    super().__init__()
    self.id = 20
    logger.info("Mask module initialised")


  async def process(self, metadata : Dict) -> None:

    """"

    Start the masking processing.

    Applies a static, user-defined mask to the data.

    Parameters:

      metadata["mask"] : Numpy Array
        Mask with the length of the number of channels in the
        data. Only values 0 and 1 are allowed. If not supplied,
        none of the channels will be masked.

      medatata["multiply"] : bool
        If true, then the mask is multiplicative, i.e. the data
        is multiplied with the values is the mask; if false, the
        mask is logical, i.e. 0 means not masking and 1 means
        masking. Defaults to True, i.e. multiplicative mask.

    Returns:

      None

    """
    logger.debug("Mask module starting processing")
    mask = metadata["mask"]
    # Need to revert to a multiplicative mask anyway
    if metadata["multiply"] == False:
      mask = logical_not(mask)

    self._data.data = self._data.data * mask[:, newaxis] 
    logger.debug("Mask module finished processing")

class ThresholdModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 30
    logger.info("Threshold module initialised")

