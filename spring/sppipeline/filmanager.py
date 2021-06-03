import logging

from multiprocessing.managers import BaseManager
from os import path

from numpy import copy, float32, floor, fromfile, reshape

logger = logging.getLogger(__name__)

class FilManager(BaseManager):

  """

  Custom multiprocessing manager.

  Does nothing beyond simple BaseManager extension.

  """

  pass

class FilDataTable:
  
  """

  Class for storing the filterbank data

  It should be registered as a new data type with the multiprocessing
  manager. Its state will be shared by multiple processes - pipeline
  process for adding the initial data and plotting/archiving process
  for removing the data if necessary.
  
  This class wraps a dictionary and is used to store the raw filterbank
  data currently being processed. This helps to avoid the need to
  constantly read the filterbank file for every candidate. With every
  new candidate using the same filterbank file, the reference counter
  for the file is increased.

  The data in this table will be stored until the candidate clustering
  results are received by the archiver. If the candidate is marked as
  a valid candidate, the filterbank data is saved in the archive and
  the reference counter is decreased, otherwise, only the reference
  counter decrease occurs. If the reference counter reaches 0, i.e.
  all the previously added candidates were processed successfully, the
  data is removed from the table.

  To prevent the table from filling the entire available RAM, a
  user-specified hard limit is used. If this limit is reached, the data
  is no longer added to this table, and every candidate will have the
  full filterbank file read separately. As the pipeline progresses
  serially through every candidate, in such a scenario, only a single
  candidate filterbank file will be in memory at any given time.

  Parameters:

    size_limit: float [GB], default 20.0
      Size limit of the data table. If exceeded, candidates are no
      longer added to it and have to be processed using a slower
      approach of reading the filterbank file separately for every
      candidate.

  Attributes:

    _data: Dict
      Filterbank data table. Each entry is identified by the filterbank
      file name and includes the actual data and the reference counter.

    _size_limit: float [GB], default 20.0
      Size limit of the data table. If exceeded, candidates are no
      longer added to it and have to be processed using a slower
      approach of reading the filterbank file separately for every
      candidate.

  """

  def __init__(self, size_limit: float = 20.0):
    self._data = {}
    self._size_limit = size_limit

    logger.info("Filterbank data table initialised")

  def test(self):

    logger.info("Data table can be called!")

  def add_candidate(self, filterbank):
    
    """

    Add the candidate to the data table.

    If the filterbank file already exists, i.e. it was already saved
    in the table by another candidate, just increase the reference
    counter. Otherwise, create the table entry with the filterbank
    data copied in and a properly set reference counter.

    Parameters:

      filterbank: Dict
        Contains the name of the filterbank file, used as a key in the
        data table and the actual raw filterbank data.

    Returns:
    
      : NumPy Array
        Array containing the raw filterbank data.

    """

    if filterbank["fil_file"] in self._data:
      logger.info("Filterbank %s already exists in the data table", 
                  filterbank["fil_file"])
      self._data[filterbank["fil_file"]]["ref_counter"] += 1
    else:
      logger.info("Reading filterbank %s into the data table", 
                  filterbank["fil_file"])

      # Read the data now - only if the data is actually going to be
      # processed by the subsequent stages
      file_path = path.join(filterbank["full_dir"],
                            filterbank["fil_file"])      

      fil_data = fromfile(file_path, dtype='B')[filterbank["header_size"]:]
      # Just in case we do not have a full filterbank written
      # This can happen when pipeline is stopped during the write
      nchans = filterbank["nchans"]
      time_samples = int(floor(fil_data.size / nchans))
      fil_data = reshape(fil_data[:(time_samples * nchans)],
                        (time_samples, nchans)).astype(float32).T

      self._data[filterbank["fil_file"]] = {"data": fil_data,
                                        "ref_counter": 1}
      
    return self._data[filterbank["fil_file"]]["data"]

  def remove_candidate(self, filterbank):

    if self._data[filterbank["name"]]["ref_counter"] == 1:
      self._data[filterbank["name"]] = None
      del self._data[filterbank["name"]]

FilManager.register("FilData", FilDataTable)