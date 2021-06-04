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

    size_limit: float [GiB], default 20.0
      Size limit of the data table. If exceeded, candidates are no
      longer added to it and have to be processed using a slower
      approach of reading the filterbank file separately for every
      candidate.

  Attributes:

    _current_size: int [B]
      Current size of the filterbank arrays in the data table. Not
      counting exactly everything, but just the most 'expensive' parts,
      i.e. the raw filterbank file data. Calculated simply by summing
      the sizes of all the data arrays currently in the data table.
      This value is updated every time new file is added to or removed
      from the data table.

    _data: Dict
      Filterbank data table. Each entry is identified by the filterbank
      file name and includes the filterbank header file, properly
      reshaped and transposed raw data , and the reference counter.

    _size_limit: int [B]
      Size limit of the data table. If exceeded, candidates are no
      longer added to it and have to be processed using a slower
      approach of reading the filterbank file separately for every
      candidate.

  """

  def __init__(self, size_limit: float = 20.0):
    
    self._current_size = int(0)
    self._data = {}
    # Convert from GiB to B for int comparison
    self._size_limit = int(size_limit * 1024 * 1024 * 1024)

    logger.info("Filterbank data table initialised")

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
      return self._data[filterbank["fil_file"]]["data"]

    else:

      logger.info("Reading filterbank %s into the data table", 
                  filterbank["fil_file"])

      fil_data = self._read_filterbank(filterbank["full_dir"],
                                        filterbank["fil_file"])
      
      fil_header = copy(fil_data[:filterbank["header_size"]])
      fil_data = fil_data[filterbank["header_size"]:]
      # Just in case we do not have a full filterbank written
      # This can happen when pipeline is stopped during the write
      nchans = filterbank["nchans"]
      time_samples = int(floor(fil_data.size / nchans))
      fil_data = reshape(fil_data[:(time_samples * nchans)],
                        (time_samples, nchans)).astype(float32).T

      if self._current_size + fil_data.size <= self._size_limit:
        self._data[filterbank["fil_file"]] = {"header": fil_header,
                                            "data": fil_data,
                                            "ref_counter": 1}
      
        # Take just the filterbank file into account - this is the main
        # contribution to the data table size
        self._current_size += fil_data.size
        logger.info("Current data table size: %dB/%.2fMiB",
                      self._current_size,
                      self._current_size / 1024.0 / 1024.0)

        return self._data[filterbank["fil_file"]]["data"]

      else:

        logger.warning("Exceeded the allowed size of the data table!")
        logger.warning("Filterbank %s will not be put in the data table!",
                        filterbank["fil_file"])
        # Don't put any entries in the data table
        # Just return the data - every candidate will have to read the
        # data, same goes for the archive
        return fil_data

  def remove_candidate(self, filterbank):

    """

    Remove the candidate from the data table.

    Will be used exclusively by the plotting and archiving modules at
    the very end of the processing chain.
    If the filterbank file exists, decrease the  reference counter and
    return the properly shaped data.
    If asked to decrease the counter by the very last candidate using
    given filterbank file, remove the data from the data table.
    If the filterbank file doesn't exist in the data table (i.e. the
    data table reached its maximum allowed size when the candidate was
    added), read the data and return it.

    Parameters:

      filterbank: Dict
        Contains the name of the filterbank file, used as a key in the
        data table and the actual raw filterbank data.

    Returns:
    
      : Dict
        Dictionary containing the raw filterbank data and header.

    """

    if filterbank["fil_file"] in self._data:

      logger.info("Filterbank %s in the data table", 
                  filterbank["fil_file"])

      if self._data[filterbank["fil_file"]]["ref_counter"] == 1:
        logger.info("Removing filterbank %s from the data table", 
                    filterbank["fil_file"])
        self._current_size -= self._data[filterbank["fil_file"]]["data"].size
        # Copy the data before we actually remove it
        fil_data = {"header": self._data[filterbank["fil_file"]]["header"],
                    "data": self._data[filterbank["fil_file"]]["data"]}

        self._data[filterbank["fil_file"]] = None
        del self._data[filterbank["fil_file"]]

        logger.info("Current data table size: %dB/%.2fMiB",
                      self._current_size,
                      self._current_size / 1024.0 / 1024.0)
        return fil_data

      else:

        self._data[filterbank["fil_file"]]["ref_counter"] -= 1
        # This flattens the array to the format we want it to be
        return {"header": self._data[filterbank["fil_file"]]["header"],
                    "data": self._data[filterbank["fil_file"]]["data"]}

    else:

      logger.info("Filterbank %s not in the data table", 
                  filterbank["fil_file"])
      fil_data = self._read_filterbank(filterbank["full_dir"],
                                    filterbank["fil_file"])

      fil_header = copy(fil_data[:filterbank["header_size"]])
      fil_data = fil_data[filterbank["header_size"]:]
      # Just in case we do not have a full filterbank written
      # This can happen when pipeline is stopped during the write
      nchans = filterbank["nchans"]
      time_samples = int(floor(fil_data.size / nchans))
      fil_data = reshape(fil_data[:(time_samples * nchans)],
                        (time_samples, nchans)).astype(float32).T

      return {"header": fil_header,
              "data": fil_data}                

  def _read_filterbank(self, fil_dir: str, fil_file: str):

    """

    Reads and returns the full filterbank file.

    Parameters:

      fil_dir: str
        The directory where the filterbank file resides

      fil_file: str
        The name of the filterbank file

    Returns:

      : NumPy Array
        Full filterbank file in its original format: header + originally
        shaped data.

    """

    file_path = path.join(fil_dir, fil_file)
    return fromfile(file_path, dtype='B')

FilManager.register("FilData", FilDataTable)