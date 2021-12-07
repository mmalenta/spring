import asyncio
import logging

from glob import glob
from json import load
from os import path, scandir, stat
from pandas import read_csv
from pandas.errors import EmptyDataError, ParserError
from struct import unpack
from retry.api import retry_call
from time import mktime, perf_counter, strptime, sleep
from typing import Dict, List

from spcandidate.candidate import Candidate as Cand
from spmodule.sputility.spinput.inputmodule import InputModule
from spqueue.candidatequeue import CandidateQueue as CandQueue

logger = logging.getLogger(__name__)

class WatchModule(InputModule):

  """

  Module responsible for finding new filterbank files in directories

  Using default behaviour, this module will be watching the last 'n'
  directories, where 'n' is a value provided by the user on the
  pipeline launch. As the pipeline is running, the number of
  directories being actively watched can and will change dynamically.
  Then 'n' becomes the HIGHEST number of directories to watch at a
  time.
  For performance reasons, old directories, that have not
  yielded new files for a period of time (i.e. there is a newer
  directory in front of then and they themselves have been fully
  processed) will be removed from the list and not watched.
  Ideally this module will not have to watch multiple directories
   at the same time real-time processing being the main requirement.

  Parameters:

    base_directory: str
      Base directory where the watched directories reside.

    max_watch: int, default 3
      Maximum number of directories to watch at the same time.

  Attributes:

    _base_directory: str
      Base directory where the watched directories reside.

    _directories: List[str]
      Directories to watch.

    _fil_header_size: int
      Size of the filterbank file header. Used to skip the header and
      read the data. Hardcoded value will soon be deprecated to allow
      for different filterbank files, with potentially varying
      header sizes to be processed.

    _fil_wait_sec: float
      Number of seconds that is spent on checking whether a given
      filterbank file is still being written into. If the file is still
      being written into after that time, watcher moves to another file
      and the previous file is picked up again on the next iteration.

    _max_watch: int
      Maximum number of directories to watch at the same time.

    _spccl_wait_sec: float
      Number of seconds that is spend on checkign whether a given
      .spccl file exists and has candidates that can be matched with a
      filterbank file. If the file doesn't exist or there are no
      candidates in it, it is skipped and the filterbank file list is
      reset for that particular directory.

    _start_limit_hour: int
      If there are directories older than the newest directory by this
      limit, then they are not included in the watcher list.

    _spccl_header: List[str]
      Header names used in the .sppcl file.

  """

  def __init__(self, base_directory: str, max_watch: int = 3):

    super().__init__()
    self._base_directory = base_directory
    self._max_watch = max_watch

    # Size of the filterbank header file in bytes
    self._fil_header_size = 136

    self._fil_wait_sec = 5
    self._start_limit_hour = 24
    self._spccl_wait_sec = 2
    self._spccl_header = ['MJD', 'DM', 'Width', 'SNR']

    # To convert seconds duration into MJD duration
    self._mjd_const = 1 / 86400.0

    logger.info("Watcher initialised")
    logger.info("Will watch %s directories in %s",
                self._max_watch, self._base_directory)

  def watch(self, cand_queue: CandQueue) -> None:

    """

    Watch the directories for updated .spccl and .fil files.

    This methods finds new filterbank files in watched directories and
    matches them with single-pulse canidates from the .spccl files.
    Each matched candidates is pushed to the candidate queue for
    further processing.
    This methods runs indefinitely in a loop until the pipeline
    is stopped. It starts with directories that match
    the selection criteria: at most self._max_watch directories,
    but less if the self._start_limit_hour wait limits are exceeded.
    Each beam directory scanned individually in a loop.
    The directory list is updated if necessary on each iteration.
    Async sleep at the end of every iteration to enable other work. 

    Parameters:

      cand_queue: CandQueue
        Asynchronous candidates queue for detected candidates. Each
        added candidates is later picked up by the processing pipeline.

    Returns:

      None

    """

    dirs_data = self._get_current_dirs()

    ### IMPORTANT ###
    # Due to the asynchronous nature of candidate processing .spccl
    # file will be written to before the filterbank file is saved
    # IN MOST CASES, but there is still a chance that a filterbank
    # file will be written first

    while True:

      try:

        # Loops over directories
        for data in dirs_data:
          # Loops over beams within a single directory
          for ibeam in data["logs"]:

            # Operate using relative beam numbers
            rel_number = ibeam["beam_rel"]
            full_dir = path.join(data["dir"], "beam"
                                 + "{:02}".format(rel_number))
            processed_files = data["processed_files"][rel_number]

            # Check if the directory exists
            # Should fire only if something goes really wrong
            # If it doesn't skip the processing
            # This should be done once only if directory is present
            # If it is present, it is not going anywhere

            # Do this check only when new directories are added
            if not path.isdir(full_dir):
              logger.warning("Directory %s does not exist", full_dir)
              continue

            fil_files = []
            all_files = scandir(full_dir)

            for ifile in all_files:
              if ifile.name.endswith("fil") and ifile.name not in processed_files:
                fil_files.append([ifile.name, ifile.stat().st_mtime,
                                  ifile.stat().st_size])

            logger.info("Found %d new filterbank files in %s",
                        len(fil_files), full_dir)

            # Get the newest file time per beam
            if len(fil_files) > 0:

              try:

                cands = retry_call(self._read_spccl, fargs=[full_dir],
                                    exceptions=(FileNotFoundError, 
                                                EmptyDataError,
                                                ParserError),
                                    tries=int(self._spccl_wait_sec / 0.5),
                                    delay=0.5)

              except FileNotFoundError:
                logger.error("No .spccl file under %s after %.2f seconds. "
                                "Will try again during the next iteration!",
                                full_dir,
                                self._spccl_wait_sec)
                continue

              except EmptyDataError:
                logger.error("Empty .spccl file under %s after %.2f seconds. "
                                "Will try again during the next iteration!",
                                full_dir,
                                self._spccl_wait_sec)
                continue

              except ParserError:
                logger.error("Incomplete .spccl file under %s after %.2f seconds. "
                                "Will try again during the next iteration!",
                                full_dir,
                                self._spccl_wait_sec)
                continue

              for ifile in fil_files:
                # Currently we have no better way to check
                # Check if the header is still being written to

                file_path = path.join(full_dir, ifile[0])
                with open(file_path, 'rb') as ff:

                  # Check whether we are still writing to a file
                  # This is less than ideal, as there is no guarantee
                  # that the file size will increase between getting the
                  # .fil file list and now, but it's all we have now
                  try:

                    retry_call(self._check_fil_write,
                                fargs=[ff, ifile, file_path],
                                exceptions=(RuntimeError),
                                tries=int(self._fil_wait_sec / 0.5),
                                delay=0.5)

                  except RuntimeError:

                    logger.error("File %s not complete after %.2f seconds",
                                  file_path, self._fil_wait_sec)
                    continue

                  ff.seek(0, 0)
                  header = self._read_header(ff)
                  header["fil_file"] = ifile[0]
                  header["full_dir"] = full_dir
                  header["header_size"] = self._fil_header_size

                  file_samples = int((ifile[2] - self._fil_header_size)
                                     / header["nchans"])
                  file_length_s = (file_samples * float(header["tsamp"]))
                  file_length_mjd = file_length_s * self._mjd_const
                  file_start = header["mjd"]
                  file_end = file_start + file_length_mjd
                  logger.debug("File %s spanning MJDs between %.6f and %.6f",
                               file_path, file_start, file_end)

                  matched_cands = cands[(cands["MJD"] >= file_start) &
                                        (cands["MJD"] < file_end)]

                  # Bail out early for this file
                  if len(matched_cands) == 0:
                    logger.warning("No candidates found for file %s", file_path)
                    continue

                  cand_dict = {
                      "data": None,
                      "fil_metadata": header,
                      "cand_metadata": {},
                      "beam_metadata": ibeam,
                      "time": perf_counter()
                  }

                  for _, cand in matched_cands.iterrows():

                    cand_metadata = {
                        "mjd": cand["MJD"],
                        "dm": cand["DM"],
                        "width": cand["Width"],
                        "snr": cand["SNR"],
                        "known": ""
                    }

                    cand_dict["cand_metadata"] = cand_metadata

                    cand_queue.put_candidate((0, Cand(cand_dict)))
                    logger.debug("Candidate queue size is now %d",
                                 cand_queue.qsize())

                  # Only append the file if it was properly processed
                  processed_files.add(ifile[0])

        logger.info("Recalculating directories...")
        dirs_data = self._get_current_dirs(dirs_data)

        sleep(1)
        logger.debug("Candidate queue size is now %d",
                     cand_queue.qsize())

        # End of try block

      except asyncio.CancelledError:
        logger.info("Watcher quitting")
        return
    
    # End of while loop




  def _get_current_dirs(self, current_directories: List = []) -> List:

    """

    Methods for getting new directories and their information.

    If new directories appear, the oldest ones are removed from
    the current directories list.
    If no new directories are present, the list is kept the same
    to ensure that the filterbank file and candidate watching
    stays consistent.

    Parameters:

      current_directories: List, default []
        List of dictionary that contains the information about current
        directories being watched. If the list is empty, that implies
        first run and a new list is returned that contains all the
        directories that matched our requirements.

    Returns:

      current_directories: List
        List of directories to be watched. Updated version of what was
        passed if new directories are present, otherwise the old list
        is passed through.

    """

    # Oldest directories will be at the start of this list
    directories = sorted(glob(path.join(self._base_directory, 
                         "20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_"
                         + "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*/")))

    # If we ask for more directories than there are present, we will
    # only get whatever there is
    directories = directories[-1 * self._max_watch :]

    if len(directories) == 0:
      logger.info("No directories found under %s", self._base_directory)
      # Don't bother with any extra work
      return []

    logger.info("%d directories at the start: %s",
                len(directories), ", ".join(directories))
    start_dirs = len(directories)

    if start_dirs < self._max_watch:
      logger.info("Starting with fewer directories than requested!")
      logger.info("Using %d directories instead of requested %d",
                  start_dirs, self._max_watch)

    # Strip all of the directory structure to leave just
    # the UTC part. Then convert it to time since epoch for every
    # directory in the list
    dir_times = [mktime(strptime(val[val[:-1].rfind('/')+1:-1],
                 "%Y-%m-%d_%H:%M:%S")) for val in directories]

    # Drop everything that is more than self._start_limit_hour hours
    # older than the newest directory
    directories = [val[0] for val in zip(directories, dir_times)
                   if abs(val[1] - dir_times[-1]) <
                   self._start_limit_hour * 3600]

    dropped = start_dirs - len(directories)

    # This will also fire if there are fewer directories than
    # we asked to watch
    if dropped > 0:
      logger.info(f"Dropping {dropped} "
                  + f"{'directories' if dropped > 1 else 'directory'}"
                  + f" due to the time limit of {self._start_limit_hour}h")

    tmp_current = [curr["dir"] for curr in current_directories]
    # There are some new directories that we need to include
    # in the list of current watched directories
    # Or directories were removed due to the time limit
    if (sorted(directories) != sorted(tmp_current)):

      logger.info("Directories change detected. Updating the watch list...")

      # This is not overly sophisticated way of updating the current
      # directories list, but it covers all of the age cases in the
      # least amount of code
      tmp_current_directories = []
      for idir in sorted(directories):

        if idir in tmp_current:
        
          for icurr in current_directories:
        
            if icurr["dir"] == idir:
              tmp_current_directories.append(icurr)
              break

        else:

          # Try getting the run_summary.json file for a full minute
          try:

            dir_logs = retry_call(self._read_logs, fargs=[idir],
                                  exceptions=FileNotFoundError,
                                  tries=12,
                                  delay=5,
                                  logger=logger)
            num_beams = len(dir_logs)

            tmp_current_directories.append({"dir": idir,
                    "logs": dir_logs,
                    "total_fil": 0,
                    "new_fil": 0,
                    "processed_files": [set() for _ in range(num_beams)],
                    "total_cands": [0] * num_beams,
                    "new_cands": [0] * num_beams,
                    "last_cand": [0] * num_beams})

          except FileNotFoundError:
            # TODO: Currently we just skip this directory in the current
            # watcher loop iteration. Prevent the watcher from using
            # using this directory at all - if the file doesn't appear
            # after one minute, it is highly unlikely it will appear at
            # all
            logger.error("Did not find a run_summary.json file in %s",
                          idir)

      current_directories = tmp_current_directories

    return current_directories

  def _check_fil_write(self, file_buffer, file_meta, file_path):

    """
    
    Check if the filterbank file is being written into.

    Currently does a very simple test where the size of the file is
    expected to be greater than the header file. This will catch a very
    fresh file in some cases, but it has to be improved further.

    Parameters:

      file_buffer: BufferedReader
        Open filterbank file buffer.

      file_meta: List
        Basic file information such as name modification time and size.
      
      file_path: str
        Full filterbank file path including the directory and filename.

    Returns:

      None

    """

    if ( (file_buffer.seek(0, 2) < self._fil_header_size) or 
        (stat(file_path).st_size > file_meta[2]) ):

      file_meta[2] = stat(file_path).st_size
      raise RuntimeError("File %s is smaller than the header size" % file_path)

  def _read_logs(self, directory: str, 
                 log_file: str = "run_summary.json") -> List:

    """

    Read JSON setup for current directory.

    Reads a JSON file which contains the information abotu the current
    observign run. Currently used to extract beam information.

    Parameters:

      directory: str
        Base UTC directory where the JSON file can be found. There
        should be only one such JSON file for UTC directory.
      
      log_file: str, default "run_summary.json"
        Name of the JSON file to be read.

    Returns:

      beam_info: List[Dict]
        List of dictionaries with the relevant beam information. One
        dictionary per beam.

    """

    beam_info = []

    with open(path.join(directory, log_file)) as run_file:

      run_json = load(run_file)

      for beam in run_json["beams"]["list"]:

        beam_info.append({
            "beam_abs": beam["absnum"],
            "beam_rel": beam["relnum"],
            "beam_type": 'C' if beam["coherent"] == True else 'I',
            "beam_ra": beam["ra_hms"],
            "beam_dec": beam["dec_dms"]
        })

    return beam_info

  def _read_header(self, file) -> Dict:

    """

    Read the header of the filterbank file.

    For performance reasons this is not a generic reader. It works only
    with MeerTRAP filterbank headers and will have to be changed if
    that changes as well.

    Parameters:

      file: 
        Filterbank file buffer.

    Returns:

      header : Dict
        Dictionary with relevant header values.

    """

    file.seek(24, 0)
    fch1, = unpack('d', file.read(8))
    file.seek(8, 1)
    foff, = unpack('d', file.read(8))
    # It is required to be negative
    foff = -1.0 * abs(foff)
    file.seek(23, 1)
    nchans, = unpack('i', file.read(4))
    file.seek(21, 1)
    tsamp, = unpack('d', file.read(8))
    file.seek(10, 1)
    tstart, = unpack('d', file.read(8))
    file.seek(0, 0)

    header = {
        "fch1": fch1,
        "foff": foff,
        "nchans": nchans,
        "tsamp": tsamp,
        "mjd": tstart
    }

    return header

  def _read_spccl(self, beam_dir):

    """
    
    Read the candidate .spccl file into pandas DataFrame.

    Parameters:

      beam_dir: str
        Beam directory where the candidate .spccl file is expected
        to be.
    
    Raises:

      FileNotFoundError:
        When candidate .spccl file is not present.

      EmptyDataError:
        When there are no candidates in the .spccl file.

    Returns:

        : DataFrame
        Pandas DataFrame with all the candidates currently present in
        the .spccl file. Information includes candidate MJD, DM,
        Width (ms) and SNR.

    """

    spccl_path = path.join(beam_dir, "*.spccl")
    spccl_file = glob(spccl_path)

    if len(spccl_file) == 0:
      raise FileNotFoundError("No spccl file under %s" % beam_dir)

    # This is a weird way of doing it, but it will trigger
    # the EmptyDataError exception
    spccl_cands = read_csv(spccl_file[0], delimiter="\s+", 
                            header=None, skiprows=1)
    # Check if there are any NaNs and wait
    if spccl_cands.isnull().values.any():
      raise ParserError("Incomplete spccl file under %s" % beam_dir)
    
    spccl_cands.columns = self._spccl_header

    return spccl_cands
