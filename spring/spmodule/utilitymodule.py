import asyncio
import json
import logging

from glob import glob
from json import load
from numpy import fromfile, reshape
from os import path, scandir, stat
from pandas import read_csv
from struct import unpack
from time import mktime, strptime
from typing import Dict, List

from spcandidate.candidate import Candidate as Cand
from spmodule.module import Module
from spqueue.candidatequeue import CandidateQueue as CandQueue

logger = logging.getLogger(__name__)

class UtilityModule(Module):

  """
  Parent class for all the utility modules.

  This class should not be used explicitly in the code.
  """
  def __init__(self):

    super().__init__()
    logger.info("Utility module")



class WatchModule(UtilityModule):

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
  Ideally we will not have to watch multiple directories at the same
  time as we aim to have real-time processing.

  Attributes:

      _base_directory: str
          Base directory where the watched directories reside

      _directories: List[str]
          Directories to watch.

      _max_watch: int
          Maximum number of directories to watch at the same time

      _start_limit_hour: int
          If at the start, the newest directory is more than 24h
          younger than the other _max_watch - 1 directories, the other
          directories are not included in the first run

  """

  def __init__(self, base_directory: str, max_watch: int = 3) -> None:

    super().__init__()
    self._base_directory = base_directory
    self._max_watch = max_watch

    # Size of the filterbank header file in bytes
    self._fil_header_size = 136

    self._start_limit_hour = 24
    self._spccl_wait_sec = 2
    self._spccl_header = ['MJD', 'DM', 'Width', 'SNR']

    # To convert seconds duration into MJD duration
    self._mjd_const = 1 / 86400.0

    logger.info("Watcher initialised")
    logger.info("Will watch %d directories in %s" %
                (self._max_watch, self._base_directory))
    
  async def watch(self, cand_queue: CandQueue) -> None:

    directories = sorted(glob(path.join(self._base_directory,
                          "20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_"
                          + "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*/")))

    directories = directories[-1 * self._max_watch :]

    logger.info(f"{self._max_watch} directories at the start: " 
                + ", ".join(f"{directory}" 
                for directory in directories))

    # First we strip all of the directory structure to leave just
    # the UTC part. Then we convert it to time since epoch for every
    # directory in the list
    dir_times = [mktime(strptime(val[val[:-1].rfind('/')+1:-1],
                  "%Y-%m-%d_%H:%M:%S")) for val in directories]

    # Now we drop everything that is more than
    # self._start_limit_hour hours older than the newest directory
    directories = [val[0] for val in zip(directories, dir_times)
                  if abs(val[1] - dir_times[-1]) < 
                  self._start_limit_hour * 3600]

    dropped = self._max_watch - len(directories)

    if dropped > 0:
      logger.info(f"Dropping {dropped} "
                  + f"{'directories' if dropped > 1 else 'directory'}"
                  + f" due to the time limit of {self._start_limit_hour}h")

    dirs_data = [{"dir": idir,
              "logs": self._read_logs(idir),
              "total_fil": 0,
              "new_fil": 0,
              "last_file": [0] * len(self._read_logs(idir)),
              "total_cands": [0] * len(self._read_logs(idir)),
              "new_cands": [0] * len(self._read_logs(idir)),
              "last_cand": [0] * len(self._read_logs(idir))}
              for idir in directories]

    while True:

      try:

        for data in dirs_data:
          
          last_file = data["last_file"]
          total_cands = data["total_cands"]

          for ibeam in data["logs"]:

            # Operate using relative file numbers
            rel_number = ibeam["beam_rel"]
            full_dir = path.join(data["dir"], "beam"
                                + "{:02}".format(rel_number))

            # Check if the directory exists
            # Should fire only if something goes really wrong
            # If it doesn't skip the processing
            # This should be done once only if directory is present
            # If it is present, it is not going anywhere

            # Do this check only when new directories are added
            if not path.isdir(full_dir):
              logger.warning(f"Directory {full_dir} does not exist")
              continue

            fil_files = []
            all_files = scandir(full_dir)

            for ifile in all_files:
              if (ifile.name.endswith("fil")
                  and (ifile.stat().st_mtime > last_file[rel_number])):
                fil_files.append([ifile.name, ifile.stat().st_mtime,
                                  ifile.stat().st_size])
            
            logger.info(f"Found {len(fil_files)} new filterbank files"
                        + f" in {full_dir}")

            # Get the newest file time per beam
            if len(fil_files) > 0:
              last_file[rel_number] = max(fil_files, key = lambda ff: ff[1])[1]

              # Check if the .spccl file exists
              cand_file = glob(path.join(full_dir, "*.spccl"))

              waited = 0.0

              while((len(cand_file) == 0) and waited < self._spccl_wait_sec):
                if len(cand_file) == 0:
                  logger.warning(f"No .spccl file found yet under {full_dir}."
                                  + " Waiting...")
                  await asyncio.sleep(0.1)
                  cand_file = glob(path.join(full_dir, "*.spccl"))
                  waited = waited + 0.1

              if waited >= self._spccl_wait_sec:
                logger.error("No valid .spccl file after "
                              + f"{self._spccl_wait_sec} seconds "
                              + f"under {full_dir}."
                              + " Will reset filterbank candidates")
                last_file[rel_number] = 0
                continue

              cands = read_csv(cand_file[0], delimiter='\s+', 
                                names=self._spccl_header, skiprows=1)
              
              waited = 0.0
              while ((len(cands) == 0) and waited < self._spccl_wait_sec):
                logger.warning("No candidates in .spccl file under "
                                + f"{full_dir}. Waiting...")
                await asyncio.sleep(0.1)
                cands = read_csv(cand_file[0], delimiter='\s+', 
                                  names=self._spccl_header, skiprows=1)
                waited = waited + 0.1

              if waited >= self._spccl_wait_sec:
                logger.error("Empty .spccl file after "
                              + f"{self._spccl_wait_sec} seconds "
                              + f"under {full_dir}."
                              + " Will reset filterbank candidates")
                last_file[rel_number] = 0
                continue

              for ifile in fil_files:
                # Currently we have no better way to check
                # Check if the header is still being written to

                file_path = path.join(full_dir, ifile[0])
                logger.debug(file_path)
                with open(file_path, 'rb') as ff:
                
                  # Check whether we are still writing to a file
                  # This is less than ideal, as there is no guarantee
                  # that the file size will increase between getting the
                  # .fil file list and now, but it's all we have now
                  while ( (ff.seek(0, 2) < self._fil_header_size) or 
                        (stat(file_path).st_size > ifile[2]) ):
                    logger.info(f"File {path.join(full_dir, ifile[0])} "
                                + "is still being written into")
                    
                    await asyncio.sleep(0.1)
                    # Update new size
                    ifile[2] = stat(file_path).st_size

                  ff.seek(0, 0)
                  header = self._read_header(ff)
                  
                  file_samples = int((ifile[2] - self._fil_header_size) 
                                  / header["nchans"])
                  file_length_s = (file_samples * float(header["tsamp"]))
                  file_length_mjd = file_length_s * self._mjd_const
                  file_start = header["tstart"]
                  file_end = file_start + file_length_mjd
                  logger.debug(f"File {file_path} spanning MJDs between "
                                + f"{file_start} and {file_end}")

                  matched_cands = cands[(cands["MJD"] >= file_start) &
                                          (cands["MJD"] < file_end)]

                  if len(matched_cands) == 0:
                    logger.warning(f"No candidates found for file {file_path}")

                  fil_data = fromfile(file_path, dtype='B')[self._fil_header_size:]

                  cand_dict = {
                    "data": reshape(fil_data, (-1, header["nchans"])).T,
                    "metadata": header
                  }

                  await cand_queue.put(Cand(cand_dict))
                  logger.debug(f"Candidate queue size is now {cand_queue.qsize()}")
          # Update the newest file times for all the beams
          data["last_file"] = last_file

        logger.debug("Recalculating directories...")

        await asyncio.sleep(1)
        logger.debug(f"Candidate queue size is now {cand_queue.qsize()}")

        foo = 1

      except asyncio.CancelledError:
        logger.info("Watcher quitting")
        return
      

  def _read_logs(self, directory: str, 
                  log_file: str = "run_summary.json") -> List:

    beam_info = []

    with open(path.join(directory, log_file)) as run_file:

      run_json = json.load(run_file)

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

      file : 
        Filterbank file buffer

    Returns:

      header : Dict
        Dictionary with relevant header values

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
      "tstart": tstart
    }

    return header

