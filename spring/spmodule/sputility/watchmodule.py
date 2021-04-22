import asyncio
import logging

from astropy.coordinates import SkyCoord
from astropy.units import hourangle as ap_ha, deg as ap_deg
from glob import glob
from json import load
from numpy import floor, fromfile, reshape
from os import path, scandir, stat
from pandas import read_csv
from struct import unpack
from time import mktime, perf_counter, strptime
from typing import Dict, List

from psrmatch import Matcher

from spcandidate.candidate import Candidate as Cand
from spmodule.sputility.utilitymodule import UtilityModule
from spqueue.candidatequeue import CandidateQueue as CandQueue

logger = logging.getLogger(__name__)

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
  Ideally this module will not have to watch multiple directories
   at the same time real-time processing being the main requirement.

  Arguments:

    base_directory: str
      Base directory where the watched directories reside.

    max_watch: int, default 3
      Maximum number of directories to watch at the same time.

  Attributes:

    _base_directory: str
      Base directory where the watched directories reside.

    _directories: List[str]
      Directories to watch.

    _fil_wait_sec: float
      Number of seconds that is spent on checking whether a given
      filterbank file is still being written into. If the file is still
      being written into after that time, watcher moves to another file
      and the previous file is picked up again on the next iteration.

    _matcher: Matcher
      Known source matcher

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

    _spccl_header: List
      Header names used in the .sppcl file.

  """

  def __init__(self, base_directory: str, max_watch: int = 3):

    super().__init__()
    self._base_directory = base_directory
    self._max_watch = max_watch

    # Size of the filterbank header file in bytes
    self._fil_header_size = 136

    self._matcher = Matcher()
    self._matcher.load_catalogue("psrcat")
    self._matcher.create_search_tree()

    self._fil_wait_sec = 5
    self._start_limit_hour = 24
    self._spccl_wait_sec = 2
    self._spccl_header = ['MJD', 'DM', 'Width', 'SNR']

    # To convert seconds duration into MJD duration
    self._mjd_const = 1 / 86400.0

    logger.info("Watcher initialised")
    logger.info("Will watch %s directories in %s",
                self._max_watch, self._base_directory)

  async def watch(self, cand_queue: CandQueue) -> None:

    dirs_data = self._get_current_dirs()

    while True:

      try:

        for data in dirs_data:

          last_file = data["last_file"]

          for ibeam in data["logs"]:

            # Operate using relative beam numbers
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
              logger.warning("Directory %s does not exist", full_dir)
              continue

            fil_files = []
            all_files = scandir(full_dir)

            for ifile in all_files:
              if (ifile.name.endswith("fil")
                  and (ifile.stat().st_mtime > last_file[rel_number])):
                fil_files.append([ifile.name, ifile.stat().st_mtime,
                                  ifile.stat().st_size])

            logger.info("Found %d new filterbank files in %s",
                        len(fil_files), full_dir)

            # Get the newest file time per beam
            if len(fil_files) > 0:
              last_file[rel_number] = max(fil_files, key = lambda ff: ff[1])[1]

              # Check if the .spccl file exists
              cand_file = glob(path.join(full_dir, "*.spccl"))

              waited = 0.0

              while((len(cand_file) == 0) and waited < self._spccl_wait_sec):
                if len(cand_file) == 0:
                  logger.warning("No .spccl file found yet under %s. \
                                 Waiting...", full_dir)
                  await asyncio.sleep(0.1)
                  cand_file = glob(path.join(full_dir, "*.spccl"))
                  waited = waited + 0.1

              if waited >= self._spccl_wait_sec:
                logger.error("No valid .spccl file after %.2f seconds \
                             under %s. Will reset filterbank candidates",
                             self._spccl_wait_sec, full_dir)
                last_file[rel_number] = 0
                continue

              cands = read_csv(cand_file[0], delimiter="\s+", 
                               names=self._spccl_header, skiprows=1)

              waited = 0.0
              while ((len(cands) == 0) and waited < self._spccl_wait_sec):
                logger.warning("No candidates in .spccl file under %s. \
                               Waiting...", full_dir)
                await asyncio.sleep(0.1)
                cands = read_csv(cand_file[0], delimiter="\s+", 
                                 names=self._spccl_header, skiprows=1)
                waited = waited + 0.1

              if waited >= self._spccl_wait_sec:
                logger.error("Empty .spccl file after %.2f seconds \
                             under %s. Will reset filterbank candidates",
                             self._spccl_wait_sec, full_dir)
                last_file[rel_number] = 0
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
                  # TODO: Need a guard just in case it fails completely
                  # and we get a file with less than the header size
                  # saved - just have a wait time as in other cases
                  waited = 0.0
                  while (((ff.seek(0, 2) < self._fil_header_size) or 
                          (stat(file_path).st_size > ifile[2])) and
                         waited < self._fil_wait_sec):
                    logger.info("File %s is being written into", file_path)

                    await asyncio.sleep(0.1)
                    # Update new size
                    ifile[2] = stat(file_path).st_size
                    waited = waited + 0.1

                  if waited >= self._fil_wait_sec:
                    logger.error("File %s not complete after %.2f seconds",
                                 file_path, self._fil_wait_sec)
                    continue

                  ff.seek(0, 0)
                  header = self._read_header(ff)
                  header["fil_file"] = ifile[0]
                  header["full_dir"] = full_dir

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

                  if len(matched_cands) == 0:
                    logger.warning("No candidates found for file %s", file_path)

                  fil_data = fromfile(file_path, dtype='B')[self._fil_header_size:]

                  # We can have a filterbank file being written when
                  # the pipeline is stopped
                  time_samples = int(floor(fil_data.size / header["nchans"]))

                 

                  cand_dict = {
                      "data": reshape(fil_data[:(time_samples * header["nchans"])],
                                      (time_samples, header["nchans"])).T,
                      "fil_metadata": header,
                      "cand_metadata": {},
                      "beam_metadata": ibeam,
                      "time": perf_counter()
                  }

                  beam_position = SkyCoord(ra = ibeam["beam_ra"],
                                            dec = ibeam["beam_dec"],
                                            frame="icrs",
                                            unit=(ap_ha, ap_deg))

                  for candidx, cand in matched_cands.iterrows():

                    known_matches = self._matcher.find_matches(beam_position,
                                                              cand["DM"])

                    if known_matches is not None:

                      logger.info("This candidate is a known source")
                      logger.info("It will not be processed further")

                      continue

                    cand_metadata = {
                        "mjd": cand["MJD"],
                        "dm": cand["DM"],
                        "width": cand["Width"],
                        "snr": cand["SNR"]
                    }

                    cand_dict["cand_metadata"] = cand_metadata

                    await cand_queue.put(Cand(cand_dict))
                    logger.debug("Candidate queue size is now %d",
                                 cand_queue.qsize())
          # Update the newest file times for all the beams
          data["last_file"] = last_file

        logger.info("Recalculating directories...")
        dirs_data = self._get_current_dirs(dirs_data)

        await asyncio.sleep(1)
        logger.debug("Candidate queue size is now %d",
                     cand_queue.qsize())

        # End of try block

      except asyncio.CancelledError:
        logger.info("Watcher quitting")
        return
    
    # End of while loop

  def _get_current_dirs(self, current_directories: List = []) -> List:

    """

    Methods for getting new directories and their info.

    If new directories appear, the oldest ones are removed from
    the current directories list.
    If no new directories are present, the list is kept the same
    to ensure that the filterbank file and candidate watching
    stays consistent.

    Arguments:

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

          tmp_current_directories.append({"dir": idir,
                  "logs": self._read_logs(idir),
                  "total_fil": 0,
                  "new_fil": 0,
                  "last_file": [0] * len(self._read_logs(idir)),
                  "total_cands": [0] * len(self._read_logs(idir)),
                  "new_cands": [0] * len(self._read_logs(idir)),
                  "last_cand": [0] * len(self._read_logs(idir))})

      current_directories = tmp_current_directories

    return current_directories

  def _read_logs(self, directory: str, 
                 log_file: str = "run_summary.json") -> List:

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
        "mjd": tstart
    }

    return header
