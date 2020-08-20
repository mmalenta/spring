import asyncio
import logging

from glob import glob
from os import path
from time import mktime, strptime

from spmodule.module import Module

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

        self._start_limit_hour = 24 * 11

        logger.info("Watcher initialised")
        logger.info("Will watch %d directories in %s" %
                        (self._max_watch, self._base_directory))
    
    async def watch(self):

        directories = sorted(glob(path.join(self._base_directory,
                                    "20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]_"
                                     + "[0-9][0-9]:[0-9][0-9]:[0-9][0-9]*/")))

        directories = directories[-1 * self._max_watch :]

        logger.info("%d directories at the start: %s, %s and %s" %
                        (self._max_watch, directories[0], directories[1],
                        directories[2]))

        # First we strip all of the directory structure to leave just
        # the UTC part. Then we convert it to time since epoch for every
        # directory in the list
        dir_times = [mktime(strptime(val[val[:-1].rfind('/')+1:-1],
                            "%Y-%m-%d_%H:%M:%S")) for val in directories]

        # Now we drop everything that is more than
        # self._start_limit_hour hours older than the newest directory

        print(dir_times)
        print(dir_times[-1])

        print([abs(val - dir_times[-1]) for val in dir_times])
        crit = [abs(val - dir_times[-1]) < self._start_limit_hour * 3600 for val in dir_times]

        print(crit)
        print(directories[crit == True])

        directories = [val[0] for val in zip(directories, dir_times)
                        if abs(val[1] - dir_times[-1]) < 
                        self._start_limit_hour * 3600]

        logger.info("Dropping %d directories due to time limit of %dh" %
                        (self._max_watch - len(directories),
                        self._start_limit_hour))

        while True:

            try:

                logger.debug("Recalculating directories...")

                await asyncio.sleep(1)
                foo = 1

            except asyncio.CancelledError:
                logger.info("Watcher quitting")
                break
            

