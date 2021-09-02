import matplotlib
from numpy.core.fromnumeric import var
matplotlib.use('Agg')

import argparse as ap
import asyncio
import logging
import signal

from functools import partial
from os import path
from typing import Dict

from json import load

from sppipeline.pipeline import Pipeline
from sppipeline.configuration import Configuration

logger = logging.getLogger()

class ColouredFormatter(logging.Formatter):

  custom_format = "[%(asctime)s] [%(process)d %(processName)s] [\033[1;{0}m%(levelname)s\033[0m] [%(module)s] %(message)s"

  def format(self, record):

    colours = {
      logging.DEBUG: 30,
      logging.INFO: 32,
      15: 36,
      logging.WARNING: 33,
      logging.ERROR: 31
    }

    colour_number = colours.get(record.levelno)
    return logging.Formatter(self.custom_format.format(colour_number), datefmt="%a %Y-%m-%d %H:%M:%S").format(record)

class CandidateFilter():

  def __init__(self, level = 15):
    self._level = level
  
  def filter(self, record):
    return record.levelno == self._level

def main():

  """

  Main entrypoint to the post-processing pipeline.

  Parses the command line arguments and starts the processing pipeline.

  """

  parser = ap.ArgumentParser(description="MeerTRAP real-time post-processing \
                                          pipeline",
                              usage="%(prog)s [options]",
                              argument_default=ap.SUPPRESS,
                              epilog="For any bugs or feature requests, \
                                      please start an issue at \
                                      https://github.com/mmalenta/spring")

  parser.add_argument("--config",
                      help="JSON config file",
                      required=False,
                      type=str)

  parser.add_argument("-l", "--log_level",
                      help="Log level", 
                      required=False,
                      type=str,
                      choices=["debug", "info", "warn"])

  parser.add_argument("-d", "--base_directory",
                      help="Base data directory to watch",
                      required=False,
                      type=str)

  parser.add_argument("-w", "--watchers",
                      help="Number of UTC directories to watch for new \
                              candidate files",
                      required=False,
                      type=int)

  """
  parser.add_argument("-m", "--modules",
                      help="Transform modules to enable",
                      required=False,
                      # If used, require at least one extra module
                      nargs="+",
                      type=str,
                      choices=["known", "iqrm", "zerodm", "threshold", "mask",
                               "multibeam", "plot", "archive"])
  """

  parser.add_argument("--model", help="Model name and model directory",
                      required=False,
                      nargs=2,
                      type=str)

  # Plots selection is input in the format:
  # [plot1][size[1],[plot2][size2][:][plot3]
  # Plots separated by a comma appear in the same row
  # Their sizes are a fraction of the full row width and can run
  # between 0.0 and 1.0 (if they do not sum up to 1.0, they are scaled
  # accordingly). Sizes are optional - if not included, all the plots
  # in the row will have the same width
  # Colon is used to indicate a new row
  # Row heights are currently set to be equal
  parser.add_argument("-p", "--plots", help="Plots to enable",
                      required=False,
                      type=str)

  parser.add_argument("-c", "--channels", help="Plot output channels",
                      required=False,
                      type=int)

  parser.add_argument("-t", "--tar", help="Enable candidate tarballs",
                      required=False,
                      action="store_true")

  arguments = parser.parse_args()

  config_parser = Configuration(vars(arguments))
  config_parser.parse_configuration()
  config_parser.print_configuration()
  configuration = config_parser.get_configuration()

  """
      ## Put that as part of modules
      "plots": {
          "plots": plots,
          "out_chans": arguments.channels,
          "modules": modules_abbr,
      }
  """

  logging.addLevelName(15, "CANDIDATE")
  logger.setLevel(getattr(logging, configuration["log_level"].upper()))
  # Might set a separate file handler for warning messages
  cl_handler = logging.StreamHandler()
  cl_handler.setLevel(getattr(logging, configuration["log_level"].upper()))
  cl_handler.setFormatter(ColouredFormatter())
  logger.addHandler(cl_handler)
  

  fl_handler = logging.FileHandler(path.join(configuration["base_directory"], "candidates.dat"))
  formatter = logging.Formatter("%(asctime)s: %(message)s",
                                datefmt="%a %Y-%m-%d %H:%M:%S")
  fl_handler.setFormatter(formatter)
  fl_handler.addFilter(CandidateFilter())
  logger.addHandler(fl_handler)

  # Separate into list of lists of tuples (one inner list per row)
  """
  plots = [ [(cell[0], float(cell[1:])) for cell in row.split(",")]
            for row in arguments.plots.split(":") ]

  modules_abbr = [module[0].upper() for module in arguments.modules]
  """

  pipeline = Pipeline(configuration)
  loop = asyncio.get_event_loop()
  # Handle CTRL + C
  loop.add_signal_handler(getattr(signal, 'SIGINT'),
                          partial(pipeline.stop, loop))    
  loop.create_task(pipeline.run(loop))

  try:
    loop.run_forever()
    logger.info("Pipeline finished processing")
  finally:
    loop.close()
    logger.info("Processing closed successfully")

if __name__ == "__main__":
  main()
