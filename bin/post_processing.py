import matplotlib
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

def check_frbid_model(model_name: str, model_dir: str) -> bool:

  """

  Checks whether the provided FRB model configuration is correct and
  the selected model loads.

  Parameters:

    model_name: str
      Name of the model to be used

    mode_dir: str
      Directory where the model exists

  Returns:

    : bool
      True if the checks are passed and the model can be loaded.
      False if the checks fail and the model cannod be loaded.

  """

  if (model_name not in
    ["NET1_32_64", "NET1_64_128", "NET1_128_256", "NET2", "NET3"]):
    logger.error("Unrecognised model %s!", model_name)
    return False
  
  if not path.isdir(model_dir):
    logger.error("Model directory %s does not exist!", model_dir)
    return False
  else:
    if ( (not path.exists(path.join(model_dir, model_name + ".json")) ) or 
          (not path.exists(path.join(model_dir, model_name + ".h5")) )):
      logger.error("Model %s not exist!",
                    path.join(model_dir, model_name))
      return False
  
  return True

def parse_config_file(current_config: Dict, config_file: str) -> Dict:

  """

  Parses the configuration JSON file.

  The configuration JSON file exists to simplify the passing of initial
  arguments to the pipeline. Configuration file takes precedence over
  arguments provided on the command line.

  Parameters:

    current_config: Dict
      Initial configuration dictionary obtained from command line
    
    config_file: str
      Path to the JSON configuration file

  Returns:

    current_config: Dict
      An updated configuration dictionary

  """

  with open(config_file, "r") as cf:

    file_config = {}
    config_json = load(cf)

    for key, val in config_json.items():

      if (key == "modules"):

        file_config["modules"] = []

        for module in config_json["modules"].items(): 

          if module[0] == "frbid":
            print(module[1])
            if ("model" not in module[1] or "model_dir" not in module[1]):
              logger.error("Did not provide a sufficient configuration for FRBID!")
              logger.error("Will quit now!")
              exit()
            
            if not check_frbid_model(module[1]["model"], module[1]["model_dir"]):
              logger.error("Did not provide a correct configuration for FRBID!")
              logger.error("Will quit now!")
              exit()

          file_config["modules"].append(module)

      else:

        file_config[key] = val

  current_config.update(file_config)
  return current_config

def main():

  """

  Main entrypoint to the post-processing pipeline.

  Parses the command line arguments and starts the processing pipeline.

  """

  parser = ap.ArgumentParser(description="MeerTRAP real-time post-processing \
                                          pipeline",
                              usage="%(prog)s [options]",
                              epilog="For any bugs or feature requests, \
                                      please start an issue at \
                                      https://github.com/mmalenta/spring")

  parser.add_argument("--config",
                      help="JSON config file",
                      required=False,
                      type=str)

  parser.add_argument("-l", "--log",
                      help="Log level", 
                      required=False,
                      type=str,
                      choices=["debug", "info", "warn"],
                      default="debug")

  parser.add_argument("-d", "--directory",
                      help="Base data directory to watch",
                      required=True,
                      type=str)

  parser.add_argument("-w", "--watchers",
                      help="Number of UTC directories to watch for new \
                              candidate files",
                      required=False,
                      type=int,
                      default=3)

  parser.add_argument("-m", "--modules",
                      help="Optional modules to enable",
                      required=False,
                      # If used, require at least one extra module
                      nargs="+",
                      type=str,
                      choices=["known", "iqrm", "zerodm", "threshold", "mask",
                               "multibeam", "plot", "archive"])

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
                      # If used, require at least one type of plot
                      type=str,
                      default="s0.75,b0.25:f0.5,t0.5")

  parser.add_argument("-c", "--channels", help="Plot output channels",
                      required=False,
                      type=int,
                      default=64)

  parser.add_argument("-t", "--tar", help="Enable candidate tarballs",
                      required=False,
                      action="store_true")

  arguments = parser.parse_args()

  logging.addLevelName(15, "CANDIDATE")
  logger.setLevel(getattr(logging, arguments.log.upper()))
  # Might set a separate file handler for warning messages
  cl_handler = logging.StreamHandler()
  cl_handler.setLevel(getattr(logging, arguments.log.upper()))
  cl_handler.setFormatter(ColouredFormatter())
  logger.addHandler(cl_handler)
  
  fl_handler = logging.FileHandler(path.join(arguments.directory, "candidates.dat"))
  formatter = logging.Formatter("%(asctime)s: %(message)s",
                                datefmt="%a %Y-%m-%d %H:%M:%S")
  fl_handler.setFormatter(formatter)
  fl_handler.addFilter(CandidateFilter())
  logger.addHandler(fl_handler)

  modules = [(module, {}) for module in arguments.modules]
  if arguments.model is not None:

    chosen_model = arguments.model[0].upper()
    chosen_dir = arguments.model[1]

    if not check_frbid_model(chosen_model, chosen_dir):
      logger.error("Did not provide a correct configuration for FRBID!")
      logger.error("Will quit now!")
      exit()

    modules.append(("frbid", {"model": chosen_model,
                              "model_dir": chosen_dir}))

  # Separate into list of lists of tuples (one inner list per row)
  plots = [ [(cell[0], float(cell[1:])) for cell in row.split(",")]
            for row in arguments.plots.split(":") ]

  modules_abbr = [module[0].upper() for module in arguments.modules]

  configuration = {
      "base_directory": arguments.directory,
      "num_watchers": arguments.watchers,
      "modules": modules,
      "model": [chosen_model, chosen_dir],
      "plots": {
          "plots": plots,
          "out_chans": arguments.channels,
          "modules": modules_abbr,
      }
  }

  if arguments.config is not None:
    logger.warning("JSON configuration file provided! \
                    Some command line options may be overwritten!")
    configuration = parse_config_file(configuration, arguments.config)

  if ("frbid" not in [module[0] for module in configuration["modules"]]):
    logger.error("Did not provide configuration for FRBID!")
    logger.error("Will quit now!")
    exit()

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
