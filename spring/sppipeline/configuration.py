import argparse as ap
import logging

from json import load
from os import path
from typing import Dict

logger = logging.getLogger(__name__)

class Configuration():

  def __init__(self, config: Dict):
    self._raw_config = config
    self._parsed_config = {}

  def parse_configuration(self):

    """
    
    Parses and verifies the provided configuration.

    Parameters:

      None

    Returns:

      None
    
    """

    if "config" in self._raw_config:

      self._parse_config_file()
      del self._raw_config["config"]

    if len(self._raw_config) >= 1:

      logger.warning("Configuration for \033[;1m%s\033[0m will be overwritten",
                    ", ".join(self._raw_config.keys()))
      self._parse_cl_config()

    self._verify_config()

  def _parse_config_file(self):
    
    """

    Parses the configuration JSON file.

    The configuration JSON file exists to simplify the passing of the
    initial arguments to the pipeline. Configuration file is parsed 
    first and later overwritten by the command line arguments if
    necessary.

    Parameters:

      None

    Returns:

      None

    """

    config_file = self._raw_config["config"]

    # There is actually not much parsing going on here
    # Leave it like that though, just in case we need extra
    # functionality in the future
    with open(config_file, "r") as cf:

      config_json = load(cf)

      for key, value in config_json.items():
        self._parsed_config[key] = value

  def _parse_cl_config(self):

    """
    
    Parses the command line configuration.

    Overwrites JSON configuration, if it exists. If a transform module
    is provided via command line and it exists in the JSON
    configuration, it is overwritten with an empty configuration - this
    will result in a default configuration being used. IF the module
    does not exist in the JSON configuration, it will simply be
    initialised with the default configuration
    
    """

    for key, value in self._raw_config.items():

      if key == "modules":

        for module in value:

          # Use the default module configuration for all the CL modules,
          # but only print out warning if the JSON configuration will
          # be overwritten
          if module in self._parsed_config["modules"]["transform"]:
            logger.warning("Overwriting module \033[;1m%s\033[0m "
                            "with default configuration!", module)
          self._parsed_config["modules"]["transform"][module] = {}

      else:

        self._parsed_config[key] = value

  def _verify_config(self):

    """
    
    Verifies the provided configuration.
    
    """

    valid_configuration = True

    # Validate that the correct data directory was supplied
    try:
      watch_dir = self._parsed_config["modules"]["utility"]["input"]["watcher"]
      if not path.isdir(watch_dir["base_directory"]):
        valid_configuration = False
        # We should really add an additional check for UTC directories
        # existing in the data directory
        logger.error("Invalid data directory provided!")

    except KeyError:
      valid_configuration = False
      logger.error("Invalid watcher configuration provided!")

    # We need at least one output module
    try:
      output_modules = self._parsed_config["modules"]["utility"]["output"]
      if len(output_modules) == 0:
        raise RuntimeError

      if "plot" in output_modules:
        output_modules["plot"]["modules"] = [module[0].upper() for module in \
                          self._parsed_config["modules"]["transform"].keys()]

    except (KeyError, RuntimeError):
      valid_configuration = False
      logger.error("Invalid output configuration provided")

    # We need a correct FRBID configuration - we mostly care about
    # the access to the correct model
    try:
      frbid_config = self._parsed_config["modules"]["transform"]["frbid"]
      self._verify_frbid_model(frbid_config["model"],
                                frbid_config["model_dir"])

    except (KeyError, RuntimeError):
      valid_configuration = False
      logger.error("Invalid FRBID configuration provided!")

    if not valid_configuration:
      logger.error("Invalid configuration provided! Will now quit!")
      exit()

  def _verify_frbid_model(self, model_name: str, model_dir: str):

    """

    Checks whether the provided FRB model configuration is correct and
    the selected model loads.

    Parameters:

      model_name: str
        Name of the model to be used

      mode_dir: str
        Directory where the model exists

    Returns:

      None

    Raises:

      RuntimeError:
        This exception is raised if any of the conditions is not met

    """

    if (model_name not in
      ["NET1_32_64", "NET1_64_128", "NET1_128_256", "NET2", "NET3"]):
      logger.error("Unrecognised model %s!", model_name)
      raise RuntimeError

    if not path.isdir(model_dir):
      logger.error("Model directory %s does not exist!", model_dir)
      raise RuntimeError
    else:
      if ( (not path.exists(path.join(model_dir, model_name + ".json")) ) or 
            (not path.exists(path.join(model_dir, model_name + ".h5")) )):
        logger.error("Model %s not exist!",
                      path.join(model_dir, model_name))
        raise RuntimeError

  def get_configuration(self) -> Dict:

    """
    
    Returns the parsed configuration
    
    Parameters:

      None

    Returns

      self._parsed_config : Dict

    """

    return self._parsed_config

  def print_configuration(self):

    """
    
    Print the configuration.

    Currently just prints the dictionary as it is. Will make it look
    better later.

    Parameters:

      None

    Returns:

      None
    
    """

    print(self._parsed_config)   
