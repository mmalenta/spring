import argparse as ap
import asyncio
import logging
import signal

from functools import partial
from os import path

from spmodule.moduleregistry import ModuleRegistry
from sppipeline.pipeline import Pipeline

from sppipeline.configuration import Configuration

logger = logging.getLogger()

class ModulesAction(ap.Action):

  """

  Class for custom handling of the modules argparse command line
  option.

  Parses the string passed to the module option to enable and disable
  provided modules. The format of the option is
  -m module1[=disable],[key=value]... module2[=disable],[key=value]...
  The =disable is optional and only required if a module is to be disabled,
  when e.g. overriding  a default JSON configuration. Not providing an
  option to the module is equal to enabling the module with the default
  configuration. Comma separated key=value pairs can be used to provide
  non-default configuration parameters.

  """

  def __init__(self, option_strings, dest, nargs=None, **kwargs):
    super(ModulesAction, self).__init__(option_strings, dest, nargs, **kwargs)

    self._modules = {"transform": {}}

  def __call__(self, parser, namespace, values, option_string=None) -> None:

    # For each module provided as a space separated value
    for module in values:

      # Don't include empty strings in case something like "frbid,"
      # or "," is passed by mistake
      module_config = [s for s in module.split(',') if s]

      if len(module_config) == 1:

        module_toggle = module_config[0].split('=')
        module_name = module_toggle[0]
        self._toggle_module(module_toggle)

      else:

        module_toggle = module_config[0].split('=')
        module_name = module_toggle[0]
        self._toggle_module(module_toggle)

        if self._modules["transform"][module_name]["enabled"]:

          module_params = module_config[1:]
          params_dict = {}

          for param in module_params:

            key, value, *extras = param.split("=")
            if extras:
              print(f"Will ignore extra values for parameter {key}")

            params_dict[key] = value

          self._modules["transform"][module_name]["parameters"] = params_dict

    setattr(namespace, self.dest, self._modules)

  def _toggle_module(self, module_toggle) -> None:

    """

    Enable or disable the module.

    If =disable is provided for the module it is disabled. If not,
    then enable the module.

    Parameters:

      module_toggle: List[str]
        List containing the module name and optional configuration
        parameters.

    Returns:

      None

    """

    module_name = module_toggle[0]
    if (len(module_toggle) == 2
        and (module_toggle[1] == "disable")):
      print(f"Disabling module {module_name}")
      self._modules["transform"][module_name] = {"enabled": False}
    elif (len(module_toggle) == 1) or (module_toggle[1] == "enable"):
      print(f"Enabling module {module_name} with empty configuration")
      empty_params = {"enabled": True, "parameters": {}}
      self._modules["transform"][module_name] = empty_params
    else:
      print("Unrecognised options, "
              f"will not enable module {module_name}")

class ColouredFormatter(logging.Formatter):

  """

  Provides a custom logger formatter.

  """

  custom_format = "[%(asctime)s] [%(process)d %(processName)s] " \
                  "[\033[1;{0}m%(levelname)s\033[0m] [%(module)s] %(message)s"

  def format(self, record):

    colours = {
      logging.DEBUG: 30,
      logging.INFO: 32,
      15: 36,
      logging.WARNING: 33,
      logging.ERROR: 31
    }

    colour_number = colours.get(record.levelno)
    return logging.Formatter(self.custom_format.format(colour_number),
                              datefmt="%a %Y-%m-%d %H:%M:%S").format(record)

class CandidateFilter():

  """

  Provide a custom logger filter for candidate messages.

  """

  def __init__(self, level=15):
    self._level = level


  def filter(self, record) -> bool:
    """

    Checks whether the level of the current record is the same as the
    level of the candidate filter.

    Parameters:

      None

    Returns:

      : bool

    """
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

  parser.add_argument("-m", "--modules",
                      help="Transform modules to enable",
                      action=ModulesAction,
                      required=False,
                      # If used, require at least one extra module
                      nargs="+",
                      type=str)

  parser.add_argument("--model", help="Model name and model directory",
                      required=False,
                      nargs=2,
                      type=str)

  parser.add_argument("-c", "--channels", help="Plot output channels",
                      required=False,
                      type=int)

  parser.add_argument("-t", "--tar", help="Enable candidate tarballs",
                      required=False,
                      action="store_true")

  arguments = parser.parse_args()

  module_registry = ModuleRegistry()
  module_registry.discover_modules()
  module_registry.print_modules()

  config_parser = Configuration(vars(arguments))
  config_parser.parse_configuration(module_registry)
  config_parser.print_configuration()
  configuration = config_parser.get_configuration()

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
