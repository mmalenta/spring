import logging

from os import path, scandir

logger = logging.getLogger(__name__)

class ModuleRegistry:

  """

  Available module registry.

  Discovers available modules and divides them into a tree-like structure.

  Parameters:

    None

  Attributes:

    _modules: ModuleContainer
      Container that stores all the module information.

    _base_classes: List[str]
      List of base class modules to ignore. These are not useful modules, as
      they should only be used to derive modules that do actual work and never
      be used in the code directly.

  """

  class ModuleContainer:

    """

    Basic class for storing module information.

    Implements a tree-like structure to store the information on the module
    itself and its potential children. Currently the naming convention can
    be a bit confusing as "module" is used to mean both the Python module
    and processing unit used by the pipeline. This should be changed in the future.

    Parameters:

      module_type: str
        Type of the module: "dir" for directory (pretty much meaning a Python
        package directory) or "module" for an actual module file (both meaning
        Python module and pipeline processing unit).

      name: str
        Name of the module.

    Attributes:

      _type: str
        Type of the module: "dir" for directory (pretty much meaning a Python
        package directory) or "module" for an actual module file (both meaning
        Python module and pipeline processing unit).

      _name: str
        Name of the module.

      _children: List[ModuleContainer]
        List of module children. Only useful if type of the module is "dir".
        Stores subsequent child modules, whether they are other "dir"s or
        "modules".

    """

    def __init__(self, module_type, name=None):

      self._type = module_type
      self._name = name
      if self._type == "dir":
        self._children = []
      else:
        self._children = None

    def add_child(self, module):

      """

      Adds a child to the current list of children.

      Only makes sense if the module issuing this call is of type "dir".

      Parameters:

        module: ModuleContainer
          Module to be added to the list of children.

      Returns:

        None

      Raises:

        None

      """

      self._children.append(module)

    def __getitem__(self, key):

      """

      Return a module if it is present in the module tree.

      Parameters:

        key: str
          Name of the module searched for.

      Returns:

        module: ModuleContainer
          If module is found in the module tree, it is returned to the
          calling code.

      Raises:

        KeyError: raised when no module with a given name is found in the
        module tree.

      """

      for module in self._children:
        if module._name == key:
          return module

      raise KeyError

    def __contains__(self, key):

      """

      Check whether module is present in the module tree.

      Parameters:

        key: str
          Name of the module searched for.

      Returns

        : bool
          True or False depending on whether the module is found.

      """

      for module in self._children:
        if module._name == key:
          return True
      return False

  def __init__(self):

    self._modules = self.ModuleContainer("dir")
    # Base classes will be ignored
    self._base_classes = ["module", "transformmodule",
                          "utilitymodule", "inputmodule", "outputmodule"]

  def discover_modules(self):

    """

    Find available pipeline modules.

    Just a "public" wrapper around another method.

    Parameters:

      None

    Returns:

      None

    """

    self._scan_module_dir(path.join("spring/spmodule"), self._modules)

  def _scan_module_dir(self, module_dir, parent_module):

    """

    Find available pipeline modules.

    Scands the pipeline modules directory and puts available modules
    into the module tree container. Called recursively on every modules
    directory/subdirectory. Module directories and actual modules added
    with the correct type to the module tree.

    Parameters:

      dir: str
        Directory to scan for new modules

      parent_module: ModuleContainer
        Current parent module to attach children to (if any)

    Returns:

      None

    """

    submodules = [module for module in scandir(module_dir)
                  if not module.name.startswith("__")
                  and (module.name.endswith(".py") or module.is_dir())]

    for module in submodules:

      if module.is_dir():

        tmp_module = self.ModuleContainer("dir", module.name)
        self._scan_module_dir(path.join(module_dir, module.name), tmp_module)
        parent_module.add_child(tmp_module)

      else:
        if module.name.endswith(".py"):
          tmp_name = module.name[:-3]
          # Ignore base classes
          if tmp_name not in self._base_classes:
            # -6 strips the "module"
            parent_module.add_child(self.ModuleContainer("module", tmp_name[:-6]))

  def available_modules(self):

    """

    Return modules currently stored in the module tree.

    As it stands, this function is not useful, as it doesn't return a
    human-friendly output.

    Parameters:

      None

    Returns:

      : ModuleContainer
        Modules tree

    """

    return self._modules

  def print_modules(self):

    """

    Pretty-print available modules.

    Parameters:

      None

    Returns:

      None

    """
    print("Available modules: ")
    def _print_modules(parent_module, sep=""):

      for module in parent_module._children:

        if module._type == "dir":
          print(sep + "D " + module._name)
          _print_modules(module, sep+"  ")
        else:
          print(sep + "M " + module._name)

    _print_modules(self._modules)

  def __getitem__(self, key):

    """

    Return a module if it is one of the uppermost modules in the module tree.

    This is usef only if the calling code is trying to get an uppermost
    module. As ModuleContainer is returned, correct chaining of keys is
    possible, as in a["b"]["c"]: a["b"] will call this method and return
    an instance of ModuleContainer, ["c"] will call __getitem__ method
    of the ModuleContaienr class.

    Parameters:
      key: str
        Name of the module searched for.

    Returns:

      module: ModuleContainer
      If module is found in the uppermost modules of the tree, it is
      returned to the calling code.

    """

    return self._modules[key]

  def __contains__(self, key):

    """

    Check whether a module is one of the uppermost modules in the module tree.

    This particular method is called only if we're checking for the
    presence of a module at the root of the tree, as in "b" in a,
    where a is an instance of ModuleRegistry.
    Same arguments as for __getitem__ chaining apply: "c" in a["b"] will first
    call a __getitem__ of this class, which returns an instance of
    ModuleContainer. In turn the __contains__ method of ModuleContainer is
    called to check for the presence of module "c" within "b".

    Parameters:

      key: str
        Name of the module searched for.

    Returns

      : bool
        True or False depending on whether the module is one of the uppermost
        modules in the module tree.

    """

    for module in self._modules._children:
      if module._name == key:
        return True

    return False
