import logging

from multiprocessing import Process, Value
from time import sleep
from typing import Dict

import cupy as cp
import psutil

logger = logging.getLogger(__name__)

class ResourceManager:

  def __init__(self):

    self._proc = Process(target=self.watch_resources)
    self._proc.start()

    num_devices = cp.cuda.runtime.getDeviceCount()

    self._gpu_props = { "num_devices": num_devices, "devices": []}

    for dev_id in range(num_devices):

      with cp.cuda.Device(dev_id):

        tot_mem_b = cp.cuda.runtime.memGetInfo()[1]

        self._gpu_props["devices"].append({"id": dev_id,
                                          "tot_mem_B": tot_mem_b,
                                          "tot_mem_GiB": tot_mem_b / 1024.0 / 1024.0 / 1024.0,
                                          # MaxSharedMemoryPerBlock
                                          "block_smem_B": cp.cuda.runtime.deviceGetAttribute(8, dev_id),
                                          # MaxSharedMemoryPerMultiprocessor
                                          "sm_smem_B": cp.cuda.runtime.deviceGetAttribute(81, 0)})

    tot_mem = psutil.virtual_memory().total
    avbl_mem = psutil.virtual_memory().available

    self._host_props = {"physical_cores": psutil.cpu_count(logical=False),
                        "tot_mem_b": tot_mem,
                        "tot_mem_gib": tot_mem / 1024.0 / 1024.0 / 1024.0,
                        "avbl_mem_b": avbl_mem,
                        "avbl_mem_gib": avbl_mem / 1024.0 / 1024.0 / 1024.0}

  def watch_resources(self):

    logger.setLevel("DEBUG")
    cl_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s, RES MGR, %(levelname)s: %(message)s",
                                  datefmt="%a %Y-%m-%d %H:%M:%S")
    cl_handler.setLevel("DEBUG")
    cl_handler.setFormatter(formatter)
    logger.addHandler(cl_handler)

    try:

      while True:

        meminfo = cp.cuda.runtime.memGetInfo()
        logger.debug("Available GPU memory: %.2fGiB", meminfo[0] / 1024.0 / 1024.0 / 1024.0)

        sleep(2)
    # An super simple CTRL+C handler
    except KeyboardInterrupt:
      logger.info("Resource manager shutting down")

  def join(self):
    self._proc.join()

manager = ResourceManager()