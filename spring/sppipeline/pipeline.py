import asyncio
import logging

from os import path
from time import perf_counter, sleep
from typing import Dict

import cupy as cp

from numpy import array, float32, ones

from spmodule.sputility.watchmodule import WatchModule
from spmodule.sputility.plotmodule import PlotModule
from spmodule.sputility.archivemodule import ArchiveModule
from spqueue.computequeue import ComputeQueue
from spqueue.candidatequeue import CandidateQueue as CandQueue

logger = logging.getLogger(__name__)

class Pipeline:

  """
  Main pipeline class.

  Pipeline is a high-level object that manages the flow of candidates
  and data. At startup it initialises all the requested modules and
  puts them in the relevant processing queues.
  When processing, it is responsible for pulling new candidates from
  candidate queue and pushing them to further processing.

  Parameters:

    config : Dict
      Configuration parameters for the pipeline. Mainly used to
      initialise relevant modules.

  Attributes:

    _running: bool
      Indicates wherer the pipeline is still running. Running pipeline
      is not paused and processing the data.

    _paused: bool
      Indicates whether the pipeline is paused. Paused pipeline is not
      running and not processing any data. This state can be used
      for updating the pipeline configuration.

    _watch_module: UtilityModule
      Module responsible for watching directories and finding new
      candidates.

    _plot_module: UtilityModule
      Module responsible for creating the candidate JPG plots.

    _archive_module: UtilityModule
      Module responsible for creating the candidate HDF5 archives.

    _module_queue: List[ComputeModule]
      List of modules responsible for processing of candidates.

    _candidate_queue: CandQueue
      Queue where new candidates are push to by the Watch Module and
      are later picked up by pipeline for further processing.

    _final_queue: CandQueue
      Queue where candidates that passed all the additional processing
      stages from the _module_queue pipeline are pushed and are later
      picked up by the plot and archive modules.

  """
  def __init__(self, config: Dict):
    
    self._running = False
    self._paused = False

    self._watch_module = WatchModule(config["base_directory"],
                                        config["num_watchers"])

    # For now we keep them as separate modules
    # Might developed into full 'post' queue like with the processing
    # if we want to enable/disable different modules
    self._plot_module = PlotModule(config["plots"])
    self._archive_module = ArchiveModule(config)

    self._module_queue = ComputeQueue(config["modules"])
    self._candidate_queue = CandQueue()
    self._final_queue = CandQueue()

    logger.debug("Created queue with %d modules",
                  (len(self._module_queue)))

    self._module_queue["frbid"].set_out_queue(self._final_queue)
    cp.cuda.Device(0).use()

  async def _listen(self, reader, writer) -> None:

    """
    Listens for an incoming connection from the head node.

    Currently broken implementation. Will be used to stop the processing
    and update the pipeline parameters if relevant request is sent from
    the head node

    Parameters:

      reader:

      writer:

    Returns:

      None

    """

    try:

      data = await reader.read(100)
      message = data.decode()
      print("Received message " + message)
      self._update()

    except asyncio.CancelledError:
      logger.info("Listener quitting")


  async def _process(self, cand_queue) -> None:

    """

    Asynchronous method for processing the candidates.

    This method waits for new candidates pused by the watch module
    to the asynchronous queue. When a candidate is picked up, it is
    then passed through all the modules in the _module_queue and 
    pushed to the _final_queue by the FRBID module.

    Parameters:

      cand_queue: CandQueue
        Queue where new candidates are push to by the Watch Module and
        are later picked up by pipeline for further processing.

    Returns:

      None

    """

    while True:

      try:

        # This will be gone at some point
        metadata = {
          "known": {

          },
          "iqrm": {

          },
          "threshold": {
          },
          "zerodm": {
          },
          "mask": {
              "multiply": True,
              "mask": array([0, 1, 1, 0])
          },
          "candmaker": {
          },
          "frbid": {
            "model": "NET3",
            "threshold": 0.5,
          },
          "multibeam": {
          }
        }

        cand_data = await cand_queue.get()
        self._module_queue[0].initialise(cand_data)

        metadata["mask"]["mask"] = ones(cand_data.metadata["fil_metadata"]["nchans"]).astype(float32)

        for module in self._module_queue:
          if await module.process(metadata[(module.__class__.__name__[:-6]).lower()]) is not None:
            break

        logger.debug("Candidate finished processing %.4fs\
                     after being added to the queue",
                     perf_counter() - cand_data.time_added)

      except asyncio.CancelledError:
        logger.info("Compute modules quitting")
        return

  async def _finalise(self, final_queue) -> None:

    """

    Asynchronous method for candidate plotting and archiving.

    This method waits for new candidates to be pused to the
    queue by the FRBID module after they were processed by all the
    modules in the _module_queue.

    Parameters:

      final_queue:
        Queue where candidates that passed all the additional processing
        stages from the _module_queue pipeline are pushed and are later
        picked up by the plot and archive modules.

    Returns:

      None

    """

    while True:

      try:

        cand_data = await final_queue.get()
        logger.debug(cand_data.metadata)

        await self._plot_module.plot(cand_data)
        await self._archive_module.archive(cand_data)

      except asyncio.CancelledError:
        logger.info("Computing has been finalised")
        return


  async def run(self, loop: asyncio.AbstractEventLoop) -> None:
    """

    Start the processing.

    This methoch starts watching for incoming candidates and other
    asynchronous processing methods.

    Returns:

      None

    """
    
    self._running = True
    self._paused = False

    logger.info("Starting up processing...")

    watcher = loop.create_task(self._watch_module.watch(self._candidate_queue))
    computer = loop.create_task(self._process(self._candidate_queue))
    finaliser = loop.create_task(self._finalise(self._final_queue))
    listener = loop.create_task(asyncio.start_server(self._listen,
                                "127.0.0.1", 9999))

    await asyncio.gather(listener, watcher, computer, finaliser)

    logger.info("Finishing the processing...")
    loop.stop()

  def stop(self, loop: asyncio.AbstractEventLoop) -> None:
    """
    Completely stops and cleans the pipeline.

    This method should be used only when the processing script is
    to be quit completely, i.e. after an exception that cannot be 
    recovered from occurs or a SIGKILL is caught.

    Returns:

      None

    """
    
    logger.info("Stopping the pipeline")

    self._running = False
    self._paused = False

    tasks = asyncio.Task.all_tasks()
    for t in tasks:
      if t._coro.__name__ != "run":
        print(t._coro.__name__)
        t.cancel()

  def _update(self) -> None:
    """
    Update the pipeline.

    Pauses the pipeline and then updates the parameters requested
    by the user. Pipeline processing is resumed upon the update
    completion.

    Returns:

      None

    """
    
    self._pause()
    
    print("Updating")
    sleep(5)

    self._resume()

  def _add_module(self, module: str) -> None:
    """
    Add a module to the module queue.

    In-place changes the current module queue. Must not be called
    on its own, but only as a part of the update() method.

    Parameters:

      module: str
        Name of the module to add to the processing pipeline.

    Returns:

      None

    """
    
    self._module_queue.add_module(module)

  def _remove_module(self, module: str) -> None:
    """
    Remove a module to the module queue.

    In-place changes the current module queue. Must not be called
    on its own, but only as a part of the update() method.

    Parameters:

      module: str
        Name of the module to remove from the processing pipeline.

    Returns:

      None

    """
    
    self._module_queue.remove_module(module)

  def _pause(self) -> None:
    """
    Pause the processing, to be resumed later.

    First waits for the current processing (if any) to finish and
    pasues the pipeline. When paused, new candidates can be added
    to the candidate queue, but they are not dispatched for 
    processing. This method should be used when the pipeline 
    parameters, e.g. the list of optional modules, have to be 
    updated upon user's request or a recoverable exception is 
    encountered.

    Returns:

      None

    """
    
    self._paused = True
    self._running = False

  def _resume(self) -> None:
    """
    Resume the previously paused pipeline

    Returns:

      None

    """
    
    self._paused = False
    self._running = True
