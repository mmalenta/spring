import asyncio
import functools
import logging
import signal

from numpy import array, ones, random
from time import perf_counter, sleep
from typing import Dict

from spmodule.utilitymodule import WatchModule
from spqueue.computequeue import ComputeQueue
from spqueue.candidatequeue import CandidateQueue as CandQueue

logger = logging.getLogger(__name__)

class Pipeline:

  """
  Main pipeline class.

  Pipeline is a high-level object that manages the flow of candidates
  and data.

  """
  def __init__(self, config: Dict):
    """
    Constructs the Pipeline object.

    After the construction is done, we end up with pipeline ready 
    for processing.

    Parameters:

        config : Dict


    """
    
    self._running = False
    self._paused = False

    self._watch_module = WatchModule(config["base_directory"],
                                        config["num_watchers"])
    self._module_queue = ComputeQueue(config["modules"])
    self._candidate_queue = CandQueue()

    logger.debug("Create queue with %d modules" %
                  (len(self._module_queue)))

  async def _listen(self, reader, writer) -> None:

    """
    Listens for an incoming connection from the head node.

    Currently broken implementation. Will be used to stop the processing
    and update the pipeline parameters if relevant request is sent from
    the head node

    """

    try:

      data = await reader.read(100)
      message = data.decode()
      print("Received message " + message)
      self._update()

    except asyncio.CancelledError:
      logger.info("Listener quitting")


  async def _process(self, cand_queue) -> None:

    while True:

      try:

        # This will be gone at some point
        metadata = {
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
          },
          "multibeam": {
          }
        }

        cand_data = await cand_queue.get()
        logger.debug(cand_data._metadata)
        self._module_queue[0].initialise(cand_data)

        for module in self._module_queue:
          await module.process(metadata[(module.__class__.__name__[:-6]).lower()])

        logger.debug("Candidate finished processing "
                      + f"{(perf_counter() - cand_data._time_added):.4}s "
                      + "after being added to the queue.")
      except asyncio.CancelledError:
          logger.info("Compute modules quitting")
          return

  async def run(self, loop: asyncio.AbstractEventLoop) -> None:
    """
    Start the processing.

    This starts watching for incoming candidates.
    """
    
    self._running = True
    self._paused = False

    logger.info("Starting up processing...")

    watcher = loop.create_task(self._watch_module.watch(self._candidate_queue))
    computer = loop.create_task(self._process(self._candidate_queue))
    listener = loop.create_task(asyncio.start_server(self._listen,
                                "127.0.0.1", 9999))

    await asyncio.gather(listener, watcher, computer)

    logger.info("Finishing the processing...")
    loop.stop()

  def stop(self, loop: asyncio.AbstractEventLoop) -> None:
    """
    Completely stops and cleans the pipeline.

    This method should be used only when the processing script is
    to be quit completely, i.e. after an exception that cannot be 
    recovered from occurs or a SIGKILL is caught.
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
    """
    
    self._pause()
    
    """
    
    Update code goes here

    """
    
    print("Updating")
    sleep(5)

    self._resume()

  def _add_module(self, module: str) -> None:
    """
    Add a module to the module queue.

    In-place changes the current module queue. Must not be called
    on its own, but only as a part of the update() method.
    """
    
    self._module_queue.add_module(module)

  def _remove_module(self, module: str) -> None:
    """
    Remove a module to the module queue.

    In-place changes the current module queue. Must not be called
    on its own, but only as a part of the update() method.
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

    Called by update() method.
    """
    
    self._paused = True
    self._running = False

  def _resume(self) -> None:
    """
    Resume the previously paused pipeline

    Called by update() method.
    """
    
    self._paused = False
    self._running = True
