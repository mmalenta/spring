import asyncio
import cupy as cp
import functools
import logging
import signal


from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
from numpy import array, ones, random
from os import path
from tensorflow import ConfigProto, Session
from time import perf_counter, sleep
from typing import Dict

from spmodule.utilitymodule import ArchiveModule, PlotModule, WatchModule
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

    # For now we keep them as separate modules
    # Might developed into full 'post' queue like with the processing
    # if we want to enable/disable different modules
    self._plot_module = PlotModule(config["plots"])
    self._archive_module = ArchiveModule(config)

    self._module_queue = ComputeQueue(config["modules"])
    self._candidate_queue = CandQueue()
    self._final_queue = CandQueue()

    logger.debug("Created queue with %d modules" %
                  (len(self._module_queue)))

    logger.debug("Setting up TensorFlow...")

    tf_config = ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.25
    set_session(Session(config=tf_config))
    cp.cuda.Device(0).use()

    with open(path.join(config["model"][1],
              config["model"][0] + ".json"), "r") as mf:
      model = model_from_json(mf.read())

    model.load_weights(path.join(config["model"][1],
                        config["model"][0] + ".h5"))
    # FRBID should ususally be at the end
    # Just in case there is some extra post-classification processing
    self._module_queue["frbid"].set_model(model)
    self._module_queue["frbid"].set_out_queue(self._final_queue)

    logger.debug("TensorFlow has been set up")

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
            "model": "NET3",
            "threshold": 0.5,
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

  async def _finalise(self, final_queue) -> None:

    while True:

      try:

        cand_data = await final_queue.get()
        logger.debug(cand_data._metadata)
        logger.debug(cand_data._data)
        logger.debug(cand_data._ml_cand)

        await self._plot_module.plot(cand_data)
        await self._archive_module.archive()

      except asyncio.CancelledError:
          logger.info("Computing has been finalised")
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
