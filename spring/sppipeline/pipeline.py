import asyncio
import logging

from json import loads
from multiprocessing import Process, Manager
from os import path
from socket import gethostname
from time import perf_counter, sleep
from typing import Dict

import cupy as cp
import pika

from numpy import array, float32, ones

from spmodule.sputility.spinput.watchmodule import WatchModule
from spmodule.sputility.spoutput.archivemodule import ArchiveModule
from spmodule.sputility.spoutput.plotmodule import PlotModule
from spmodule.sputility.spoutput.tarmodule import TarModule
from sppipeline.filmanager import FilManager
from spqueue.computequeue import ComputeQueue
from spqueue.candidatequeue import CandidateManager

logger = logging.getLogger(__name__)

class Pipeline:

  """

  Main pipeline class.

  Pipeline is a high-level object that manages the flow of candidates
  and data. At startup it initialises all the requested modules and
  puts them in the relevant processing queues.
  When processing, it is responsible for pulling new candidates from
  the candidate queue and pushing them to further processing.

  Parameters:

    config : Dict
      Configuration parameters for the pipeline. Mainly used to
      initialise the requested modules.

  Attributes:

    _running: bool
      Indicates whether the pipeline is still running. Running pipeline
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

    _module_queue: List[TransformModule]
      List of modules responsible for processing of candidates.

    _candidate_queue: CandQueue
      Queue where new candidates are pushed to by the Watch Module and
      are later picked up by pipeline for further processing.

    _final_queue: CandQueue
      Queue where candidates that passed all the additional processing
      stages from the _module_queue pipeline are pushed and are later
      picked up by the plot and archive modules.

    _manager: FilManager
      Filterbank data table manager. Use to properly share the data
      between compute and plotting/archiving processes.

    _fil_table: FilDataTable
      The underlying class that manager shares between the processes.

  """
  def __init__(self, config: Dict):
    
    self._running = False
    self._paused = False

    intput_configs = config["modules"]["utility"]["input"]
    output_configs = config["modules"]["utility"]["output"]


    # This currently does not have any options
    self._archive_module = ArchiveModule(output_configs["archive"])

    self._fil_manager = FilManager()
    self._fil_manager.start()
    self._fil_table = self._fil_manager.FilData()

    self._final_manager = Manager()
    self._final_table = self._final_manager.dict()

    self._candidate_manager = CandidateManager()
    self._candidate_manager.start()
    self._candidate_queue = self._candidate_manager.CandidateQueue()

    self._module_queue = ComputeQueue(config["modules"]["transform"], self._fil_table)

    logger.debug("Created queue with %d modules",
                  (len(self._module_queue)))

    self._watch_module = WatchModule(intput_configs["watch"]["base_directory"],
                                      intput_configs["watch"]["num_watchers"])

    output_configs["plot"]["modules"] = self._module_queue.started_modules()

    self._plot_module = PlotModule(output_configs["plot"])

    self._tarball_module = TarModule(output_configs["tar"])

    self._module_queue["frbid"].set_out_queue(self._final_table)
    
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

        cand_data = cand_queue.get_candidate()[1]
        self._module_queue[0].initialise(cand_data)

        for module in self._module_queue:
          if await module.process() is not None:
            break

        logger.debug("Candidate finished processing %.4fs\
                     after being added to the queue",
                     perf_counter() - cand_data.time_added)

      except asyncio.CancelledError:
        logger.info("Compute modules quitting")
        return

  def _finalise(self, cand_table, fil_table, plot_module, archive_module, tar_module) -> None:

    """

    Asynchronous method for candidate plotting and archiving.

    This method waits for new candidates to be pused to the
    queue by the FRBID module after they were processed by all the
    modules in the _module_queue.

    TODO: Really have to accept and run a variable number of modules

    Parameters:

      final_queue:
        Queue where candidates that passed all the additional processing
        stages from the _module_queue pipeline are pushed and are later
        picked up by the plot and archive modules.

    Returns:

      None

    """

    """

    while True:

      try:

        cand_data = await final_queue.get()
        logger.debug(cand_data.metadata)

        save_fil_data = (cand_data.metadata["cand_metadata"]["label"] or
                          cand_data.metadata["cand_metadata"]["known"])

        await self._plot_module.plot(cand_data)
        await self._archive_module.archive(cand_data, save_fil_data)

      except asyncio.CancelledError:
        logger.info("Computing has been finalised")
        return

    """

    def ack(ch, method, properties, body):

      message = loads(body.decode("utf-8"))
      ch.basic_ack(delivery_tag=method.delivery_tag)

      try:
        cand_data = cand_table[message["cand_hash"]]
      except KeyError: 
        logger.error("Candidate with hash %s "
                      "not present in the candidate table",
                      message["cand_hash"])
      else:

        save_fil_data = (cand_data.metadata["cand_metadata"]["label"] or
                          cand_data.metadata["cand_metadata"]["known"])
        # We need the filterbank data for the plotting
        try:
          cand_data.data = fil_table.remove_candidate(cand_data.metadata["fil_metadata"])
        except FileNotFoundError:
          # We can get an exception if the data table size was exceeded
          # and we can no longer store filterbank files in it and 
          # therefore have to read them from the disk after they were
          # removed
          logger.error("Can no longer plot and archive the file!")
        else:  
          plot_module.plot(cand_data)
          archive_module.archive(cand_data, save_fil_data)
          tar_module.create_tarball(cand_data)

    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()
    channel.basic_qos(prefetch_count=5)
    hostname = gethostname()
    channel.queue_declare("archiving_" + hostname + "_offline", durable=True, auto_delete=True)
    channel.queue_bind("archiving_" + hostname + "_offline", "post_processing")
    channel.basic_consume(queue="archiving_" + hostname + "_offline", auto_ack=False,
                          on_message_callback=ack)
    channel.start_consuming()

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

    computer = loop.create_task(self._process(self._candidate_queue))
    listener = loop.create_task(asyncio.start_server(self._listen,
                                "127.0.0.1", 9999))

    watcher = Process(target=self._watch_module.watch, name="Watcher",
                      args=(self._candidate_queue,))

    finiliser = Process(target=self._finalise, name="Finaliser",
                        args=(self._final_table,
                          self._fil_table,
                          self._plot_module,
                          self._archive_module,
                          self._tarball_module))

                        
    finiliser.start()
    watcher.start()

    await asyncio.gather(listener, computer)

    logger.info("Finishing the processing...")
    loop.stop()
    finiliser.join()
    watcher.join()

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
