import logging

from typing import Dict

from module.modulequeue import ModuleQueue


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

        self._module_queue = ModuleQueue(config["modules"])

    def run(self) -> None:
        """
        Start the processing.

        This starts watching for incoming candidates.
        """
        
        self._running = True
        self._paused = False

    def stop(self) -> None:
        """
        Completely stops and cleans the pipeline.

        This method should be used only when the processing script is
        to be quit completely, i.e. after an exception that cannot be 
        recovered from occurs or a SIGKILL is caught.
        """
        
        self._running = False
        self._paused = False

    def update(self) -> None:
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
