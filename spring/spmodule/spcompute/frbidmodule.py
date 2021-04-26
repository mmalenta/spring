import logging
import pika

from json import dumps
from time import perf_counter, time
from typing import Dict

from FRBID_code.prediction_phase import load_candidate, FRB_prediction
from spmodule.spcompute.computemodule import ComputeModule

logger = logging.getLogger(__name__)

class FrbidModule(ComputeModule):

  """
  Class responsible for running the ML classifier

  Runs the FRBID classifier on every candidate sent to it.
  Currently runs every candidate individually, without any input
  batching, which hurts the performance.

  Arguments:

    None

  Attributes:

    id: int
      Module ID
    
    _model: Keras model
      Preloaded Keras model including weights

    _out_queue: CandQueue
      Queue for sending candidates to archiving.

    _connection: BlockingConnection
      Connection for sending messages to the broker

    _channel: BlockingChannel
      Channel for sending messages to the broker

  """

  def __init__(self):

    super().__init__()
    self.id = 60
    logger.info("FRBID module initialised")
    self._model = None
    self._out_queue = None

    self._connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    self._channel = self._connection.channel()

  def set_model(self, model) -> None:

    self._model = model

  def set_out_queue(self, out_queue) -> None:

    self._out_queue = out_queue

  async def process(self, metadata: Dict) -> None:

    """"
    Run the FRBID classification on submitted candidate

		This method receives the candidate from the previous stages of
		processing and runs the ML classification on the correctly
		pre-processed candidate.

		After the classification all the candidates (this may change in
		the future, depending on the requirements) are sent
		to the archiving. Only candidates with the label of 1 are send
		to the Supervisor and will participate in triggering.

		Arguments:

			metadata: Dict
				Metadata information for the FRBID processing. Currently
				includes hardcoded values for the model name (NET3)
				and probability threshold for assigning the candidate
				label of 1 (0.5)

		Returns:

			None

    """

    logger.debug("FRBID module starting processing")

    pred_start = perf_counter()

    pred_data = load_candidate(self._data.ml_cand)
    prob, label = FRB_prediction(model=self._model, X_test=pred_data,
                                  probability=metadata["threshold"])

    pred_end = perf_counter()

    logger.info("Label %d with probability of %.4f", label, prob)

    self._data.metadata["cand_metadata"]["label"] = label
    self._data.metadata["cand_metadata"]["prob"] = prob

    await self._out_queue.put(self._data)

    if label > 0.0:
      message = {
        "dm": self._data.metadata["cand_metadata"]["dm"],
        "mjd": self._data.metadata["cand_metadata"]["mjd"],
        "snr": self._data.metadata["cand_metadata"]["snr"],
        "beam_abs": self._data.metadata["beam_metadata"]["beam_abs"],
        "beam_type": self._data.metadata["beam_metadata"]["beam_type"],
        "ra": self._data.metadata["beam_metadata"]["beam_ra"],
        "dec":	self._data.metadata["beam_metadata"]["beam_dec"],
        "time_sent": time()
      }

      logger.debug("Sending the data")
      self._channel.basic_publish(exchange="post_processing",
                                  routing_key="clustering",
                                  body=dumps(message))

    logger.debug("Prediction took %.4fs", pred_end - pred_start)

class MultibeamModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 70
    logger.info("Multibeam module initialised")

  async def process(self, metadata : Dict) -> None:

    """"

    Start the multibeam processing

    """

    logger.debug("Multibeam module starting processing")
    # TODO: Remember to remove it
    self._data.data = self._data.data + 1
    logger.debug("Multibeam module finished processing")