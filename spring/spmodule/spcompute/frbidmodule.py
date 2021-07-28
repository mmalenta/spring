import logging
import pika

from json import dumps
from os import path
from socket import gethostname
from time import perf_counter, time
from typing import Dict

from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
from tensorflow import ConfigProto, Session

from FRBID_code.prediction_phase import load_candidate, FRB_prediction
from spmodule.spcompute.computemodule import ComputeModule

logger = logging.getLogger(__name__)

class FrbidModule(ComputeModule):

  """

  Module responsible for running the Machine Learning classifier.
  !!! CURRENTLY REQUIRES A CUDA-CAPABLE GPU !!!

  Loads the requested model, together with the weights. This model is
  then used to run the FRBID classifier on every candidate sent to it.
  Currently runs every candidate individually, without any input
  batching, which hurts the performance. Maintains and updated the
  configuration of the TensorFlow session

  Parameters:

    config: Dict, default None
      Configuration dictionary. Has to contain the information about
      the ML model used and the directory where the model can be found.
      If empty dictionary is passed or the default None, the pipeline
      will quit processing.

  Attributes:

    id: int
      Module ID
    
    _model: Keras model
      Preloaded Keras model including weights

    _out_queue: CandQueue
      Queue for sending candidates to plotting and archiving.

    _connection: BlockingConnection
      Connection for sending messages to the broker

    _channel: BlockingChannel
      Channel for sending messages to the broker

  """

  def __init__(self, config: Dict = None):

    super().__init__()
    self.id = 60
    self.type = "M"
    
    tf_config = ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.25 # pylint: disable=no-member
    set_session(Session(config=tf_config))
    
    if (config == None) or not config:
      logger.error("Invalid FRBID configuration!")
      logger.error("Will quit now!")
      exit()

    self._model = None
    self._out_queue = None

    with open(path.join(config["model_dir"],
                        config["model"] + ".json"), "r") as mf:
      self._model = model_from_json(mf.read())
    
    self._model.load_weights(path.join(config["model_dir"],
                                        config["model"] + ".h5"))

    self._connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    self._channel = self._connection.channel()

    logger.info("FRBID module initialised")

  def set_model(self, model) -> None:

    """

    Set the ML model.

    Currently not used. Might be used later to hot-swap the ML models
    during the processing

    Parameters:

      model: Keras model
        Model used for the ML classification

    Returns:

      None

    """

    self._model = model

  def set_out_queue(self, out_queue) -> None:

    """

    Set the queue used for plotting and archiving.

    Parameters:

      out_queue: CandQueue
        Queue for sending candidates to archiving.

    Returns:

      None

    """

    self._out_queue = out_queue

  async def process(self, metadata: Dict) -> None:

    """

    Run the FRBID classification on submitted candidate

		This method receives the candidate from the previous stages of
		processing and runs the ML classification on the correctly
		pre-processed candidate.

		After the classification all the candidates (this may change in
		the future, depending on the requirements) are sent
		to the archiving. Only candidates with the label of 1 are sent
		to the Supervisor and will participate in triggering.

		Parameters:

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

    cand_metadata = self._data.metadata["cand_metadata"]

    cand_metadata["label"] = label
    cand_metadata["prob"] = prob
    
    print(self._data.metadata["fil_metadata"]["fil_file"])

    logger.log(15, "Candidate at MJD %.6f and DM %.2f "
                    "from file %s, beam %d(%d) "
                    "with label %d with probability of %.4f",
                cand_metadata["mjd"],
                cand_metadata["dm"],
                self._data.metadata["fil_metadata"]["fil_file"],
                self._data.metadata["beam_metadata"]["beam_abs"],
                self._data.metadata["beam_metadata"]["beam_rel"],
                cand_metadata["label"],
                cand_metadata["prob"])

    # We add everything to the archiving
    cand_hash = str(hash(str(cand_metadata["dm"]) + str(cand_metadata["mjd"])
                      + str(self._data.metadata["beam_metadata"]["beam_abs"])))

    cand_metadata["cand_hash"] = cand_hash
    # No need to store the filterbank file data
    self._data.data = None
    self._out_queue[cand_hash] = self._data

    # Send the data to clustering only if the candidate is labelled
    # as probable and is not a known source
    if label > 0.0 and not cand_metadata["known"]:
      message = {
        "dm": self._data.metadata["cand_metadata"]["dm"],
        "mjd": self._data.metadata["cand_metadata"]["mjd"],
        "snr": self._data.metadata["cand_metadata"]["snr"],
        "beam_abs": self._data.metadata["beam_metadata"]["beam_abs"],
        "beam_type": self._data.metadata["beam_metadata"]["beam_type"],
        "ra": self._data.metadata["beam_metadata"]["beam_ra"],
        "dec":	self._data.metadata["beam_metadata"]["beam_dec"],
        "cand_hash": cand_hash,
        "time_sent": time(),
        "hostname": gethostname()
      }

      logger.debug("Sending the data")

      try:
        self._channel.basic_publish(exchange="post_processing",
                                    routing_key="clustering",
                                    body=dumps(message))

      # This is less than ideal, but anything more requires time we do
      # not currently have
      except:
        logger.error("Resetting the lost RabbitMQ connection")
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
        self._channel = self._connection.channel()
        self._channel.basic_publish(exchange="post_processing",
                                    routing_key="clustering",
                                    body=dumps(message))

    # If the candidate is not labelled as probable or is a known source
    # then send directly to archiving
    else:
      message = {
        "cand_hash": cand_hash
      }

      try:
        self._channel.basic_publish(exchange="post_processing",
                                    routing_key="archiving_" + gethostname(),
                                    body=dumps(message))

      except:
        logger.error("Resetting the lost RabbitMQ connection")
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
        self._channel = self._connection.channel()
        self._channel.basic_publish(exchange="post_processing",
                                    routing_key="archiving_" + gethostname(),
                                    body=dumps(message))

    logger.debug("Prediction took %.4fs", pred_end - pred_start)


class MultibeamModule(ComputeModule):

  """

  This module is currently not in use at all.

  FUnctionality might be added in the future or it might be removed
  altogether and replaced by the cluster-wide clustering.

  """

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