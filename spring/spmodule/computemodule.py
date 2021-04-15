import logging
import pika

from math import ceil
from time import perf_counter, time
from typing import Dict

import cupy as cp

from json import dumps
from numpy import append, array, clip, linspace, logical_not, mean, median
from numpy import newaxis, random, std

from FRBID_code.prediction_phase import load_candidate, FRB_prediction
from mtcutils.core import normalise
from mtcutils import iqrm_mask as iqrm
from spcandidate.candidate import Candidate as Cand
from spmodule.module import Module

logger = logging.getLogger(__name__)

# Seconds in a day
DAY_SEC = 86400.0

class ComputeModule(Module):

  """
  Parent class for all the compute modules.

  This class should not be used explicitly in the code.


  We break the standard class naming convention here a bit.
  To create your own module, use the CamelCase naming convention,
  with the module indentifier, followed by the word 'Module'. If an 
  acronym is present in the identifier, capitalise the first letter of
  the acronym only if present at the start; if present somewhere else,
  write it all in lowercase. This is linked to how module names and
  their corresponding command-line names are processed when added to
  the processing queue.

  """
  def __init__(self):

    self._data = array([])
    self.id = 0
    super().__init__()

  def initialise(self, indata: Cand) -> None:

    self.set_input(indata)

  def set_input(self, indata: Cand) -> None:

    self._data = indata

  def get_output(self) -> Cand:

    return self._data

class IqrmModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 10
    logger.info("IQRM module initialised")


  async def process(self, metadata : Dict):

    """"

    Start the IQRM processing

    """

    logger.debug("IQRM module starting processing")
    iqrm_start = perf_counter()
    scaled, norm_mean, norm_stdev = normalise(self._data.data)
    # TODO: Make maxlag properly configurable
    mask = iqrm(norm_stdev, maxlag=15)
    scaled[mask] = 0
    self._data.data = scaled
    self._data.mean = norm_mean
    self._data.stdev = norm_stdev
    iqrm_end = perf_counter()
    logger.debug("IQRM module finished processing in %.4fs",
                 iqrm_end - iqrm_start)

class MaskModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 20
    logger.info("Mask module initialised")


  async def process(self, metadata : Dict) -> None:

    """"

    Start the masking processing

    Applies the user-defined mask to the data.

    Parameters:

      metadata["mask"] : array
        Mask with the length of the number of channels in the
        data. Only values 0 and 1 are allowed. If not supplied,
        none of the channels will be masked.

      medatata["multiply"] : bool
        If true, then the mask is multiplicative, i.e. the data
        is multiplied with the values is the mask; if false, the
        mask is logical, i.e. 0 means not masking and 1 means
        masking. Defaults to True, i.e. multiplicative mask.

    """
    logger.debug("Mask module starting processing")
    mask = metadata["mask"]
    # Need to revert to a multiplicative mask anyway
    if metadata["multiply"] == False:
      mask = logical_not(mask)

    self._data.data = self._data.data * mask[:, newaxis] 
    logger.debug("Mask module finished processing")

class ThresholdModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 30
    logger.info("Threshold module initialised")

class ZerodmModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 40
    logger.info("ZeroDM module initialised")


  async def process(self, metadata : Dict) -> None:

    """"

    Start the zeroDM processing

    """

    logger.debug("ZeroDM module starting processing")
    zerodm_start = perf_counter()
    self._data.data = self._data.data - mean(self._data.data, axis=0)
    zerodm_end = perf_counter()
    logger.debug("ZeroDM module finished processing in %.4fs",
                 zerodm_end - zerodm_start)

class CandmakerModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 50
    # Output padding on each side of the candidate
    self._time_padding = 128
    self._time_samples = 256
    self._freq_bands = 256
    self._trial_dms = 256
    logger.info("Candmaker module initialised")

  def _normalise_clip(self, data, clip_range=None):
    """

    Normalise and clip the data.

    Normalisation is done to zero median and unit standard deviation.
    Median and standard deviation are calculated globally over the
    flattened array.
    Parematers:

      data: array
        Original data array
    
      clip_range: float
        Sigma to clip to

    Returns:

      data: array
        Normalised and clipped data

    """
    data = array(data, dtype=cp.float32)
    med = median(data)
    stdev = std(data)
    logging.debug("Data median: %.6f", med)
    logging.debug("Data std: %.6f", stdev)
    data -= med
    data /= stdev

    if clip_range != None:
      data = clip(data, -1.0 * clip_range, clip_range) 
    return data

  async def process(self, metadata : Dict) -> None:

    """

    Start the candmaker processing

    """

    cand_metadata = self._data.metadata["cand_metadata"]
    fil_metadata = self._data.metadata["fil_metadata"]

    candmaker_start = perf_counter()
    logger.debug("Candmaker module starting processing")
    
    # Averaging padding
    width_ms = cand_metadata["width"]
    tsamp_s = fil_metadata["tsamp"]
    width_samp = int(round((width_ms * 1e-03) / tsamp_s))
    logger.debug("Pulse of width %.4fms is %d samples at %.5fs",
                 width_ms, width_samp, tsamp_s)
    time_avg = int(width_samp / 2) if width_samp > 1 else 1
    time_avg_factor = 1 / time_avg
    logger.debug("Will average by a factor of %d", time_avg)

    fil_mjd = fil_metadata["mjd"]
    cand_mjd = cand_metadata["mjd"]

    fil_samp = self._data.data.shape[1]
    avg_padding_required = ceil(fil_samp * time_avg_factor) * time_avg - fil_samp
    logger.debug("%d original samples require %d samples of padding",
                 fil_samp, avg_padding_required)

    pad_start = perf_counter()
    self._data.data = append(self._data.data, 
                              self._data.data[:, -1 * avg_padding_required :],
                              axis=1)
    pad_end = perf_counter()
    logger.debug("Padding finished in %.4fs", pad_end - pad_start)

    fil_samp = self._data.data.shape[1]
    logger.debug("%d samples after padding", fil_samp)

    # Averaging
    nchans = fil_metadata["nchans"]
    avg_start = perf_counter()
    self._data.data = self._data.data.reshape(nchans,
                        -1, time_avg).sum(axis=2) / time_avg

    avg_end = perf_counter()
    logger.debug("Averaging finished in %.4fs", avg_end - avg_start)
    fil_samp = self._data.data.shape[1]
    logger.debug("%d samples after the averaging", fil_samp)

    # Dedispersion padding
    # How many TIME AVERAGED samples is between our candidate and the
    # start of the filterbank file
    tsamp_s = tsamp_s * time_avg
    samps_from_start = round((cand_mjd - fil_mjd) * DAY_SEC / tsamp_s)
    logger.debug("Candidate at the MJD of %.11f is %d samples away\
                 from the start of file MJD of %.11f",
                 cand_mjd, samps_from_start, fil_mjd)

    # How many start padding samples we need in the final product
    # If this becomes negative - we do not need padding, we need to
    # cut the data instead
    padding_required_start = int(self._time_padding - samps_from_start)

    # How many TIME AVERAGED samples is between the end of our
    # dispersed candidate and the end of the filterbank file
    samps_from_end = fil_samp - samps_from_start
    # For the DMT, the highest DM we go up to is twice the candidate DM
    top_dm = 2.0 * cand_metadata["dm"]
    freq_top = fil_metadata["fch1"]
    freq_band = fil_metadata["foff"]
    # That calculation assumes we go from the middle of top channel
    # to the middle of the bottom channel
    freq_bottom = freq_top + (nchans - 1) * freq_band
    scaling = 4.148808e+03 / tsamp_s
    max_delay_samples = int(round(scaling * top_dm * (1 / freq_bottom**2
                            - 1 /freq_top**2)))

    logger.debug("The delay at the DM of %.5f is %d samples at %.5fms",
                 top_dm, max_delay_samples, tsamp_s)

    # Each DM in the DMT output needs end padding samples
    samples_required = int(max_delay_samples + self._time_padding)

    # How many end padding samples we need in the final product
    # If this becomes negative - we do not need padding, we need to
    # cut the data instead
    padding_required_end = int(samples_required - samps_from_end)
    full_samples = padding_required_start + fil_samp + padding_required_end

    logger.debug(f"{abs(padding_required_start)} samples "
                + f"{'of padding' if padding_required_start > 0 else 'removed'}"
                + f" at the start and {abs(padding_required_end)} samples "
                + f"{'of padding' if padding_required_end > 0 else 'removed'}"
                + f" at the end for dedispersion")

    logger.debug("Final input with %d samples will be used", full_samples)

    if padding_required_start < 0:
      input_skip_start = abs(padding_required_start)
      padding_required_start = 0
    else:
      input_skip_start = 0

    if padding_required_end < 0:
      input_skip_end = abs(padding_required_end)
      padding_required_end = 0
    else:
      input_skip_end = 0

    pad_start = perf_counter()
    # Recalculate these
    new_mean = mean(self._data.data[:, input_skip_start:fil_samp - input_skip_end], axis=1)
    new_stdev = std(self._data.data[:, input_skip_start:fil_samp - input_skip_end], axis=1)

    padded_data = ((random.randn(nchans, full_samples).astype(cp.float32).T * new_stdev).T
                  + new_mean[:, newaxis])
    padded_data[:, padding_required_start:padding_required_start + fil_samp - input_skip_end - input_skip_start] \
                = self._data.data[:, input_skip_start:fil_samp - input_skip_end]
    pad_end = perf_counter()
    logger.debug("Dedispersion padding finished in %.4fs", pad_end - pad_start)
    
    # DMT
    dmt_gpu_in = cp.asarray(padded_data)
    dmt_gpu_out = cp.zeros((self._trial_dms, self._time_samples),
                            dtype=cp.float32)
    dedisp_gpu_out = cp.zeros((self._freq_bands, self._time_samples),
                              dtype=cp.float32)

    delay_factors = (scaling * (1.0 / linspace(freq_top, freq_bottom, nchans)**2
                    - 1 / freq_top**2)).astype(cp.float32)
    gpu_delays = cp.asarray(delay_factors)
    # We start at 0
    dmdiff = cp.float32(top_dm / 255)

    DMTKernel = cp.RawKernel(r"""

      extern "C" __global__ void dmt_kernel(float* __restrict__ indata,
                                            float* __restrict__ outdata,
                                            float* __restrict__ delays,
                                            float dmdiff,
                                            int nchans,
                                            int nsamp)
      {
        int tidx = threadIdx.x;
        int dmidx = blockIdx.y * blockDim.y + threadIdx.y;
        float dm = dmdiff * dmidx;
        
        int outidx = dmidx * blockDim.x + tidx;

        for (int ichan = 0; ichan < nchans; ++ichan) {
          outdata[outidx] = outdata[outidx] + 
            indata[ichan * nsamp + tidx + __float2int_rd(dm * delays[ichan])];  
        }
      }
    """, "dmt_kernel")

    DedispKernel = cp.RawKernel(r"""
    
      extern "C" __global__ void dedisp_kernel(float* __restrict__ indata,
                                                float* __restrict__ outdata,
                                                float* __restrict__ delays,
                                                float dm,
                                                int band_size,
                                                int nsamp)
      {
        int tidx = threadIdx.x;
        int bandidx = blockIdx.y * blockDim.y + threadIdx.y;
        int chanidx = bandidx * band_size;
        int outidx = blockDim.x * bandidx + tidx;

        float sum = 0.0;

        int skip_band = chanidx * nsamp;

        for (int ichan = 0; ichan < band_size; ++ichan) {
          sum += indata[skip_band + tidx + __float2int_rd(dm * delays[chanidx])];
          skip_band += nsamp;
          chanidx += 1;
        }

        outdata[outidx] = sum;

      }

    """, "dedisp_kernel")

    band_size = int(nchans / self._freq_bands)

    # These are two independed kernels - could we run then in streams?
    dmt_start = perf_counter()
    # x takes care of time, y takes care of DM
    threads_x = self._time_samples
    threads_y = int(1024 / threads_x)
    blocks_x = 1
    blocks_y = int(self._trial_dms / threads_y)

    DMTKernel((blocks_x, blocks_y), (threads_x, threads_y),
              (dmt_gpu_in, dmt_gpu_out, gpu_delays, dmdiff, nchans,
              padded_data.shape[1]))
    cp.cuda.Device(0).synchronize()
    dmt_end = perf_counter()
    logger.debug("GPU DMT took %.4fs", dmt_end - dmt_start)

    dedisp_start = perf_counter()
    # x takes care of time, y takes care of frequency band
    threads_x = self._time_samples
    threads_y = int(1024 / threads_x)
    blocks_x = 1
    blocks_y = int(self._freq_bands / threads_y)

    DedispKernel((blocks_x, blocks_y), (threads_x, threads_y),
                  (dmt_gpu_in, dedisp_gpu_out, gpu_delays,
                  cp.float32(cand_metadata["dm"]), band_size, padded_data.shape[1]))
    cp.cuda.Device(0).synchronize()
    dedisp_end = perf_counter()
    logger.debug("GPU dedispersion to DM of %.5f took %.4fs",
                 cand_metadata["dm"], dedisp_end - dedisp_start)

    dmt_cpu_out = cp.asnumpy(dmt_gpu_out)
    dedisp_cpu_out = cp.asnumpy(dedisp_gpu_out)

    self._data.ml_cand["dmt"] = self._normalise_clip(dmt_cpu_out)
    self._data.ml_cand["dedisp"] = self._normalise_clip(dedisp_cpu_out, 3)

    candmaker_end = perf_counter()
    logger.debug("Candmaker module finished in %.4fs",
                 candmaker_end - candmaker_start)

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