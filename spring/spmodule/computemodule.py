import asyncio
import cupy as cp
import logging
import h5py as h5
import matplotlib.pyplot as plt

from FRBID_code.prediction_phase import load_candidate, FRB_prediction
from math import ceil, sqrt
from mtcutils.core import normalise
from mtcutils import iqrm_mask as iqrm
from numpy import append, array, clip, linspace, logical_not, mean, median
from numpy import newaxis, random, round as npround, std
from time import perf_counter, sleep
from typing import Dict


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
    scaled, mean, stdev = normalise(self._data._data)
    mask = iqrm(stdev, maxlag=3)
    scaled[mask] = 0
    self._data._data = scaled
    self._data._mean = mean
    self._data._stdev = stdev
    iqrm_end = perf_counter()
    logger.debug("IQRM module finished processing in "
                  + f"{(iqrm_end - iqrm_start):.4}s")
    #await asyncio.sleep(5)
    sleep(0.1)

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
    if (metadata["multiply"] == False):
      mask = logical_not(mask)

    self._data = self._data * mask[:, newaxis] 
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
    print(self._data._data.shape)
    zerodm_start = perf_counter()
    self._data._data = self._data._data - mean(self._data._data, axis=0)
    zerodm_end = perf_counter()
    logger.debug("ZeroDM module finished processing in "
    + f"{(zerodm_end - zerodm_start):.4}s")

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
    logging.debug(f'Data median: {med}')
    logging.debug(f'Data std: {stdev}')
    data -= med
    data /= stdev

    if clip_range != None:
        data = clip(data, -1.0 * clip_range, clip_range) 
    return data

  async def process(self, metadata : Dict) -> None:

    """"

    Start the candmaker processing

    """

    cand_metadata = self._data._metadata["cand_metadata"]
    fil_metadata = self._data._metadata["fil_metadata"]

    candmaker_start = perf_counter()
    logger.debug("Candmaker module starting processing")
    
    # Averaging padding
    width_ms = self._data._metadata["cand_metadata"]["width"]
    tsamp_s = self._data._metadata["fil_metadata"]["tsamp"]
    width_samp = int(round((width_ms * 1e-03) / tsamp_s))
    logger.debug(f"Pulse of width {width_ms}ms is {width_samp} samples "
                  + f"@ {tsamp_s:.5}s")
    time_avg = int(width_samp / 2) if width_samp > 1 else 1
    time_avg_factor = 1 / time_avg
    logger.debug(f"Will average by a factor of {time_avg}")

    fil_mjd = self._data._metadata["fil_metadata"]["mjd"]
    cand_mjd = self._data._metadata["cand_metadata"]["mjd"]

    fil_samp = self._data._data.shape[1]
    avg_padding_required = ceil(fil_samp * time_avg_factor) * time_avg - fil_samp
    logger.debug(f"{fil_samp} original samples require {avg_padding_required} "
                  + "samples of padding")

    pad_start = perf_counter()
    self._data._data = append(self._data._data, 
                              self._data._data[:, -1 * avg_padding_required :],
                              axis=1)
    pad_end = perf_counter()
    logger.debug(f"Padding finished in {(pad_end - pad_start):.4}s")

    fil_samp = self._data._data.shape[1]
    logger.debug(f"{fil_samp} samples after padding")

    # Averaging
    nchans = self._data._metadata["fil_metadata"]["nchans"]
    avg_start = perf_counter()
    self._data._data = self._data._data.reshape(nchans,
                        -1, time_avg).sum(axis=2) / time_avg

    

    avg_end = perf_counter()
    logger.debug(f"Averaging finished in {(avg_end - avg_start):.4}s")
    fil_samp = self._data._data.shape[1]
    logger.debug(f"{fil_samp} samples after averaging")

    # Dedispersion padding
    # How many TIME AVERAGED samples is between our candidate and the
    # start of the filterbank file
    tsamp_s = tsamp_s * time_avg
    samps_from_start = round((cand_mjd - fil_mjd) * DAY_SEC / tsamp_s)
    logger.debug(f"Candidate at the MJD of {cand_mjd:.11} is "
                  + f"{samps_from_start} samples away "
                  + f"from the start of file MJD of {fil_mjd:.11}")

    # How many start padding samples we need in the final product
    # If this becomes negative - we do not need padding, we need to
    # cut the data instead
    padding_required_start = int(self._time_padding - samps_from_start)

    # How many TIME AVERAGED samples is between the end of our
    # dispersed candidate and the end of the filterbank file
    samps_from_end = fil_samp - samps_from_start
    # For the DMT, the highest DM we go up to is twice the candidate DM
    top_dm = 2.0 * self._data._metadata["cand_metadata"]["dm"]
    freq_top = self._data._metadata["fil_metadata"]["fch1"]
    freq_band = self._data._metadata["fil_metadata"]["foff"]
    # That calculation assumes we go from the middle of top channel
    # to the middle of the bottom channel
    freq_bottom = freq_top + (nchans - 1) * freq_band
    scaling = 4.148808e+03 / tsamp_s
    max_delay_samples = int(round(scaling * top_dm * (1 / freq_bottom**2
                            - 1 /freq_top**2)))

    logger.debug(f"The delay at the DM of {top_dm:.5} is "
                  + f"{max_delay_samples} samples @ {tsamp_s:.5}ms")

    # Each DM in the DMT output needs end padding samples
    samples_required = int(max_delay_samples + self._time_padding)

    # How many end padding samples we need in the final product
    # If this becomes negative - we do not need padding, we need to
    # cut the data instead
    padding_required_end = int(samples_required - samps_from_end)
    full_samples = padding_required_start + fil_samp + padding_required_end

    logger.debug(f"{padding_required_start} samples "
                + f"{'of padding' if padding_required_start > 0 else 'removed'}"
                + f" at the start and {padding_required_end} samples "
                + f"{'of padding' if padding_required_end > 0 else 'removed'}"
                + f" at the end for dedispersion")

    logger.debug(f"Final input with {full_samples} samples will be used")

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
    new_mean = mean(self._data._data[:, input_skip_start:fil_samp - input_skip_end], axis=1)
    new_stdev = std(self._data._data[:, input_skip_start:fil_samp - input_skip_end], axis=1)

    padded_data = ((random.randn(nchans, full_samples).astype(cp.float32).T * new_stdev).T
                  + new_mean[:, newaxis])
    padded_data[:, padding_required_start:padding_required_start + fil_samp - input_skip_end - input_skip_start] \
                = self._data._data[:, input_skip_start:fil_samp - input_skip_end]
    pad_end = perf_counter()
    logger.debug("Dedispersion padding finished in "
                  + f"{(pad_end - pad_start):.4}s")
    
    """
    fig, ax = plt.subplots(2, 1, figsize=(15,20))
    ax[0].imshow(padded_data, aspect="auto")
    ax[1].hist(padded_data.flatten(), bins=100)    
    fig.savefig("testinput" + self._data._metadata["fil_metadata"]["fil_file"] + ".png")
    plt.close()
    """
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
    logger.debug(f"GPU DMT took {dmt_end - dmt_start:.4}s")

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
    logger.debug(f"GPU dedispersion to DM of {cand_metadata['dm']} "
                  + f"took {dedisp_end - dedisp_start:.4}s")

    dmt_cpu_out = cp.asnumpy(dmt_gpu_out)
    dedisp_cpu_out = cp.asnumpy(dedisp_gpu_out)

    self._data._ml_cand["dmt"] = self._normalise_clip(dmt_cpu_out)
    self._data._ml_cand["dedisp"] = self._normalise_clip(dedisp_cpu_out, 3)

    fig, ax = plt.subplots(2, 1, figsize=(15,20))
    ax[0].imshow(dmt_cpu_out, aspect="auto", interpolation="none")
    ax[1].imshow(dedisp_cpu_out, aspect="auto", interpolation="none")    
    fig.savefig("testoutput" + self._data._metadata["fil_metadata"]["fil_file"] + ".png")
    plt.close()
    
    hdf5_start = perf_counter()


    hdf5_end = perf_counter()
    logger.debug(f"HDF5 writing took {hdf5_end - hdf5_start:.4}s")


    self._data._data = self._data._data + 1
    logger.debug("Candmaker module finished processing")
    candmaker_end = perf_counter()
    logger.debug("Candmaker module finished in "
                  + f"{(candmaker_end - candmaker_start):.4}s")

class FrbidModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 60
    logger.info("FRBID module initialised")

    self._model = None
    self._out_queue = None

  def set_model(self, model) -> None:

    self._model = model

  def set_out_queue(self, out_queue) -> None:

    self._out_queue = out_queue

  async def process(self, metadata : Dict) -> None:

    """"

    Start the FRBID processing

    """

    logger.debug("FRBID module starting processing")

    pred_start = perf_counter()

    pred_data = load_candidate(self._data._ml_cand)
    prob, label = FRB_prediction(model=self._model, X_test=pred_data,
                                  probability=metadata["threshold"])

    pred_end = perf_counter()

    logger.info(f"Label {label} with probability of {prob}")

    if label > 0.0:
      await self._out_queue.put(self._data)

    logger.debug(f"Prediction took {pred_end - pred_start:.4}s")
    logger.debug("FRBID module finished processing")

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
    self._data._data = self._data + 1
    logger.debug("Multibeam module finished processing")