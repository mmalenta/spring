import asyncio
import cupy as cp
import logging
import matplotlib.pyplot as plt

from math import ceil
from mtcutils.core import normalise
from mtcutils import iqrm_mask as iqrm
from numpy import append, array, linspace, logical_not, mean, newaxis, random
from numpy import round as npround
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
    scaled, mean, std = normalise(self._data._data)
    mask = iqrm(std, maxlag=3)
    scaled[mask] = 0
    self._data._data = scaled
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
    logger.info("Candmaker module initialised")

  async def process(self, metadata : Dict) -> None:

    """"

    Start the candmaker processing

    """

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
    padding_required = ceil(fil_samp * time_avg_factor) * time_avg - fil_samp
    logger.debug(f"{fil_samp} original samples require {padding_required} "
                  + "samples of padding")

    pad_start = perf_counter()
    self._data._data = append(self._data._data, 
                              self._data._data[:, -1 * padding_required :],
                              axis=1)
    pad_end = perf_counter()
    logger.debug(f"Padding finished in {(pad_end - pad_start):.4}s")

    fil_samp = self._data._data.shape[1]
    logger.debug(f"{fil_samp} samples after padding")

    # Averaging
    nchans = self._data._metadata["fil_metadata"]["nchans"]
    test_sum = self._data._data[512, :time_avg].sum()
    avg_start = perf_counter()
    self._data._data = self._data._data.reshape(nchans,
                                                -1, time_avg).sum(axis=2)
    avg_end = perf_counter()
    logger.debug(f"Averaging finished in {(avg_end - avg_start):.4}s")
    fil_samp = self._data._data.shape[1]
    logger.debug(f"{fil_samp} samples after averaging")
    assert test_sum == self._data._data[512, 0]

    # Dedispersion padding

    # How many TIME AVERAGED samples is between our candidate and the
    # start of the filterbank file
    tsamp_s = tsamp_s * time_avg
    samps_from_start = round((cand_mjd - fil_mjd) * DAY_SEC / tsamp_s)
    logger.debug(f"Candidate at the MJD of {cand_mjd:.11} is "
                  + f"{samps_from_start} samples away "
                  + f"from the start of file MJD of {fil_mjd:.11}")

    # How many padding samples we need in the final product
    output_padding = 128
    # This works with negative values as well
    # If there is more data than the required padding
    # we simply ignore it
    padding_required_start = int(output_padding - samps_from_start)

    # How many TIME AVERAGED samples is between the end of our
    # dispersed candidate and the end of the filterbank file
    samps_from_end = fil_samp - samps_from_start

    # For the DMT, the highest DM we go up to is twice the candidate DM
    # Dedispersion
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
    samples_required = int(max_delay_samples + output_padding)

    if samps_from_end > samples_required:
      padding_required_end = 0
    else: 
      padding_required_end = int(max_delay_samples + output_padding
                                - samps_from_end)

    logger.debug(f"{padding_required_start} samples of padding at the start "
                  + f"and {padding_required_end} samples of padding "
                  + f"at the end required for dedispersion")

    full_samples = padding_required_start + fil_samp + padding_required_end

    pad_start = perf_counter()
    padded_data = random.random_sample((nchans, full_samples)).astype(cp.float32)
    padded_data[:, padding_required_start:padding_required_start + fil_samp] \
                = self._data._data
    pad_end = perf_counter()
    logger.debug("Dedispersion padding finished in "
                  + f"{(pad_end - pad_start):.4}s")
    fig, ax = plt.subplots(2, 1, figsize=(15,20))
    ax[0].imshow(padded_data, aspect="auto")
    ax[1].hist(padded_data.flatten(), bins=100)    
    fig.savefig("testinput.png")
    plt.close()
    # DMT
    gpu_in = cp.asarray(padded_data)
    gpu_out = cp.zeros((256, 256), dtype=cp.float32)

    # x takes care of time, y takes care of DM
    threads_x = 256
    threads_y = int(1024 / threads_x)

    blocks_x = 1
    blocks_y = int(256 / threads_y)

    delay_factors = (scaling * (1.0 / linspace(freq_top, freq_bottom, 1024)**2
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

    dmt_start = perf_counter()
    DMTKernel((blocks_x, blocks_y), (threads_x, threads_y),
              (gpu_in, gpu_out, gpu_delays, dmdiff, nchans, padded_data.shape[1]))
    cp.cuda.Device(0).synchronize()
    dmt_end = perf_counter()
    logger.debug(f"GPU DMT took {dmt_end - dmt_start:.4}s")

    cpu_out = cp.asnumpy(gpu_out)

    fig, ax = plt.subplots(2, 1, figsize=(15,20))
    ax[0].imshow(cpu_out, aspect="auto")
    ax[1].hist(cpu_out.flatten(), bins=100)    
    fig.savefig("testoutput.png")
    plt.close()

    self._data._data = self._data._data + 1
    logger.debug("Candmaker module finished processing")

class FrbidModule(ComputeModule):

  def __init__(self):

    super().__init__()
    self.id = 60
    logger.info("FRBID module initialised")

  async def process(self, metadata : Dict) -> None:

    """"

    Start the FRBID processing

    """

    logger.debug("FRBID module starting processing")
    self._data._data = self._data._data + 1
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