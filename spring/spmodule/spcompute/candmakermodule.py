import logging

from time import perf_counter
from typing import Dict

import cupy as cp

from math import ceil
from numpy import append, array, clip, linspace, logical_not, mean, median
from numpy import newaxis, random, std

from spmodule.spcompute.computemodule import ComputeModule

logger = logging.getLogger(__name__)

# Seconds in a day
DAY_SEC = 86400.0

class CandmakerModule(ComputeModule):

  def __init__(self, config: Dict = None):

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