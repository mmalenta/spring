from statistics import median
import cupy as cp
import logging
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import pkg_resources

from numpy import append, arange, ceil, float32, floor, int32, linspace, log10
from numpy import max as npmax, mean, min as npmin, newaxis, random, reshape
from numpy import std, sum as npsum, zeros
from os import mkdir, path
from time import perf_counter
from typing import Dict

from spcandidate.candidate import Candidate as Cand
from spmodule.sputility.spoutput.outputmodule import OutputModule

logger = logging.getLogger(__name__)

class PlotModule(OutputModule):

  """

  Module responsible for plotting candidates that pass earlier
  processing stages. !!! CURRENTLY REQUIRES A CUDA-CAPABLE GPU !!!

  Creates a plot used for candidate inspection. Currently using a
  non-configurable plot layout (configurability will be added in the
  future),

  Parameters:

    config: Dict
      Module configuration dictionary

  Attributes:

    _plots: List[List[tupele]]
      Currenty not in use.
      Plots configuration. Each row of the plot is represented by the
      inner list. The configuration of the individual plot in that row
      is represented by the tuple.

    _out_bands: int
      Number of the output bands in the non-dedispersed and dedisperse
      Frequency-Time plots
    
    _modules: List[str]
      Used processing modules abbreviations. Displayed on the plot
      to provide a quick identification of modules that were used
      to produce given plot.

    _version: str
      Current pipeline version. Displayed on the plot to provide a
      quick identification of the pipeline version that was used to
      produce given plot.

  """

  def __init__(self, config: Dict):
    super().__init__()

    #self._plots = config["plots"]
    self._out_bands = config["out_chans"]
    self._modules = config["modules"]

    """
    # Normalise, just in case values do not add up to 1.0
    for row in self._plots:
      norm = sum([y for (x, y) in row])
      for idx, (x, y) in enumerate(row):
        row[idx] = (x, y / norm)
    """

    self._version = pkg_resources.require("spring")[0].version

  def _pad_data(self, inputdata, fil_mjd,
                cand_dm, cand_mjd,
                ftop, fband, nchans, outbands,
                disp_const, plot_pad_s, tsamp_scaling, thread_x):

    """

    Extracts and pads the data for dedispersion.

    Extracts enough data to cover the whole dispersion sweep in the
    non-dedispersed frequency-time plot plus additional padding on both
    sides. If not enough data is present, i.e. candidate detected
    at the very start or the end of the file, additional padding drawn
    from random normal distribution is added. This of course can lead
    to some pulses ending abruptly.

    Parameters:

      inputdata: NumPy Array
        Filterbank data. Depending on the previous processing steps,
        it might not be the original raw filterbank file.
      
      fil_mjd: float
        MJD of the start of the filterbank file, as found in the
        filterbank file header.

      cand_dm: float
        DM of the candidate, as found in the .spccl file.

      cand_mjd: float
        MJD of the candidate, as found in the .spccl file.

      ftop: float [MHz]
        Frequency at the centre of the highest frequency band in the
        filterbank file, as found in the filterbank file header.

      fband: float [MHz]
        The width of a single, original (non-averaged) frequency band
        in the filterbank file, as found in the filterbank file header.

      nchans: int
        The number of original (non-averaged) frequency channels.

      outbands: int
        The number of output (averaged) frequency channels.
        The data is not averaged as such to create output bands. It is
        first properly dedispersed within the subband of outband
        channels and then averaged to keep the data quality as high
        as possible.

      disp_const: float [s / (pc cm^-3)]
        Dispersion scaling constant that expressed the delay across the
        whole filterbank file band per unit DM. To get the total delay
        in seconds, just multiply this constant by the candidate DM

      plot_pad_s: float [s]
        Length of padding to add at before the candidate start,
        marked with the candidate detection MJD and after the end of
        the dispersion sweep. Either 20 times the width of the
        candidated or 0.5s, whichever is smaller.

      tsamp_scaling: float [1/s]
        The reciprocal of sampling time.

      thread_x: int
        The number of CUDA threads per block in the x dimension. This
        dimension is responsible for processing time samples. We
        therefore make sure that the number of output samples is an
        integer multiple of this number to avoid any problems with
        remainders. Additionally set to the width of the warp so that
        we can easily use warp-level reduction primitives.

    Returns: 

      use_data: NumPy Array
        Properly extracted and padded data.

      input_samples: int
        The number of the time samples in the properly extracted and
        padded data.

      output_samples_full_dedisp_orig: int
        The original (non-thread corrected) number of the output time
        samples in the fully dedispersed data (used for the dedispersed
        freqiency-time plot and the flattened 'power' plot).

      output_samples_full_dedisp_warp_safe: int
        The number of the output time samples in the fully dedispersed
        data (used for the dedispersed freqiency-time plot and the
        flattened 'power' plot) that is a multiple of the warp size.

      output_samples_sub_dedisp_orig: int
        The original (non-tread corrected) number of the output time
        samples in the 'subband' dedispersed data (used for the 
        dispersion sweep plot).

      output_samples_sub_dedisp_warp_safe: int
        The number of the output time samples in the 'subband'
        dedispersed data (used for the dispersion sweep plot) that is a
        multiple of the warp size.

      start_padding_added: int
        The number of samples of time padding added at the start of the
        data array. This is used to properly offset the time axis in
        all the plots. 0 if no padding was added.

    """

    # Every part of the plot has to be an integer multiple of threadblock in the time dimension
    # This ensures we can have an integer number of threadblocks in the time dimension
    dm = cand_dm
    original_data_length = inputdata.shape[1]
    plot_padding_samples = int(ceil(plot_pad_s * tsamp_scaling))
    cand_samples_from_start = int(ceil((cand_mjd - fil_mjd) * 86400.0 * tsamp_scaling))

    # Dispersive delay in seconds across the entire band
    full_band_delay_seconds = disp_const * dm
    # Dispersive delay in samples - make sure tsamp is expressed in seconds so that units agree
    full_band_delay_samples = int(ceil(full_band_delay_seconds * tsamp_scaling))

    last_band_bottom = ftop + (nchans - 1) * fband
    last_band_top = last_band_bottom - (nchans / outbands - 1) * fband
    last_band_delay_samples = int(ceil(4.15e+03 * dm * (1.0 / (last_band_bottom * last_band_bottom) - 1.0 / (last_band_top * last_band_top)) * tsamp_scaling))

    zero_padding_samples_start = 0
    zero_padding_samples_end = 0
    plot_skip_samples = 0
    start_padding_added = 0

    # We don't have enough samples to cover padding at the start
    #if ((cand_mjd - plot_padding_mjd) < fil_mjd):
    if (cand_samples_from_start < plot_padding_samples):
      # Difference in samples (plot padding is now a multiple of the threadblock time dimension, so extra data padding takes this into account)
      #zero_padding_samples_start = plot_padding_samples - int(np.floor((cand_mjd - fil_mjd) * 86400.0 * self._tsamp_scaling))
      zero_padding_samples_start = plot_padding_samples - cand_samples_from_start
      start_padding_added = zero_padding_samples_start
      logger.debug("Not enough data at the start. Padding with %d samples",
                   zero_padding_samples_start)

    # We don't have enough samples to cover the dispersive delay and padding at the end
    if (cand_samples_from_start + full_band_delay_samples + plot_padding_samples + last_band_delay_samples > original_data_length):
      zero_padding_samples_end = int(cand_samples_from_start + full_band_delay_samples + plot_padding_samples + last_band_delay_samples - original_data_length)
      logger.debug("Not enough data at the end. Padding with %d samples",
                   zero_padding_samples_end)

    # Need to make sure that the output is a multiple of 32 in the time dimension
    # !!!! THIS PART NEEDS EXTRA REVIEW OF THE WARP SAFETY PADDING !!!!
    output_samples_full_dedisp_orig = int(2 * plot_padding_samples)
    output_samples_full_dedisp_warp_safe = int(ceil(output_samples_full_dedisp_orig / thread_x) * thread_x)

    output_samples_sub_dedisp_orig = int(2 * plot_padding_samples + full_band_delay_samples - last_band_delay_samples)
    output_samples_sub_dedisp_warp_safe = int(ceil(output_samples_sub_dedisp_orig / thread_x) * thread_x)

    # Unless the output size is already a multiple of 32, which is very unlikely, we need to add extra padding
    # This has to be done with subband dedispersion samples - this will always be greater than full dedisp
    # warp_safety_samples = output_samples_full_dedisp_warp_safe - output_samples_full_dedisp_orig
    warp_safety_samples = output_samples_sub_dedisp_warp_safe - output_samples_sub_dedisp_orig
    total_data_samples = zero_padding_samples_start + zero_padding_samples_end + original_data_length + warp_safety_samples
    plot_skip_samples = max(cand_samples_from_start - plot_padding_samples + start_padding_added, 0)

    # Create this array ONLY if extra padding is required

    data_mean = mean(inputdata, axis=1)
    data_std = std(inputdata, axis=1)

    padded_input_data = (random.randn(nchans, total_data_samples).astype(float32).T * data_std).T + data_mean[:, newaxis]

    #padded_input_data = zeros((nchans, total_data_samples), dtype=float32)
    padded_input_data[:, zero_padding_samples_start : zero_padding_samples_start + original_data_length] = inputdata
    
    #input_samples = int(2 * plot_padding_samples + full_band_delay_samples + last_band_delay_samples)
    input_samples = output_samples_sub_dedisp_warp_safe + int(2 * last_band_delay_samples)
    # use_data = np.copy(padded_input_data[:, plot_skip_samples : plot_skip_samples + input_samples])
    # That makes sure that all the plots have the candidate at a reasonable place - start time minus the extra window
    # This is very wrong - this can be shorter than the output samples when taking the warp safety into account
    use_data = padded_input_data[:, plot_skip_samples : plot_skip_samples + input_samples]

    logger.debug("Candidate padding information:")
    logger.debug("\tInput data length (original): %d", original_data_length)
    logger.debug("\tInput data length (padded): %d", input_samples)
    logger.debug("\tOutput plot samples: %d (subband), %d (full)",
                 output_samples_sub_dedisp_orig,
                 output_samples_full_dedisp_orig)
    logger.debug("\tWarp safe output plot samples: %d (subband), %d (full)",
                 output_samples_sub_dedisp_warp_safe,
                 output_samples_full_dedisp_warp_safe)
    logger.debug("\tDM: %.2f", dm)
    logger.debug("\tDM sweep seconds: %.6f", full_band_delay_seconds)
    logger.debug("\tDM sweep samples: %d", full_band_delay_samples)
    logger.debug("\tDelay across the last band: %d", last_band_delay_samples)
    logger.debug("\tCandidate samples from the start: %d",
                 cand_samples_from_start)
    logger.debug("\tPadding at the start: %d", zero_padding_samples_start)
    logger.debug("\tPadding at the end: %d", zero_padding_samples_end)
    logger.debug("\tSamples skipped at the start: %d", plot_skip_samples)
    logger.debug("\tWarp-safety padding: %d", warp_safety_samples)

    return use_data, input_samples, output_samples_full_dedisp_orig, output_samples_full_dedisp_warp_safe, output_samples_sub_dedisp_orig, output_samples_sub_dedisp_warp_safe, start_padding_added

  def plot(self, data: Cand) -> None:

    """

    Create the candidate plot and save it.

    Takes the incoming data, applies additional processing to make it
    fit for plotting, plots it in a (currently) non-configurable plot
    and saves it. We have chosen a decent-quality JPG files as a
    compromise between the quality and the size of the plots and save
    time.

    Parameters:

      data: Dict
        Dictionary with all the necessary candidate information. 
        Contains the the array with the filterbank data, filterbank
        metadata with all the filterbank header information, candidate
        metadata with all the candidate detection information and beam
        metadata with the information on the beam where tha candidate
        was detected.

    Returns:

      None

    """

    fil_metadata = data.metadata["fil_metadata"]
    cand_metadata = data.metadata["cand_metadata"]
    beam_metadata = data.metadata["beam_metadata"]

    nchans = fil_metadata["nchans"]
    freq_avg = int(nchans / self._out_bands)
    tsamp = fil_metadata["tsamp"]
    ftop = fil_metadata["fch1"]
    fband = fil_metadata["foff"]
    # That assumes negative channel band and that ftop is in the middle
    # of the top channel - make sure this is correct
    fbottom = ftop + (nchans - 1) * fband
    # Plots will have padding of 20 times the pulse width or 0.5s,
    # whichever is smaller
    # Quite an arbitrary number that just 'looks good' - no real science
    # behind it
    # REMEMBER: width is expressed in ms
    plot_pad_s = min(ceil(20.0 * cand_metadata["width"]) * 1e-03, 0.5)
    # This is expressed in s per unit DM
    # Just multiply by a DM value to get a delay across the band in s
    disp_const = 4.15e+03 * (1.0 / (fbottom * fbottom) - 1.0 / (ftop * ftop))
    avg_fband = fband * freq_avg
    width_samp = int(round(cand_metadata["width"] * 1e-03 / tsamp))
    # Currently assume that we receive the data from the candmaker
    # and it is averaged
    # TODO: pass the average value rather than recalculate it
    time_avg = int(width_samp / 2) if width_samp > 1 else 1

    # Recreate what candmaker is doing - we no longer get the averaged
    # data
    fil_samp = data.data["data"].shape[1]
    avg_padding_required = int(ceil(fil_samp / time_avg) * time_avg - fil_samp)

    data.data["data"] = append(data.data["data"], 
                              data.data["data"][:, -1 * avg_padding_required :],
                              axis=1)

    data.data["data"] = data.data["data"].reshape(nchans,
                        -1, time_avg).sum(axis=2) / time_avg

    tsamp = tsamp * time_avg
    tsamp_scaling = 1.0 / tsamp

    # For subband dedispersion - dispersion delays WITHIN the band
    cpu_intra_band_delays = zeros((nchans, ), dtype=int32)
    # For final dedispersion - dispersion delays BETWEEN the bands
    cpu_inter_band_delays = zeros((self._out_bands,), dtype=int32)
    
    # Common values that we better not recalculate
    ftop_part = 1.0 / (ftop * ftop)
    scaling = 4.15e+03 * cand_metadata["dm"] * tsamp_scaling
    # How much to offset the line from the original pulse
    offset = 0.75 * plot_pad_s * tsamp_scaling

    # Need to pythonify this properly and remove for loops
    for iband in arange(self._out_bands):
      # Need to move to the middle of the channel
      bandtop = ftop + iband * avg_fband + 0.5 * fband
      for ichan in arange(freq_avg):
        full_chan = iband * freq_avg + ichan
        chanfreq = bandtop + ichan * fband
        cpu_intra_band_delays[full_chan] =  int(round(scaling * (1.0 / (chanfreq * chanfreq) - 1.0 / (bandtop * bandtop))))

      centre_band = ftop + iband * avg_fband
      cpu_inter_band_delays[int(iband)] =  int(round(scaling * (1.0 / (centre_band * centre_band) - ftop_part)))

    # Delay values for a line in the plot
    line_delays = cpu_inter_band_delays + offset

    thread_y = 32
    thread_x = 32
    # Pad the data
    use_data, input_samples, full_dedisp_samples_orig, \
      full_dedisp_samples_gpu, sub_dedisp_samples_orig, \
      sub_dedisp_samples_gpu, skip_samples = \
      self._pad_data(data.data["data"], fil_metadata["mjd"],
                     cand_metadata['dm'], cand_metadata['mjd'],
                     ftop, fband, nchans, self._out_bands,
                     disp_const, plot_pad_s, tsamp_scaling, thread_x)

    SubDedispGPUHalf = cp.RawKernel(r'''
      // This version of the kernel assumes we use 16 channels per dedispersed subband 

      extern "C" __global__ void sub_dedisp_kernel_half(float* __restrict__ indata, float* __restrict__ outdata,
                                              int* __restrict__ intra_band_delays,
                                              int input_samples, int sub_dedisp_samples, int bands) {
          
          __shared__ float inchunk[32][32];
          
          if ((2 * blockIdx.x * blockDim.x + 2 * threadIdx.y) < sub_dedisp_samples) {
            int band = blockIdx.y;
            // This kernel is used specifically for dedispersing 16 channels per subband
            int band_size = 16;
            // Half of the .y threads read time samples
            int channel = threadIdx.y & 15;
            int time_chunk = threadIdx.y >> 4;
            int lane = threadIdx.x % 32;
            // Each thread processes two time samples
            int time = blockIdx.x * blockDim.x * 2 + time_chunk * blockDim.x + threadIdx.x + intra_band_delays[band * band_size + channel];

            int skip_band = band * band_size * input_samples;
            int skip_channel = channel * input_samples;
            // Quick and dirty transpose and dedispersion of the data
            inchunk[(threadIdx.x >> 1) + (time_chunk << 4)][((threadIdx.x & 1) << 4) + channel] = indata[skip_band + skip_channel + time];
            __syncthreads();

            float val = inchunk[threadIdx.y][threadIdx.x];
            // Make sure each thread in a warp has a separate channel
            for (int offset = 1; offset < 16; offset *= 2) {
              val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            }

            __syncwarp();

            int out_skip_band = sub_dedisp_samples * band;
            int out_skip_time = blockIdx.x * blockDim.x * 2 + threadIdx.y * 2;

            if(lane == 0) {
              outdata[out_skip_band + out_skip_time + 0] = val;
            } else if (lane == 16) {
              outdata[out_skip_band + out_skip_time + 1] = val;
            }
          }
      }

    ''', 'sub_dedisp_kernel_half')

    SubDedispGPUWarp = cp.RawKernel(r'''
    
      extern "C" __global__ void sub_dedisp_kernel_warp(float* __restrict__ indata, float* __restrict__ outdata,
                                              int* __restrict__ intra_band_delays,
                                              int input_samples, int sub_dedisp_samples, int freqavg) {
          
          __shared__ float inchunk[32][32];
          __shared__ float subband[32];
          subband[threadIdx.y] = 0;

          int time = blockIdx.x * blockDim.x + threadIdx.x;
          int band = blockIdx.y;
          int channel = threadIdx.y;
          int lane = threadIdx.x % 32;
          int intra_delay;

          int skip_band = band * freqavg * input_samples;
          int skip_channel = channel * input_samples;
          // Quick and dirty transpose and dedispersion of the data
          //inchunk[threadIdx.x * blockDim.y + threadIdx.y] = indata[skip_band + skip_channel + time + intra_delay];

          int num_reps = freqavg / 32;
          for (int irep = 0; irep < num_reps; ++ irep) {
              
              intra_delay = intra_band_delays[band * freqavg + channel];
              inchunk[threadIdx.x][threadIdx.y] = indata[skip_band + skip_channel + time + intra_delay];
              __syncthreads();

              float val = inchunk[threadIdx.y][threadIdx.x];
              for (int offset = 16; offset > 0; offset /= 2) {
                  val += __shfl_down_sync(0xFFFFFFFF, val, offset);
              }

              if (lane == 0) {
                  subband[threadIdx.y] += val;
              }
              // We process 32 channels at once
              channel += 32;
              skip_channel += 32 * input_samples;

              __syncthreads();

          }
          __syncthreads();

          if (threadIdx.y == 0) {
              outdata[sub_dedisp_samples * band + time] = subband[threadIdx.x];
          }

      }

    ''', 'sub_dedisp_kernel_warp')

    FullDedispGPU = cp.RawKernel(r'''

      extern "C" __global__ void full_dedisp_kernel(float* __restrict__ outdata, int* __restrict__ inter_band_delays,
                                                      int sub_dedisp_samples, int full_dedisp_samples, int bands) {

          int skip_sub_dedisp = sub_dedisp_samples * bands;
          int skip_sub_and_full_dedisp = skip_sub_dedisp + full_dedisp_samples * bands;

          int band = threadIdx.y;
          int time = blockIdx.x * blockDim.x + threadIdx.x;

          int inter_delay = inter_band_delays[band];

          float val = outdata[sub_dedisp_samples * band + time + inter_delay];

          outdata[skip_sub_dedisp + full_dedisp_samples * band + time] = val;

          atomicAdd(&outdata[skip_sub_and_full_dedisp + time], val);

      }

    ''', 'full_dedisp_kernel')

    block_x = int(sub_dedisp_samples_gpu / thread_x)
    block_y = self._out_bands

    dedisp_start = perf_counter()
    gpu_input = cp.asarray(use_data)

    logger.debug("CPU input array shape: %d x %d",
                 use_data.shape[0], use_data.shape[1])
    logger.debug("GPU input array shape: %d x %d",
                 gpu_input.shape[0], gpu_input.shape[1])

    gpu_output = cp.zeros(sub_dedisp_samples_gpu * self._out_bands + full_dedisp_samples_gpu * self._out_bands + full_dedisp_samples_gpu, dtype=use_data.dtype)
    gpu_intra_band_delays = cp.asarray(cpu_intra_band_delays)
    gpu_inter_band_delays = cp.asarray(cpu_inter_band_delays)

    kernels_start = perf_counter()
    sub_kernel_start = perf_counter()
    if (freq_avg == 16):
      # We divide time sampels in .y direction as well
      # Can't floor - need to make sure we don't end up with 0 where there is 1 block originally
      # Need extra checks for overflowing inside the kernel
      block_x = int(ceil(block_x / 2))
      SubDedispGPUHalf((block_x, block_y), (thread_x, thread_y), (gpu_input, gpu_output, gpu_intra_band_delays, input_samples, sub_dedisp_samples_gpu, self._out_bands))
    else: 
      SubDedispGPUWarp((block_x, block_y), (thread_x, thread_y), (gpu_input, gpu_output, gpu_intra_band_delays, input_samples, sub_dedisp_samples_gpu, freq_avg))
    
    cp.cuda.Device(0).synchronize()
    sub_kernel_end = perf_counter()

    thread_y = self._out_bands
    thread_x = int(1024 / thread_y)

    block_x = int(full_dedisp_samples_gpu / thread_x)
    block_y = 1

    full_kernel_start = perf_counter()
    FullDedispGPU((block_x, block_y), (thread_x, thread_y), (gpu_output, gpu_inter_band_delays, sub_dedisp_samples_gpu, full_dedisp_samples_gpu, self._out_bands))
    cp.cuda.Device(0).synchronize()
    full_kernel_end = perf_counter()

    kernels_end = perf_counter()
    cpu_output = cp.asnumpy(gpu_output)

    dedisp_sub = reshape(cpu_output[:sub_dedisp_samples_gpu * self._out_bands], (self._out_bands, -1))[:, : sub_dedisp_samples_orig]
    dedisp_not_sum = reshape(cpu_output[sub_dedisp_samples_gpu * self._out_bands : sub_dedisp_samples_gpu * self._out_bands + full_dedisp_samples_gpu * self._out_bands], (self._out_bands, -1))[:, : full_dedisp_samples_orig]
    dedisp_full = cpu_output[sub_dedisp_samples_gpu * self._out_bands + full_dedisp_samples_gpu * self._out_bands : sub_dedisp_samples_gpu * self._out_bands + full_dedisp_samples_gpu * self._out_bands + full_dedisp_samples_orig]

    gpu_output = None
    gpu_input = None
    gpu_intra_band_delays = None
    gpu_inter_band_delays = None

    dedisp_end = perf_counter()

    logger.debug("Dedispersion took %.4fs", dedisp_end - dedisp_start)
    logger.debug("Kernels took %.4fs", kernels_end - kernels_start)
    logger.debug("Sub kernel took %.4fs", sub_kernel_end - sub_kernel_start)
    logger.debug("Full kernel took %.4fs", full_kernel_end - full_kernel_start)

    prep_start = perf_counter()

    # Prepare frequency ticks
    sf2_fmt = lambda x: "{:.2f}".format(x)
    avg_freq_pos = linspace(0, self._out_bands, num=5)
    avg_freq_pos[-1] = avg_freq_pos[-1] - 1       
    avg_freq_label = ftop + avg_freq_pos * avg_fband
    avg_freq_label_str = [sf2_fmt(label) for label in avg_freq_label]

    # Prepare the time ticks
    avg_time_pos = linspace(0, dedisp_sub.shape[1] - 1, num=5)
    avg_time_label = avg_time_pos * tsamp + skip_samples * tsamp + \
      ((cand_metadata["mjd"] - fil_metadata["mjd"]) * 86400 - plot_pad_s)
    avg_time_label_str = [sf2_fmt(label) for label in avg_time_label]

    avg_time_pos_orig = linspace(skip_samples, dedisp_sub.shape[1] - 1, num=5)

    avg_time_label_orig = (avg_time_pos_orig - skip_samples) * tsamp + \
                          ((cand_metadata["mjd"] - fil_metadata["mjd"]) * 86400
                          - plot_pad_s) + skip_samples * tsamp
    avg_time_label_orig_str = [sf2_fmt(abs(label)) for label in avg_time_label_orig]

    cmap = 'binary'

    fil_fig = plt.figure(figsize=(10, 7), frameon=True, dpi=100)
    fil_fig.tight_layout(h_pad=3.25, rect=[0, 0.03, 1, 0.95])

    plot_area = gs.GridSpec(2, 1)
    top_area = gs.GridSpecFromSubplotSpec(1, 5, subplot_spec=plot_area[0])
    bottom_area = gs.GridSpecFromSubplotSpec(1, 2, subplot_spec=plot_area[1])

    ax_spectrum = plt.Subplot(fil_fig, top_area[0, :-1])
    fil_fig.add_subplot(ax_spectrum)

    ax_band = plt.Subplot(fil_fig, top_area[0, -1])
    fil_fig.add_subplot(ax_band)

    ax_dedisp = plt.Subplot(fil_fig, bottom_area[0])
    fil_fig.add_subplot(ax_dedisp)

    ax_time = plt.Subplot(fil_fig, bottom_area[1])
    fil_fig.add_subplot(ax_time)

    fmtmjd = "{:.6f}".format(cand_metadata["mjd"])
    fmtsnr = "{:.2f}".format(cand_metadata["snr"])
    fmtdm = "{:.2f}".format(cand_metadata["dm"])
    fmtwidth = "{:.2f}".format(cand_metadata["width"])

    header = 'MJD: ' + fmtmjd + ', SNR: ' + fmtsnr + ', DM: ' + fmtdm + ', width: ' + fmtwidth + 'ms' + \
            ', avg: ' + str(time_avg) + 'T, ' + str(freq_avg) + 'F\n' \
            'Beam ' + str(beam_metadata["beam_abs"]) + beam_metadata["beam_type"] + ': RA ' + beam_metadata["beam_ra"] + ', Dec ' + beam_metadata["beam_dec"] + '      ' + fil_metadata['fil_file']

    ax_spectrum.imshow(dedisp_sub, interpolation='none', aspect='auto', cmap=cmap)
    ax_spectrum.plot(line_delays, arange(self._out_bands), linewidth=1.0, color='white')
    ax_spectrum.set_title(header, fontsize=9)
    ax_spectrum.set_xlabel('Time [s]', fontsize=8)
    ax_spectrum.set_ylabel('Freq [MHz]', fontsize=8)
    ax_spectrum.set_xticks(avg_time_pos_orig)
    ax_spectrum.set_xticklabels(avg_time_label_orig_str, fontsize=8)
    ax_spectrum.set_yticks(avg_freq_pos)
    ax_spectrum.set_yticklabels(avg_freq_label_str, fontsize=8)        

    ax_spectrum_orig = ax_spectrum.twiny()
    ax_spectrum_orig.set_xlim(ax_spectrum.get_xlim())
    ax_spectrum_orig.set_xticks(avg_time_pos)
    ax_spectrum_orig.set_xticklabels(avg_time_label_str, fontsize=8)

    sub_spectrum = npsum(dedisp_sub, axis=1) / dedisp_sub.shape[1]
    ax_band.plot(sub_spectrum, arange(sub_spectrum.shape[0]), color='black', linewidth=0.75) # pylint: disable=unsubscriptable-object
    ax_band.invert_yaxis()
    ax_band.set_xticks([npmin(sub_spectrum), mean(sub_spectrum), npmax(sub_spectrum)])
    ax_band.set_xticklabels(["{:.2f}".format(label) for label in [npmin(sub_spectrum), mean(sub_spectrum), npmax(sub_spectrum)]], fontsize=8)
    ax_band.yaxis.set_label_position("right")
    ax_band.set_ylim(ax_spectrum.get_ylim())
    ax_band.yaxis.tick_right()
    ax_band.set_title('Bandpass', fontsize=8)
    ax_band.set_yticks(avg_freq_pos)
    ax_band.set_yticklabels(avg_freq_label_str, fontsize=8)

    dedisp_time_pos = linspace(0, dedisp_full.shape[0] - 1, num=5)
    dedisp_time_label = dedisp_time_pos * tsamp + skip_samples * tsamp + ((cand_metadata["mjd"] - fil_metadata["mjd"]) * 86400 - plot_pad_s)        

    dedisp_time_pos_orig = linspace(skip_samples, dedisp_full.shape[0] - 1,
                                    num=5)
    dedisp_time_label_orig = (dedisp_time_pos_orig - skip_samples) * tsamp \
                            + ((cand_metadata["mjd"] - fil_metadata["mjd"])
                            * 86400 - plot_pad_s) + skip_samples * tsamp

    diff = dedisp_time_label[1] - dedisp_time_label[0]
    time_fmt = lambda x: "{:.{dec}f}".format(x, dec=total_decimals)
    total_decimals = 2

    time_unit = 's'

    # Need to do some rescaling to fit the time resolution into 2sf
    # This is a problem if we have no time averaging and are operating
    # at a natvie time resolution - then the time axis is just the same
    # number
    # Here we keep the unit as a second and just add extra decimal places
    if (diff < 0.01):
      # Check how much more we need to go
      # We can only shift the decimal place, so we're only interested in powers of 10
      log_diff = floor(log10(diff))
      # -2.0 is log10(0.01) as log_diff is smaller than that, this number will always be positive
      shift_decimals = int(-2.0 - log_diff)
      total_decimals = total_decimals + shift_decimals

    dedisp_time_label_str = [time_fmt(label) for label in dedisp_time_label]
    dedisp_time_label_orig_str = [time_fmt(label) for label in dedisp_time_label_orig]

    ax_dedisp.imshow(dedisp_not_sum, interpolation='none', aspect='auto', cmap=cmap)

    ax_dedisp.spines["bottom"].set_position(("axes", -0.15))
    ax_dedisp.set_xticks(dedisp_time_pos_orig)
    ax_dedisp.set_xticklabels(dedisp_time_label_orig_str, fontsize=8)
    ax_dedisp.set_xlabel('Time [' + time_unit + ']', fontsize=8)
    ax_dedisp.xaxis.set_label_coords(0.5, -0.025)
    ax_dedisp.tick_params(axis="x", direction="in", pad=-12)
    ax_dedisp.set_yticks(avg_freq_pos)
    ax_dedisp.set_ylabel('Freq [MHz]', fontsize=8)
    ax_dedisp.set_yticklabels(avg_freq_label_str, fontsize=8)

    ax_dedisp_orig = ax_dedisp.twiny()
    ax_dedisp_orig.set_xlim(ax_dedisp.get_xlim())
    ax_dedisp_orig.set_xticks(dedisp_time_pos)
    ax_dedisp_orig.set_xticklabels(dedisp_time_label_str, fontsize=8)

    ax_dedisp_off = ax_dedisp.twiny()
    ax_dedisp_off.set_xlim(ax_dedisp.get_xlim())
    ax_dedisp_off.spines["top"].set_position(("axes", -0.15))
    ax_dedisp_off.set_xticks(dedisp_time_pos_orig)

    dedisp_off_label = (dedisp_time_label_orig - median(dedisp_time_label_orig)) / disp_const
    dedisp_off_label_str = [time_fmt(label) for label in dedisp_off_label]

    ax_dedisp_off.set_xticklabels(dedisp_off_label_str, fontsize=8)
    ax_dedisp_off.set_xlabel(r"$\Delta$ DM", fontsize=8)
    ax_dedisp_off.xaxis.set_label_coords(0.5, -0.275)
    ax_dedisp_off.tick_params(axis="x", direction="in", pad=-15)

    dedisp_norm_pos = [0.0, 0.5, 1.0]
    dedisp_norm_label_sr = [sf2_fmt(label) for label in dedisp_norm_pos]

    # Normalise the 'power' to between 0 and 1
    dedisp_max = npmax(dedisp_full[:])
    dedisp_min = npmin(dedisp_full[:])
    dedisp_range = dedisp_max - dedisp_min
    # That would be some bad stuff happening
    if dedisp_range == 0.0:
      dedisp_range = 1
    dedisp_full = (dedisp_full - dedisp_min) / dedisp_range

    ax_time.plot(dedisp_full[:], linewidth=1.0, color='grey')
    ax_time.set_xlim([min(dedisp_time_label), max(dedisp_time_label)])
    ax_time.set_ylim([0.0, 1.0])
    ax_time.set_xticks(dedisp_time_pos_orig)
    ax_time.set_xticklabels(dedisp_time_label_orig_str, fontsize=8)
    ax_time.set_xlabel('Time [' + time_unit + ']', fontsize=8)
    ax_time.set_yticks(dedisp_norm_pos)
    ax_time.set_yticklabels(dedisp_norm_label_sr, fontsize=8)
    ax_time.set_ylabel('Norm power', fontsize=8)

    median_pos = median(dedisp_time_pos_orig)

    ax_time.axvline(median_pos, color="gray", linewidth=0.5)
    ax_time.axvline(median_pos - int(cand_metadata["width"] / 2.0 / 1000.0 / tsamp), color="gray", linewidth=0.5, linestyle="--")
    ax_time.axvline(median_pos + int(cand_metadata["width"] / 2.0 / 1000.0 / tsamp), color="gray", linewidth=0.5, linestyle="--")

    """
    Keep the pulse limits out for now - things do not align properly

    centre = int(round(dedisp_full.shape[0] / 2))
    half_width = int(round(cand_metadata["width"] * 1e-03 / 2 / tsamp))
    ax_time.axvline(centre, color='grey', linewidth=1.0)
    ax_time.axvline(centre + half_width, color='grey', linewidth=1.0, linestyle="--")
    ax_time.axvline(centre - half_width, color='grey', linewidth=1.0, linestyle="--")
    """

    ax_time_orig = ax_time.twiny()
    ax_time_orig.set_xticks(dedisp_time_pos)
    ax_time_orig.set_xticklabels(dedisp_time_label_str, fontsize=8)
    

    plt.text(0.75, 0.01, self._version, fontsize=8, in_layout=False, transform=plt.gcf().transFigure)
    plt.text(0.95, 0.01, " ".join(self._modules), horizontalalignment="right", weight="bold", fontsize=8, in_layout=False, transform=plt.gcf().transFigure)

    prep_end = perf_counter()
    logger.debug("Preparing the plot took %.4fs", prep_end - prep_start)
    
    plot_name = str(cand_metadata["mjd"]) + '_DM_' + fmtdm + '_beam_' + \
                str(beam_metadata["beam_abs"]) + beam_metadata["beam_type"] + '.jpg'

    save_start = perf_counter()

    if not path.isdir(path.join(fil_metadata["full_dir"], 'Plots')):
      try:
        mkdir(path.join(fil_metadata["full_dir"], 'Plots'))
      # That should never fire, unless there are some other scripts
      # running in the background that create this directory between
      # the `if` part and `mkdir` part
      except FileExistsError:
        pass

    fil_fig.savefig(path.join(fil_metadata["full_dir"], 'Plots', plot_name), transparent=False, backend="agg", bbox_inches = 'tight', quality=85)
    plt.close(fil_fig)
    save_end = perf_counter()
    logger.debug("Saving the plot took %.4fs", save_end - save_start)

    extra_file = path.join(fil_metadata["full_dir"], 'Plots', 'used_candidates.spccl.extra')

    with open(extra_file, 'a') as ef:
      ef.write("%d\t%.10f\t%.4f\t%.4f\t%.2f\t%d\t%s\t%s\t%s\t%s\t%s\n" % 
                (0, cand_metadata["mjd"], cand_metadata["dm"],
                cand_metadata["width"], cand_metadata["snr"],
                beam_metadata["beam_abs"], beam_metadata["beam_type"],
                beam_metadata['beam_ra'], beam_metadata["beam_dec"],
                fil_metadata["fil_file"], plot_name))
