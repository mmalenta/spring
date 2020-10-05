import cupy as cp
import logging
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import pkg_resources

from numpy import arange, ceil, float32, floor, int32, linspace, log10, max as npmax, mean, min as npmin, newaxis, random, reshape, std, sum as npsum, zeros
from os import mkdir, path
from time import perf_counter
from typing import Dict

from spcandidate.candidate import Candidate as Cand
from spmodule.sputility.utilitymodule import UtilityModule

logger = logging.getLogger(__name__)

class PlotModule(UtilityModule):

  def __init__(self, config: Dict):
    super().__init__()

    self._plots = config["plots"]
    self._out_bands = config["out_chans"]

    # Normalise, just in case values do not add up to 1.0
    for row in self._plots:
      norm = sum([y for (x, y) in row])
      for idx, (x, y) in enumerate(row):
        row[idx] = (x, y / norm)

    logger.debug(f"Normalised plots structure: {self._plots}")

    self._version = pkg_resources.require("spring")[0].version

  def _pad_data(self, inputdata, fil_mjd,
                cand_dm, cand_mjd,
                ftop, fband, avg_fband, nchans, outbands,
                disp_const, plot_pad_s, tsamp_scaling, thread_x, thread_y):
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
        logger.debug("Not enough data at the start. "
                      + f"Padding with {zero_padding_samples_start} samples")

    # We don't have enough samples to cover the dispersive delay and padding at the end
    if (cand_samples_from_start + full_band_delay_samples + plot_padding_samples + last_band_delay_samples > original_data_length):
        zero_padding_samples_end = int(cand_samples_from_start + full_band_delay_samples + plot_padding_samples + last_band_delay_samples - original_data_length)
        logger.debug("Not enough data at the end. "
                      + f"Padding with {zero_padding_samples_end} samples")

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
    logger.debug(f"\tInput data length (original): {original_data_length}")
    logger.debug(f"\tInput data length (padded): {input_samples}")
    logger.debug("\tOutput plot samples: "
                  + f"{output_samples_sub_dedisp_orig} (subband), "
                  + f"{output_samples_full_dedisp_orig} (full)")
    logger.debug("\tWarp safe output plot samples: "
                  + f"{output_samples_sub_dedisp_warp_safe} (subband), "
                  + f"{output_samples_full_dedisp_warp_safe} (full)")
    logger.debug(f"\tDM: {dm:.2}")
    logger.debug(f"\tDM sweep seconds: {full_band_delay_seconds:.6f}")
    logger.debug(f"\tDM sweep samples: {full_band_delay_samples}")
    logger.debug(f"\tDelay across the last band: {last_band_delay_samples}")
    logger.debug(f"\tCandidate samples from the start: {cand_samples_from_start}")
    logger.debug(f"\tPadding at the start: {zero_padding_samples_start}")
    logger.debug(f"\tPadding at the end: {zero_padding_samples_end}")
    logger.debug(f"\tSamples skipped at the start: {plot_skip_samples}")
    logger.debug(f"\tWarp-safety padding: {warp_safety_samples}")

    return use_data, input_samples, output_samples_full_dedisp_orig, output_samples_full_dedisp_warp_safe, output_samples_sub_dedisp_orig, output_samples_sub_dedisp_warp_safe, start_padding_added

  async def plot(self, data: Cand) -> None:

    fil_metadata = data._metadata["fil_metadata"]
    cand_metadata = data._metadata["cand_metadata"]
    beam_metadata = data._metadata["beam_metadata"]

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
      self._pad_data(data._data, fil_metadata["mjd"],
                      cand_metadata['dm'], cand_metadata['mjd'], 
                      ftop, fband, avg_fband, nchans, self._out_bands,
                      disp_const, plot_pad_s, tsamp_scaling, thread_x, thread_y)

    SubDedispGPUHalf = cp.RawKernel(r'''
      // This version of the kernel assumes we use 16 channels per dedispersed subband 

      extern "C" __global__ void sub_dedisp_kernel_half(float* __restrict__ indata, float* __restrict__ outdata,
                                              int* __restrict__ intra_band_delays,
                                              int input_samples, int sub_dedisp_samples, int bands) {
          
          __shared__ float inchunk[32][32];
          
          if ((blockIdx.x * blockDim.x + threadIdx.x) < sub_dedisp_samples) {
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

    logger.debug(f"CPU input array shape: {use_data.shape}")
    logger.debug(f"GPU input array shape: {gpu_input.shape}")

    gpu_output = cp.zeros(sub_dedisp_samples_gpu * self._out_bands + full_dedisp_samples_gpu * self._out_bands + full_dedisp_samples_gpu, dtype=use_data.dtype)
    gpu_intra_band_delays = cp.asarray(cpu_intra_band_delays)
    gpu_inter_band_delays = cp.asarray(cpu_inter_band_delays)

    kernels_start = perf_counter()
    sub_kernel_start = perf_counter()
    if (freq_avg == 16):
      # We divide time sampels in .y direction as well
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

    logger.debug(f"Dedispersion took {(dedisp_end - dedisp_start):.4}s")
    logger.debug(f"Kernels took {(kernels_end - kernels_start):.4}s")
    logger.debug(f"Sub kernel took {(sub_kernel_end - sub_kernel_start):.4}s")
    logger.debug(f"Full kernel took {(full_kernel_end - full_kernel_start):.4}s")

    prep_start = perf_counter()

    # Prepare frequency ticks
    sf2_fmt = lambda x: "{:.2f}".format(x)
    avg_freq_pos = linspace(0, self._out_bands, num=5)
    avg_freq_pos[-1] = avg_freq_pos[-1] - 1       
    avg_freq_label = ftop + avg_freq_pos * avg_fband
    avg_freq_label_str = [sf2_fmt(label) for label in avg_freq_label]

    # Prepare the time ticks
    avg_time_pos = linspace(0, dedisp_sub.shape[1], num=5)
    avg_time_label = avg_time_pos * tsamp + skip_samples * tsamp + \
      ((cand_metadata["mjd"] - fil_metadata["mjd"]) * 86400 - plot_pad_s)
    avg_time_label_str = [sf2_fmt(label) for label in avg_time_label]

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
    ax_spectrum.set_xticks(avg_time_pos)
    ax_spectrum.set_xticklabels(avg_time_label_str, fontsize=8)
    ax_spectrum.set_yticks(avg_freq_pos)
    ax_spectrum.set_yticklabels(avg_freq_label_str, fontsize=8)        

    sub_spectrum = npsum(dedisp_sub, axis=1) / dedisp_sub.shape[1]
    ax_band.plot(sub_spectrum, arange(sub_spectrum.shape[0]), color='black', linewidth=0.75) # pylint: disable=unsubscriptable-object
    ax_band.invert_yaxis()
    ax_band.set_xticks([npmin(sub_spectrum), mean(sub_spectrum), npmax(sub_spectrum)])
    ax_band.set_xticklabels(["{:.2f}".format(label) for label in [npmin(sub_spectrum), mean(sub_spectrum), npmax(sub_spectrum)]], fontsize=8)
    ax_band.yaxis.set_label_position("right")
    ax_band.yaxis.tick_right()
    ax_band.set_title('Bandpass', fontsize=8)
    ax_band.set_yticks(avg_freq_pos)
    ax_band.set_yticklabels(avg_freq_label_str, fontsize=8)

    dedisp_time_pos = linspace(0, dedisp_full.shape[0] - 1, num=5)
    dedisp_time_label = dedisp_time_pos * tsamp + plot_pad_s + (cand_metadata["mjd"] - fil_metadata["mjd"]) * 86400.0
    dedisp_time_label = dedisp_time_pos * tsamp + skip_samples * tsamp + ((cand_metadata["mjd"] - fil_metadata["mjd"]) * 86400 - plot_pad_s)        

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

    ax_dedisp.imshow(dedisp_not_sum, interpolation='none', aspect='auto', cmap=cmap)
    ax_dedisp.set_xticks(dedisp_time_pos)
    ax_dedisp.set_xticklabels(dedisp_time_label_str, fontsize=8)
    ax_dedisp.set_xlabel('Time [' + time_unit + ']', fontsize=8)
    ax_dedisp.set_yticks(avg_freq_pos)
    ax_dedisp.set_ylabel('Freq [MHz]', fontsize=8)
    ax_dedisp.set_yticklabels(avg_freq_label_str, fontsize=8)

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
    ax_time.set_xticks(dedisp_time_pos)
    ax_time.set_xticklabels(dedisp_time_label_str, fontsize=8)
    ax_time.set_xlabel('Time [' + time_unit + ']', fontsize=8)
    ax_time.set_yticks(dedisp_norm_pos)
    ax_time.set_yticklabels(dedisp_norm_label_sr, fontsize=8)
    ax_time.set_ylabel('Norm power', fontsize=8)

    plt.text(0.05, 0.05, self._version, fontsize=8, in_layout=False, transform=plt.gcf().transFigure)
    plt.text(0.20, 0.05, "I Z", weight="bold", fontsize=8, in_layout=False, transform=plt.gcf().transFigure)

    prep_end = perf_counter()
    logger.debug(f"Preparing the plot took {(prep_end - prep_start):.4}s")
    
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
    logger.debug(f"Saving the plot took {(save_end - save_start):.4}s")

    extra_file = path.join(fil_metadata["full_dir"], 'Plots', 'used_candidates.spccl.extra')

    with open(extra_file, 'a') as ef:
      ef.write("%d\t%.10f\t%.4f\t%.4f\t%.2f\t%d\t%s\t%s\t%s\t%s\t%s\n" % 
                (0, cand_metadata["mjd"], cand_metadata["dm"],
                cand_metadata["width"], cand_metadata["snr"],
                beam_metadata["beam_abs"], beam_metadata["beam_type"],
                beam_metadata['beam_ra'], beam_metadata["beam_dec"],
                fil_metadata["fil_file"], plot_name))
