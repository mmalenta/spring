from os import path
from typing import Dict

import h5py as h5
from numpy import frombuffer, uint8

from spcandidate.candidate import Candidate as Cand
from spmodule.sputility.utilitymodule import UtilityModule

class ArchiveModule(UtilityModule):

  """ 

  Module responsible for archiving candidates that pass earlier
  processing stages.

  Picks up candidates that make it to the archiving queue. Creates an
  HDF5 archive with the most important information and data for any
  potential further candidate processing. Currently all the candidates,
  are threated equally, but the functionality of this module will change
  in the future to distinguish between negative and positive candidates
  as well as include known sources. 


  The archive follows a specific format which can be found under
  https://tinyurl.com/meertrap-hdf5 (starts a PDF document download).
  Tools for viewing the archive can be found under 
  https://github.com/mmalenta/mtah5

  Parameters:

    config: Dict
      Currently a dummy variable, not used

  Attributes:

    None

  """

  def __init__(self, config: Dict):
    super().__init__()

  async def archive(self, cand: Cand) -> None:

    """

    Creates and saves the HDF5 archive.

    Filterbank data currently not saved.

    Parameters:

      cand: Dict
        Dictionary with all the necessary candidate information. 
        Contains the the array with the filterbank data, filterbank
        metadata with all the filterbanjk header information, candidate
        metadata with all the candidate detection information and beam
        metadata with the information on the beam where tha candidate
        was detected.

    Returns:

      None

    """

    beam_metadata = cand.metadata["beam_metadata"]
    cand_metadata = cand.metadata["cand_metadata"]
    fil_metadata = cand.metadata["fil_metadata"]

    fmtdm = "{:.2f}".format(cand_metadata["dm"]) 
    file_name = str(cand_metadata["mjd"]) + '_DM_' + fmtdm + '_beam_' + \
                str(beam_metadata["beam_abs"]) + beam_metadata["beam_type"] + '_frbid.hdf5'

    with h5.File(path.join(fil_metadata["full_dir"], file_name), 'w') as h5f:

      cand_group = h5f.create_group("/cand")
      # /cand/detection
      detection_group = cand_group.create_group("detection")
      # /cand/detection/plot
      plot_group = detection_group.create_group("plot")
      # /cand/ml
      ml_group = cand_group.create_group("ml")

      detection_group.attrs["filterbank"] = fil_metadata["fil_file"]
      detection_group.attrs["mjd"] = cand_metadata["mjd"]
      detection_group.attrs["dm"] = cand_metadata["dm"]
      detection_group.attrs["snr"] = cand_metadata["snr"]
      detection_group.attrs["width"] = cand_metadata["width"]
      detection_group.attrs["beam"] = beam_metadata["beam_abs"]
      detection_group.attrs["beam_type"] = beam_metadata["beam_type"]
      detection_group.attrs["ra"] = beam_metadata["beam_ra"]
      detection_group.attrs["dec"] = beam_metadata["beam_dec"]

      plot_name = str(cand_metadata["mjd"]) + '_DM_' + fmtdm + '_beam_' + \
                  str(beam_metadata["beam_abs"]) + beam_metadata["beam_type"] + '.jpg'

      plot_group.attrs["plot_name"] = plot_name
      plot_group.attrs["representation"] = "uint8"      

      # pylint: disable=unused-variable
      with open(path.join(fil_metadata["full_dir"], 'Plots', plot_name), 'rb') as plot_file:
        binary_plot = plot_file.read()
        binary_plot_array = frombuffer(binary_plot, dtype=uint8)
        plot_dataset = plot_group.create_dataset('jpg', data=binary_plot_array, dtype=uint8)
      ml_group.attrs["label"] = cand_metadata["label"]
      ml_group.attrs["prob"] = cand_metadata["prob"]

      dm_time_dataset = ml_group.create_dataset("dm_time",
                                                data=cand.ml_cand["dmt"])
      freq_time_dataset = ml_group.create_dataset("freq_time",
                                                  data=cand.ml_cand["dedisp"])