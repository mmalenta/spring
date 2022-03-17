import logging
import tarfile

from os import path, remove
from socket import gethostname
from time import time
from typing import Dict

from spcandidate.candidate import Candidate as Cand
from spmodule.sputility.spoutput.outputmodule import OutputModule

logger = logging.getLogger(__name__)

class TarModule(OutputModule):

  """

  Module responsible for creating candidate tarballs.

  Tarballs contain basic information used for candidate inspection:
  * candidate plot
  * run_summary.json file
  * part of spccl file with relevant candidate information
  Tarballs are created with a name that contains the tatball creation time
  in ms and the node name. This prevents and is sufficient for name clashes
  (we are not going to created more than 1 tarballs per node per second).

  Parameters:

    config: Dict
      Currently a placeholder variable, not used

  Attributes:

    None

  """

  def __init__(self, config: Dict):

    super().__init__()

  def create_tarball(self, cand: Cand) -> None:

    """
    
    Creates and saves the tarball.

    Parameters:

      Parameters:

        cand: Dict
          Dictionary with all the necessary candidate information. 
          Contains the the array with the filterbank data, filterbank
          metadata with all the filterbank header information, candidate
          metadata with all the candidate detection information and beam
          metadata with the information on the beam where tha candidate
          was detected.

    Returns:

      None
    
    Raises:

      None (for now)

    """

    beam_metadata = cand.metadata["beam_metadata"]
    cand_metadata = cand.metadata["cand_metadata"]
    fil_metadata = cand.metadata["fil_metadata"]

    plots_dir = path.join(fil_metadata["full_dir"], "Plots")
    plot_file = cand_metadata["plot_name"]
    # Directory structure follows /<base>/<utc>/<beam> where <base> can 
    # generally be multiple directories
    utc_dir = fil_metadata["full_dir"].rstrip('/').split('/')[-2:-1][0]
    # Simply remove the beam directory from the end
    base_utc_dir = path.split(fil_metadata["full_dir"].rstrip('/'))[0]
    spccl_file = (utc_dir 
                  + "_beam{:0=2d}".format(beam_metadata["beam_rel"])
                  + ".spccl.log")
    try:

      # tpn-0-xx instead of tpn-0-xx.meertrap
      tpn_node = gethostname().split('.')[0]
      # This provides millisecond granularity
      timestamp = str(int(time() * 1000))
      tar_name = tpn_node + "_" + timestamp + ".tar"
      tar_file = tarfile.open(path.join(plots_dir, tar_name), 'w')
      # Add candidate plot
      tar_file.add(path.join(plots_dir, plot_file), plot_file)
      # Create and add candidate information file
      with open(path.join(plots_dir, spccl_file), 'w') as cand_file:
        
        cand_file.write("%d\t%.10f\t%.4f\t%.4f\t%.2f\t%d\t%s\t%s\t%s\t%d\t%.4f\t%s\t%s\n" %
                        (0, cand_metadata["mjd"], cand_metadata["dm"],
                        cand_metadata["width"], cand_metadata["snr"], 
                        beam_metadata["beam_abs"], beam_metadata["beam_type"],
                        beam_metadata["beam_ra"], beam_metadata["beam_dec"], 
                        cand_metadata["label"][0], cand_metadata["prob"][0],
                        fil_metadata["fil_file"], plot_file))
      tar_file.add(path.join(plots_dir, spccl_file), spccl_file)
      # Remove the candidate information file - we only need it for the tarball                  
      remove(path.join(plots_dir, spccl_file))
      # Add run_summary.json file
      tar_file.add(path.join(base_utc_dir, "run_summary.json"),
                    utc_dir + "_" + tpn_node + "_run_summary.json")
      tar_file.close()

    except:
      # TODO: better exceptions
      logger.error("An exception occured when creating a tarball!")
      pass