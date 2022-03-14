
from typing import Dict

from numpy import empty

class Candidate:

  """
  Class that wraps all the relevant candidate information
  in one place.

  This class encapsulates the data and all the metadata information
  required to process the candidate.

  Parameters:

    cand: Dict
      Candidate information passed by the watch module.

  Attributes:

    data : NumPy Array
      Data from the saved filterbank file. This data will be changed
      in-place by the subsequent piepline modules.

    metadata : Dict
      Metadata required to process the candidates. Contains filterbank
        metadata with all the filterbank header information, candidate
        metadata with all the candidate detection information and beam
        metadata with the information on the beam where tha candidate
        was detected.

  """

  def __init__(self, cand: Dict) -> None:

    self.mean = empty(0)
    self.stdev = empty(0)
    # Need to rename it to something more meaningful
    self.data = cand["data"]
    self.ml_cand = {
        "dmt": empty(0),
        "dedisp": empty(0),
    }
    self.metadata = {
        "fil_metadata": cand["fil_metadata"],
        "cand_metadata": cand["cand_metadata"],
        "beam_metadata": cand["beam_metadata"],
        "obs_metadata": cand["obs_metadata"],
    }
    self.time_added = cand["time"]
    
  def __lt__(self, other):

    sm = self.metadata["cand_metadata"]
    om = other.metadata["cand_metadata"]
    return (sm["mjd"], sm["dm"]) < (om["mjd"], om["dm"])