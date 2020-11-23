
from typing import Dict

from numpy import empty

class Candidate:

  """
  Candidate class.

  This encapsulates the data and all the metadata information
  required to process the candidate.

  Attributes:
    data : Array
      Data from the saved filterbank file

    metadata : Dict
      Metadata required to process the candidates. Contains
      information on both the filterbank file (e.g. nchans, tsamp)
      and the candidate (e.g. DM, MJD)

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
    }
    self.time_added = cand["time"]
    