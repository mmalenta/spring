from typing import Dict

class Candidate:

    """
    Candidate class.

    This encapsulates the data and all the metadata information
    required to process the candidate.

    Attributes:

        _data : Array
            Data from the saved filterbank file

        _metadata : Dict
            Metadata required to process the candidates. Contains
            information on both the filterbank file (e.g. nchans, tsamp)
            and the candidate (e.g. DM, MJD)

    """

    def __init__(self, cand: Dict) -> None:

        self._data = cand["data"]
        self._metadata = {
            "fil_metadata": cand["fil_metadata"],
            "cand_metadata": cand["cand_metadata"],
        }
        self._time_added = cand["time"]