import logging

from os import path
from random import random
from typing import Dict

from astropy.coordinates import SkyCoord
from astropy.units import hourangle as ap_ha, deg as ap_deg

from psrmatch import Matcher

from spmodule.sptransform.transformmodule import TransformModule

logger = logging.getLogger(__name__)

class KnownModule(TransformModule):

  """

  Module responsible for matching candidates with known sources

  This module checking whether candidates passed from the watch module
  are found in the existing catalogues. This initial vetting removes
  known sources and hopefully reduces the strain on processing further
  down the line when the vicinity of a bright known source is observed.
  A small percentage of known sources is passed further for ML
  classification quality checks. These sources have extra metadata
  attached that changes the archiving behaviour.

  Parameters:

    config: Dict, default None
      Dictionary with module configuration parameters

  Attributes:

    _catalogue: str
      Catalogue used for candidate matching

    _known_pass_ratio: float
      Ratio of known sources to pass to further processing

    _matcher: Matcher
      Known source matcher

    _thresh_dist: float
      Matching distance threshold in degrees

    _thresh_dm: float
      Matching DM threshold as the percentage of detection DM

  """

  def __init__(self, config: Dict = None):

    super().__init__()
    self.id = 0
    self.type = "V"

    if (config == None) or not config:
      self._catalogue = "psrcat"
      self._thresh_dist = 1.5
      self._thresh_dm = 5.0
      self._known_pass_ratio = 0.005
      self._allowed_known = True
      self._allowed_known_file = "/config/allowed_known_sources.dat"
            
      try:
        self._allowed_known_list = open(self._allowed_known_file, 'r').read().splitlines()
      except:
        logger.error("Cannot load allowed known sources file %s! "
                      "Will use an empty list instead!", 
                      self._allowed_known_file)
        self._allowed_known_list = []
      else:
        logger.info("Loaded the allowed sources list with %d sources ",
                      len(self._allowed_known_list))
    else:
      self._catalogue = config["catalogue"]
      self._thresh_dist = config["thresh_dist"]
      self._thresh_dm = config["thresh_dm"]
      self._known_pass_ratio = config["known_pass_ratio"]

    self._matcher = Matcher(self._thresh_dist, self._thresh_dm)

    if self._catalogue not in self._matcher.supported_catalogues:

      logger.warning("Unsupported catalogue %s! "
                      "Will default to PSRCAT", 
                      self._catalogue)
      self._catalogue = "psrcat"

    self._matcher.load_catalogue(self._catalogue)
    self._matcher.create_search_tree()
    logger.info("Known source module initialised")

  async def process(self):

    """
    
    Matches the candidate to a known source

    If a match is found, the candidate is usually not passed further
    in the processing chain. A small percentage of known candidates
    is passed for quality checks. Additional metadata is added when
    this happens

    Parameters:

      None

    Returns:

      None if the candidate is to be processed further

      False if the candidate is not meant to be processed further and
      pipeline is to drop it from the execution.

    """

    beam_metadata = self._data.metadata["beam_metadata"]
    cand_metadata = self._data.metadata["cand_metadata"]

    beam_position = SkyCoord(ra = beam_metadata["beam_ra"],
                              dec = beam_metadata["beam_dec"],
                              frame = "icrs",
                              unit=(ap_ha, ap_deg))

    known_matches = self._matcher.find_matches(beam_position,
                                                cand_metadata["dm"])

    if (known_matches is not None and 
        not (self._allowed_known and 
          (known_matches[0] in self._allowed_known_list))):
      
      # Save the known source information
      # Separate file per beam for now
      fil_metadata = self._data.metadata["fil_metadata"]
      known_file = path.join(fil_metadata["full_dir"],
                              'known_sources.dat')
      with open(known_file, 'a') as kf:
          kf.write("%.10f\t%.4f\t%.4f\t%.2f\t%d\t%s\t%s\t%s\t%s\t%s\n" % 
          (cand_metadata["mjd"], cand_metadata["dm"],
          cand_metadata["width"], cand_metadata["snr"],
          beam_metadata["beam_abs"], beam_metadata["beam_type"],
          beam_metadata['beam_ra'], beam_metadata["beam_dec"],
          fil_metadata["fil_file"], known_matches[0]))

      if (random() <= self._known_pass_ratio):

        logger.info("This candidate is a known source")
        logger.info("It will be processed further")
        cand_metadata["known"] = known_matches[0]

      else:

        logger.info("This candidate is a known source")
        logger.info("It will not be processed further")

        return False

    else:

      if (known_matches is not None):

        logger.info("This candidate is an allowed known source."
                    "It wil be processed further")
        cand_metadata["known"] = known_matches[0]

        # Save the known source information
        # Separate file per beam for now
        fil_metadata = self._data.metadata["fil_metadata"]
        allowed_file = path.join(fil_metadata["full_dir"],
                              'allowed_known_sources.dat')      

        with open(allowed_file, 'a') as af:
          af.write("%.10f\t%.4f\t%.4f\t%.2f\t%d\t%s\t%s\t%s\t%s\t%s\n" % 
          (cand_metadata["mjd"], cand_metadata["dm"],
          cand_metadata["width"], cand_metadata["snr"],
          beam_metadata["beam_abs"], beam_metadata["beam_type"],
          beam_metadata['beam_ra'], beam_metadata["beam_dec"],
          fil_metadata["fil_file"], known_matches[0]))

      return None



