import unittest

from math import floor
from random import shuffle
from time import perf_counter, time

from astropy.time import Time

from spcandidate.candidate import Candidate
from spqueue.candidatequeue import CandidateQueue

candidates = 20

class CandidateQueueBasic(unittest.TestCase):

  def setUp(self) -> None:
      
    self.candidate_queue = CandidateQueue()

    for idx in range(candidates):

      cand_dict = {
        "data": None,
        "fil_metadata": None,
        "cand_metadata": {},
        "beam_metadata": None,
        "time": perf_counter()
      }

      cand_metadata = {
        "mjd": Time(time(), format="unix").mjd,
        "dm": 50.0 + idx,
        "width": 0.0,
        "snr": 0.0,
        "known": False
      }

      cand_dict["cand_metadata"] = cand_metadata

      self.candidate_queue.put_candidate((0, Candidate(cand_dict)))

  def test_get_candidates(self):

    received = 0

    while not self.candidate_queue.empty():
      self.candidate_queue.get_candidate()
      received += 1

    self.assertEqual(received, candidates)

class CandidateQueueOrder(unittest.TestCase):

  def setUp(self) -> None:
    self.candidate_queue = CandidateQueue()

  def test_alternate_ordering(self):

    for idx in range(candidates):
      cand_dict = {
        "data": None,
        "fil_metadata": None,
        "cand_metadata": {},
        "beam_metadata": None,
        "time": perf_counter()
      }
      cand_metadata = {
        "mjd": Time(time(), format="unix").mjd,
        "dm": 50.0 + idx,
        "width": 0.0,
        "snr": 0.0,
        "known": False
      }
      cand_dict["cand_metadata"] = cand_metadata
      self.candidate_queue.put_candidate((idx % 2, Candidate(cand_dict)))

    received = 0
    received_0 = 0
    received_1 = 0

    while not self.candidate_queue.empty():
      cand = self.candidate_queue.get_candidate()

      received += 1
      if cand[0] == 0:
        received_0 += 1
      else:
        received_1 += 1

      # The first half of the queue should have the priority number = 0
      # The second half of the queue should have the priority number = 1
      self.assertEqual(cand[0], (received - 1) // (candidates // 2))

    self.assertEqual(received, candidates)
    self.assertEqual(received_0, candidates // 2)
    self.assertEqual(received_1, candidates // 2)

  # Test single MJD ordering
  def test_mjd_ordering_1(self):

    # These mjds are not representative of real ones, but will behave
    # the same way - we just need values that are higher than the current
    # MJD which will prevent the reprioritising
    current_mjd = Time(time(), format="unix").mjd
    mjd_offset = floor(current_mjd) + 10

    mjds = [mjd_offset + 0.70 + idx / 86400 * (idx % 3 - 1) for idx in range(candidates)]

    for idx in range(candidates):
      cand_dict = {
        "data": None,
        "fil_metadata": None,
        "cand_metadata": {},
        "beam_metadata": None,
        "time": perf_counter()
      }

      cand_metadata = {
        "mjd": mjds[idx],
        "dm": 50.0 + idx,
        "width": 0.0,
        "snr": 0.0,
        "known": False
      }
      cand_dict["cand_metadata"] = cand_metadata
      self.candidate_queue.put_candidate((0, Candidate(cand_dict)))

    mjds_sorted = sorted(mjds)

    received = 0

    while not self.candidate_queue.empty():
      cand = self.candidate_queue.get_candidate()

      self.assertEqual(mjds_sorted[received], cand[1].metadata["cand_metadata"]["mjd"])
      received += 1
    
    self.assertEqual(received, candidates)

  # Test MJD ordering together with the alternating priority number
  def test_mjd_ordering_2(self):

    # These mjds are not representative of real ones, but will behave
    # the same way - we just need values that are higher than the current
    # MJD which will prevent the reprioritising

    current_mjd = Time(time(), format="unix").mjd
    mjd_offset = floor(current_mjd) + 10

    partial_mjds = [0.7, 0.7, 0.70002314814, 0.699965277774,
    0.7, 0.70005787037, 0.69993055555, 0.7,
    0.70009259259, 0.69989583333, 0.7, 0.70012731481,
    0.699861111105, 0.7, 0.700162037036, 0.69982638889,
    0.7, 0.70019675926, 0.699791666666, 0.7]
    
    mjds = [mjd_offset + mjd for mjd in partial_mjds]

    sorted_parial_mjds = [0.699791666666, 0.699861111105, 0.69993055555,
    0.7, 0.7, 0.7, 0.7, 0.70002314814, 0.70009259259,
    0.700162037036, 0.69982638889, 0.69989583333, 0.699965277774,
    0.7, 0.7, 0.7, 0.7, 0.70005787037, 0.70012731481,
    0.70019675926]

    sorted_mjds = [mjd_offset + mjd for mjd in sorted_parial_mjds]

    for idx in range(candidates):
      cand_dict = {
        "data": None,
        "fil_metadata": None,
        "cand_metadata": {},
        "beam_metadata": None,
        "time": perf_counter()
      }

      cand_metadata = {
        "mjd": mjds[idx],
        "dm": 50.0 + idx,
        "width": 0.0,
        "snr": 0.0,
        "known": False
      }
      cand_dict["cand_metadata"] = cand_metadata
      self.candidate_queue.put_candidate((idx % 2, Candidate(cand_dict)))

    received = 0
    received_0 = 0
    received_1 = 0

    while not self.candidate_queue.empty():
      cand = self.candidate_queue.get_candidate()

      # The first half of the queue should have the priority number = 0
      # The second half of the queue should have the priority number = 1
      self.assertEqual(cand[0], received // (candidates // 2))
      self.assertEqual(sorted_mjds[received], cand[1].metadata["cand_metadata"]["mjd"])

      received += 1
      if cand[0] == 0:
        received_0 += 1
      else:
        received_1 += 1

    self.assertEqual(received, candidates)
    self.assertEqual(received_0, candidates // 2)
    self.assertEqual(received_1, candidates // 2)

  # Test MJD and DM ordering
  def test_ordering_3(self):

    # These mjds are not representative of real ones, but will behave
    # the same way - we just need values that are higher than the current
    # MJD which will prevent the reprioritising
    current_mjd = Time(time(), format="unix").mjd
    mjd_offset = floor(current_mjd) + 10
    dm_offset = 50.0

    mjds = [mjd_offset + 0.70 + idx / 86400 * (idx % 3 - 1) for idx in range(candidates)]
    dms = [dm_offset + 0.23 * (idx % 3 - 1) for idx in range(candidates)]

    mjds_dms = [(mjds[idx], dms[idx]) for idx in range(candidates)]
    sorted_mjds_dms = sorted(mjds_dms)

    for idx in range(candidates):
      cand_dict = {
        "data": None,
        "fil_metadata": None,
        "cand_metadata": {},
        "beam_metadata": None,
        "time": perf_counter()
      }

      cand_metadata = {
        "mjd": mjds_dms[idx][0],
        "dm": mjds_dms[idx][1],
        "width": 0.0,
        "snr": 0.0,
        "known": False
      }
      cand_dict["cand_metadata"] = cand_metadata
      # Keep everything at the highest priority
      self.candidate_queue.put_candidate((0, Candidate(cand_dict)))

    received = 0

    while not self.candidate_queue.empty():
      cand = self.candidate_queue.get_candidate()
      # Don't test for priority number - everything has the highest one
      # Test for MJD and DM correctness
      self.assertEqual(sorted_mjds_dms[received][0], cand[1].metadata["cand_metadata"]["mjd"])
      self.assertEqual(sorted_mjds_dms[received][1], cand[1].metadata["cand_metadata"]["dm"])

      received += 1

    self.assertEqual(received, candidates)

  # Test basic reprioritisation
  def test_reprioritise_1(self):

    # These mjds are not representative of real ones, but will behave
    # the same way - we just need values that are higher than the current
    # MJD which will prevent the reprioritising

    current_mjd = Time(time(), format="unix").mjd
    # This MJD will definitely not get reprioritised
    mjd_high = floor(current_mjd) + 10
    # This MJD will definitely get reprioritised
    mjd_low = floor(current_mjd) - 10

    offsets = [mjd_low, mjd_high]

    mjds_low = [mjd_low + 0.70 + idx / 86400 * (idx % 3 - 1) for idx in range(candidates // 2)]
    mjds_high = [mjd_high + 0.70 + idx / 86400 * (idx % 3 - 1) for idx in range(candidates // 2)]

    sorted_mjds = sorted(mjds_high)
    sorted_mjds.extend(sorted(mjds_low))

    mjds = mjds_high.copy()
    mjds.extend(mjds_low)
    shuffle(mjds)
    
    for idx in range(candidates):
      cand_dict = {
        "data": None,
        "fil_metadata": None,
        "cand_metadata": {},
        "beam_metadata": None,
        "time": perf_counter()
      }

      cand_metadata = {
        "mjd": mjds[idx],
        "dm": 50.0 + idx,
        "width": 0.0,
        "snr": 0.0,
        "known": False
      }
      cand_dict["cand_metadata"] = cand_metadata
      self.candidate_queue.put_candidate((0, Candidate(cand_dict)))

    received = 0
    received_0 = 0
    received_1 = 0

    while not self.candidate_queue.empty():
      cand = self.candidate_queue.get_candidate()

      # The first half of the queue should have the priority number = 0
      # The second half of the queue should have the priority number = 1
      self.assertEqual(cand[0], received // (candidates // 2))
      self.assertEqual(sorted_mjds[received], cand[1].metadata["cand_metadata"]["mjd"])

      received += 1
      if cand[0] == 0:
        received_0 += 1
      else:
        received_1 += 1

    self.assertEqual(received, candidates)
    self.assertEqual(received_0, candidates // 2)
    self.assertEqual(received_1, candidates // 2)

if __name__ == "__main__":
  unittest.main()
