import logging

from queue import PriorityQueue
from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from time import time

from astropy.time import Time
from prometheus_client import Counter, Summary

logger = logging.getLogger(__name__)

class CandidateManager(BaseManager):

  pass

class CandidateQueue(PriorityQueue):

  def __init__(self):
    super().__init__()
    self._lock = Lock()
    # Offset to add to the original MJD 'priority'
    # In this case it's the number of days to use for priority offset
    # This doesn't mean we will process with this MJD, it's just the
    # relative priority of the candidate that changes and MJD is used
    # as a simple way of implementing that
    self._priority_offset = 10000

    self._time_limit = 45

    self._priority = {
      "good": 0,
      "fall": 1,
      "rfi": 2
    }

    self._cand_summary = Summary("candidate_queue_size", "Number of candidates in the canddiate queue")
    self._rep_count = Counter("candidate_queue_repriorities", "Number of times the candidate queue was reprioritised")

  def put_candidate(self, candidate):
    with self._lock:
      self.put(candidate)

  def get_candidate(self):
      # This is always guaranteed to be the candidate with the lowest
      # MJD, i.e. the oldest candidate
      top_candidate = self.get()
      diff = time() - Time(top_candidate[1].metadata["cand_metadata"]["mjd"], format="mjd").unix
      if (diff >= self._time_limit) and (top_candidate[0] == self._priority["good"]):
        self.put(top_candidate)
        self.reprioritise_candidates()
        self._cand_summary.observe(self.qsize())
        return self.get()
      else:
        self._cand_summary.observe(self.qsize())
        return top_candidate

  def reprioritise_candidates(self):
    with self._lock:
      logger.info("Reprioritising the candidate queue")

      tmp = self.queue
      self.queue = []
      for cand in tmp:
        # Is there a way to bail early once we reach candidates that
        # do not need reprioritising?
        if ((time() - Time(cand[1].metadata["cand_metadata"]["mjd"], format="mjd").unix >= self._time_limit)
          and cand[0] == self._priority["good"]):
          cand = (self._priority["fall"], cand[1])

        self.put(cand)

    self._rep_count.inc()

CandidateManager.register("CandidateQueue", CandidateQueue)
    