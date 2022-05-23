import logging

from prometheus_client import start_http_server, Counter, Info
from time import sleep

logger = logging.getLogger(__name__)

class PipelineMonitor:

  def __init__(self, queue):
    self._server_ip = 8000
    self._idx = 0
  
    self._request_counter = Counter("requests_issued", "Monitoring requests issued")
    self._monitoring_queue = queue

  def monitor(self):

    start_http_server(self._server_ip)

    while True:

      self._process_request()

  
  def _process_request(self):

    msg = self._monitoring_queue.get()
    logger.warning("Sent Prometheus request")