#from asyncio import Queue as AQueue
from multiprocessing import get_context
from multiprocessing.queues import Queue as MPQueue

class CandidateQueue(MPQueue):

    """

    Candidate queue class.

    Currently does nothing beyond just extending the asynchronous
    queue. Additional functionality will be added in the future.

    """

    def __init__(self) -> None:
        ctx = get_context()
        super().__init__(ctx=ctx)