import threading
import requests as rs
from requests.adapters import HTTPAdapter

### REQUESTS HELPER ###

# The largest task-local pool is the six-category EDGE scrape.
SESSION_POOL_SIZE = 6
_thread_local = threading.local()

def make_pooled_session() -> rs.Session:
    session = rs.Session()
    adapter = HTTPAdapter(
        pool_connections=SESSION_POOL_SIZE,
        pool_maxsize=SESSION_POOL_SIZE,
        pool_block=True,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def pooled_session() -> rs.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = make_pooled_session()
        _thread_local.session = session
    return session


def get(url, session=None, **kwargs):
    """GET a URL using the supplied session or the thread-local default."""
    if session is None:
        session = pooled_session()
    return session.get(url, **kwargs)
