import threading
import requests as rs
from requests.adapters import HTTPAdapter

### REQUESTS HELPER ###

POOL_WORKERS = 2
_thread_local = threading.local()

def make_pooled_session() -> rs.Session:
    workers = max(1, int(POOL_WORKERS))
    session = rs.Session()
    adapter = HTTPAdapter(
        pool_connections=workers,
        pool_maxsize=workers,
        pool_block=True,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def pooled_session() -> rs.Session:
    workers = max(1, int(POOL_WORKERS))
    session = getattr(_thread_local, "session", None)
    if session is None or getattr(_thread_local, "pool_workers", None) != workers:
        if session is not None:
            session.close()
        session = make_pooled_session()
        _thread_local.session = session
        _thread_local.pool_workers = workers
    return session


def get(url, session=None, **kwargs):
    """GET a URL using the supplied session or the thread-local default."""
    if session is None:
        session = pooled_session()
    return session.get(url, **kwargs)
