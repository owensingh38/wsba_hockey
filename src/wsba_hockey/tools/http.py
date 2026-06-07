import threading
import requests as rs
from requests.adapters import HTTPAdapter

### REQUESTS HELPER ###

POOL_WORKERS = 4
_thread_local = threading.local()

def make_pooled_session() -> rs.Session:
    session = rs.Session()
    adapter = HTTPAdapter(
        pool_connections=POOL_WORKERS,
        pool_maxsize=POOL_WORKERS,
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


def get(url, **kwargs):
    return pooled_session().get(url, **kwargs)
