"""New container entry point; the existing local factory remains untouched."""

import os

from .app import default_app
from .observability import ObservationSink, RequestObserver, configure_safe_logging


def create_observed_app(*, inner_app=None, sink=None):
    configure_safe_logging()
    if sink is None:
        sink = ObservationSink(file_path=os.environ.get("CUEORBIT_OBSERVATION_FILE"))
    try:
        inner = default_app() if inner_app is None else inner_app
    except Exception:
        raise RuntimeError("APPLICATION_INIT_FAILED") from None
    return RequestObserver(inner, sink)
