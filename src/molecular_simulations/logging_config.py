from __future__ import annotations
import logging
import logging.config
import os
import socket
from datetime import datetime

def configure_logging(
    level: str | int | None = None,
    to_file: str | None = None,
    fmt: str | None = None,
) -> None:
    """
    Opt-in configuration for apps/CLIs/examples.
    Library code should *not* call this implicitly.
    """
    level = (level or os.getenv("MS_LOG_LEVEL") or "INFO")
    fmt = fmt or os.getenv("MS_LOG_FMT") or (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s "
        "[host=%(hostname)s pid=%(process)d rank=%(mpirank)s]"
    )

    class _ContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.hostname = socket.gethostname()
            # Fill MPI rank if available; else 0
            try:
                from mpi4py import MPI  # noqa: WPS433
                record.mpirank = MPI.COMM_WORLD.Get_rank()
            except Exception:
                record.mpirank = 0
            return True

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "filters": ["ctx"],
            "formatter": "standard",
        }
    }

    if to_file or os.getenv("MS_LOG_FILE"):
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "filters": ["ctx"],
            "formatter": "standard",
            "filename": to_file or os.getenv("MS_LOG_FILE"),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 3,
            "encoding": "utf-8",
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"ctx": {"()": _ContextFilter}},
        "formatters": {"standard": {"format": fmt}},
        "handlers": handlers,
        "root": {"level": level, "handlers": list(handlers)},
    }

    logging.config.dictConfig(config)

