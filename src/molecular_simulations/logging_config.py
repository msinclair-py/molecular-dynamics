"""Logging configuration module for molecular_simulations.

This module provides opt-in logging configuration for applications, CLIs,
and examples. Library code should not call this implicitly.
"""

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
    """Configure logging for the application.

    Opt-in configuration for apps/CLIs/examples.
    Library code should *not* call this implicitly.

    Args:
        level: The logging level. Can be a string (e.g., 'INFO', 'DEBUG') or
            an integer. If None, uses the MS_LOG_LEVEL environment variable
            or defaults to 'INFO'.
        to_file: Path to a log file. If None, uses the MS_LOG_FILE environment
            variable. If neither is set, logs only to console.
        fmt: Log message format string. If None, uses the MS_LOG_FMT environment
            variable or a default format including hostname, PID, and MPI rank.

    Example:
        >>> configure_logging(level='DEBUG', to_file='simulation.log')
    """
    level = (level or os.getenv("MS_LOG_LEVEL") or "INFO")
    fmt = fmt or os.getenv("MS_LOG_FMT") or (
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s "
        "[host=%(hostname)s pid=%(process)d rank=%(mpirank)s]"
    )

    class _ContextFilter(logging.Filter):
        """Filter that adds hostname and MPI rank to log records."""

        def filter(self, record: logging.LogRecord) -> bool:
            """Add context information to the log record.

            Args:
                record: The log record to modify.

            Returns:
                Always returns True to allow the record to be logged.
            """
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
