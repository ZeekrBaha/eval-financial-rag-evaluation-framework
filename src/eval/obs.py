"""obs.py — Observability layer for the eval pipeline.

Provides structured logging to both a run-specific file and stderr,
plus a simple timed() context manager for stage timing.

Public surface:
  configure_run_logging(run_dir, *, verbose) -> logging.Logger
  get_logger(name) -> logging.Logger
  timed(logger, label) — context manager: logs START + DONE with elapsed ms
"""

from __future__ import annotations

import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Generator


_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"

# Root logger name for the eval package.
_ROOT_NAME = "eval"


def configure_run_logging(run_dir: Path, *, verbose: bool = False) -> logging.Logger:
    """Configure the 'eval' logger for a single pipeline run.

    Sets up two handlers:
      - FILE handler  → run_dir/run.log, level DEBUG always.
      - STREAM handler → sys.stderr, level INFO (or DEBUG if verbose=True).

    Idempotent: clears existing handlers on the 'eval' logger before
    adding new ones, so repeated calls in tests don't duplicate output.

    Args:
        run_dir: Directory where run.log will be written. Must exist before
                 calling this function (create it first).
        verbose: When True the stream handler emits DEBUG messages too.

    Returns:
        The configured 'eval' logger.
    """
    logger = logging.getLogger(_ROOT_NAME)
    # Clear any existing handlers so repeated calls don't duplicate lines.
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT)

    # File handler — always DEBUG.
    log_path = run_dir / "run.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler — stderr only, INFO by default (DEBUG if verbose).
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Prevent propagation to the root logger so messages don't double-print.
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'eval' namespace.

    Args:
        name: Sub-module name, e.g. "run_eval", "metrics".

    Returns:
        logging.Logger named 'eval.<name>'.
    """
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


@contextlib.contextmanager
def timed(logger: logging.Logger, label: str) -> Generator[None, None, None]:
    """Context manager that logs 'START <label>' and 'DONE <label> in <ms> ms'.

    Uses time.monotonic for elapsed measurement. Both messages are logged at
    INFO level.

    Args:
        logger: Logger to emit messages to.
        label:  Human-readable stage label, e.g. "load_goldens".

    Example::

        with timed(log, "score_programmatic"):
            results = score_programmatic(records, goldens)
    """
    logger.info("START %s", label)
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info("DONE %s in %d ms", label, elapsed_ms)
