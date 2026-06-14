"""
utils/logger.py
---------------
Centralised logging factory for the COMSYS OCR baseline.

Usage
-----
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Hello")

Every module calls get_logger(__name__) so log lines carry the originating
module path automatically.  All loggers share one FileHandler so every stage
writes into the same run-log, while a StreamHandler provides colourised
console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# ── ANSI colour codes for the stream handler ──────────────────────────────────
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """Add ANSI colour to the level-name portion of the log record."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


# ── Module-level state (singleton file handler) ───────────────────────────────
_file_handler: Optional[logging.FileHandler] = None
_log_file_path: Optional[Path] = None


def setup_logging(
    log_dir: str = "logs",
    run_name: Optional[str] = None,
    level: int = logging.DEBUG,
) -> Path:
    """
    Initialise the shared file handler.  Call once per run (e.g. from main()).

    Parameters
    ----------
    log_dir   : Directory where log files are saved.
    run_name  : Optional prefix for the log filename.  Defaults to timestamp.
    level     : Minimum logging level written to the file.

    Returns
    -------
    Path to the created log file.
    """
    global _file_handler, _log_file_path

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{run_name}_" if run_name else ""
    log_file = log_dir_path / f"{prefix}{timestamp}.log"

    _log_file_path = log_file
    _file_handler = logging.FileHandler(log_file, encoding="utf-8")
    _file_handler.setLevel(level)
    _file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Attach to root so all child loggers inherit it
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(_file_handler)

    return log_file


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger with a colour stream handler.

    The file handler (if set up via setup_logging) is inherited from the root
    logger automatically.

    Parameters
    ----------
    name  : Logger name — use __name__ in every module.
    level : Minimum level for this specific logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate stream handlers on repeated calls
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(
            _ColourFormatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(stream_handler)

    # Prevent propagation from cluttering the root with duplicate stream lines
    logger.propagate = True  # file handler on root is still captured

    return logger


def get_log_file() -> Optional[Path]:
    """Return the currently active log file path (None if not set up yet)."""
    return _log_file_path
