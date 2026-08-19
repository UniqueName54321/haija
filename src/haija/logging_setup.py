"""Logging setup for Haija CLI and GUI. Stdlib only."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def default_log_file() -> str:
    return str(Path.home() / ".haija" / "haija.log")


def setup_logging(level: str = "info", log_file: str = "") -> None:
    """Configure the root logger. Call once, early in ``main()``."""
    logger = logging.getLogger()
    logger.setLevel(_level(level))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    logger.handlers.clear()

    # File handler
    file_path = log_file or default_log_file()
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stderr handler (for CLI visibility)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def _level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)