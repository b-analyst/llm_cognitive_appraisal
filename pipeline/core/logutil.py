"""Central logging configuration for CLI entry points."""
from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    """
    Configure root logging once (idempotent enough for repeated calls).
    Level from env PIPELINE_LOG_LEVEL (default INFO).
    """
    level_name = os.environ.get("PIPELINE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    root.setLevel(level)
