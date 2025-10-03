"""
Application-wide logging configuration.
"""
from __future__ import annotations

import logging
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "ai_pipeline") -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        _LOGGER = logging.getLogger(name)
    return logging.getLogger(name)
