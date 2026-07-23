"""Centralized logger factory used across the project instead of print()."""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger with a single stream handler.

    Safe to call repeatedly for the same name; handlers are not duplicated
    on subsequent calls.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
