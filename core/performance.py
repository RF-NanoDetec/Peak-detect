"""
Performance monitoring utilities for the Peak Analysis Tool.

This module provides functions for profiling, memory monitoring, and
performance optimization.
"""

import os
import time
import psutil
import logging
from pathlib import Path
from functools import wraps
from typing import Optional, Union

LoggerPath = Union[str, os.PathLike, Path]

_logger = logging.getLogger("peak_analysis.performance")
_logger.addHandler(logging.NullHandler())

_profiling_enabled = False
_log_handler: Optional[logging.Handler] = None
_log_path: Optional[Path] = None


def _remove_log_handler():
    global _log_handler, _log_path
    if _log_handler:
        _logger.removeHandler(_log_handler)
        _log_handler.close()
        _log_handler = None
        _log_path = None


def _configure_log_handler(log_file: Optional[LoggerPath]):
    global _log_handler, _log_path
    if not log_file:
        return

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if _log_handler and log_path == _log_path:
        return

    _remove_log_handler()

    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.DEBUG)

    _log_handler = handler
    _log_path = log_path


def set_profiling_enabled(enabled: bool, log_file: Optional[LoggerPath] = None) -> bool:
    """
    Enable or disable profiling across the application.

    Parameters
    ----------
    enabled : bool
        Whether profiling should be enabled.
    log_file : str or Path, optional
        Optional log file path. When provided, profiling output is written there.

    Returns
    -------
    bool
        The resulting profiling state.
    """
    global _profiling_enabled
    _profiling_enabled = bool(enabled)

    if _profiling_enabled:
        _logger.setLevel(logging.DEBUG)
        _configure_log_handler(log_file)
    else:
        _logger.setLevel(logging.CRITICAL)
        _remove_log_handler()

    return _profiling_enabled


def is_profiling_enabled() -> bool:
    """Return True if profiling is currently enabled."""
    return _profiling_enabled


def profiling_log(message: str):
    """Write a profiling/debug message when profiling is enabled."""
    if _profiling_enabled:
        _logger.debug(message)


def profile_function(func):
    """
    Decorator to profile function execution time and memory usage.

    Parameters
    ----------
    func : function
        Function to profile

    Returns
    -------
    function
        Wrapped function with profiling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _profiling_enabled:
            return func(*args, **kwargs)

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - diagnostic path
            elapsed_time = time.time() - start_time
            _logger.error(
                "ERROR in %s: %s (took %.2f seconds)", func.__name__, str(exc), elapsed_time
            )
            raise

        elapsed_time = time.time() - start_time
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = memory_after - memory_before

        _logger.debug(
            "%s took %.2f seconds and used %.1f MB of memory",
            func.__name__,
            elapsed_time,
            memory_diff,
        )

        return result

    return wrapper


def get_memory_usage():
    """
    Get current memory usage of the process.

    Returns
    -------
    float
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB
