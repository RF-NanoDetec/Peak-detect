"""
Caching utilities for service-layer optimization.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_auto_cutoff_cache: Dict[str, float] = {}
_auto_cutoff_cache_hits = 0
_auto_cutoff_cache_misses = 0

MAX_AUTO_CUTOFF_CACHE = 50


def get_auto_cutoff_result(result_id: str) -> Optional[float]:
    """
    Get cached auto-cutoff result if available.
    """
    global _auto_cutoff_cache_hits, _auto_cutoff_cache_misses

    if result_id in _auto_cutoff_cache:
        _auto_cutoff_cache_hits += 1
        logger.debug(
            f"Auto-cutoff cache HIT (id={result_id}, hits={_auto_cutoff_cache_hits}, misses={_auto_cutoff_cache_misses})"
        )
        return _auto_cutoff_cache[result_id]

    _auto_cutoff_cache_misses += 1
    logger.debug(
        f"Auto-cutoff cache MISS (id={result_id}, hits={_auto_cutoff_cache_hits}, misses={_auto_cutoff_cache_misses})"
    )
    return None


def cache_auto_cutoff_result(result_id: str, cutoff_freq: float) -> None:
    """
    Cache auto-cutoff result.
    """
    global _auto_cutoff_cache

    if len(_auto_cutoff_cache) >= MAX_AUTO_CUTOFF_CACHE:
        oldest_key = next(iter(_auto_cutoff_cache))
        del _auto_cutoff_cache[oldest_key]
        logger.debug(f"Auto-cutoff cache full, evicted key={oldest_key}")

    _auto_cutoff_cache[result_id] = cutoff_freq
    logger.debug(
        f"Auto-cutoff result cached (id={result_id}, cutoff={cutoff_freq}, size={len(_auto_cutoff_cache)})"
    )


def clear_all_caches() -> None:
    """Clear all caches."""
    global _auto_cutoff_cache_hits, _auto_cutoff_cache_misses

    _auto_cutoff_cache.clear()
    _auto_cutoff_cache_hits = 0
    _auto_cutoff_cache_misses = 0

    logger.info("All caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about cache usage."""
    return {
        "auto_cutoff": {
            "size": len(_auto_cutoff_cache),
            "max_size": MAX_AUTO_CUTOFF_CACHE,
            "hits": _auto_cutoff_cache_hits,
            "misses": _auto_cutoff_cache_misses,
            "hit_rate": _auto_cutoff_cache_hits
            / max(1, _auto_cutoff_cache_hits + _auto_cutoff_cache_misses),
        },
    }
