"""
Caching utilities for service layer optimization.

Provides in-memory LRU caches for expensive operations:
- Peak detection results
- Peak inspection images
- Auto-cutoff calculations
"""

import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Cache storage for peak detection results
_peak_detection_cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
_peak_detection_cache_hits = 0
_peak_detection_cache_misses = 0

# Cache storage for peak inspection images
_peak_inspection_cache: Dict[str, str] = {}
_peak_inspection_cache_hits = 0
_peak_inspection_cache_misses = 0

# Cache storage for auto-cutoff calculations
_auto_cutoff_cache: Dict[str, float] = {}
_auto_cutoff_cache_hits = 0
_auto_cutoff_cache_misses = 0

# Maximum cache sizes
MAX_PEAK_DETECTION_CACHE = 50
MAX_PEAK_INSPECTION_CACHE = 100
MAX_AUTO_CUTOFF_CACHE = 50


def compute_params_hash(*args, **kwargs) -> str:
    """
    Compute a hash for a set of parameters.
    
    Args:
        *args: Positional arguments to hash
        **kwargs: Keyword arguments to hash
        
    Returns:
        8-character hex hash string
    """
    # Build a string representation of all parameters
    parts = [str(arg) for arg in args]
    parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key = "_".join(parts)
    
    # Compute MD5 hash (fast, sufficient for cache keys)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_peak_detection_result(result_id: str, params: Dict[str, Any]) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Get cached peak detection result if available.
    
    Args:
        result_id: ID of the data result
        params: Detection parameters (prominence_threshold, distance, etc.)
        
    Returns:
        Tuple of (peaks, properties) if cached, None otherwise
    """
    global _peak_detection_cache_hits, _peak_detection_cache_misses
    
    cache_key = compute_params_hash(
        result_id,
        prominence_threshold=params.get('prominence_threshold'),
        distance=params.get('distance'),
        rel_height=params.get('rel_height'),
        width_ms=params.get('width_ms'),
        prominence_ratio=params.get('prominence_ratio'),
        time_resolution=params.get('time_resolution')
    )
    
    if cache_key in _peak_detection_cache:
        _peak_detection_cache_hits += 1
        logger.debug(f"Peak detection cache HIT (key={cache_key}, hits={_peak_detection_cache_hits}, misses={_peak_detection_cache_misses})")
        return _peak_detection_cache[cache_key]
    else:
        _peak_detection_cache_misses += 1
        logger.debug(f"Peak detection cache MISS (key={cache_key}, hits={_peak_detection_cache_hits}, misses={_peak_detection_cache_misses})")
        return None


def cache_peak_detection_result(result_id: str, params: Dict[str, Any], peaks: np.ndarray, properties: Dict[str, Any]) -> None:
    """
    Cache peak detection result.
    
    Args:
        result_id: ID of the data result
        params: Detection parameters
        peaks: Detected peak indices
        properties: Peak properties dictionary
    """
    global _peak_detection_cache
    
    cache_key = compute_params_hash(
        result_id,
        prominence_threshold=params.get('prominence_threshold'),
        distance=params.get('distance'),
        rel_height=params.get('rel_height'),
        width_ms=params.get('width_ms'),
        prominence_ratio=params.get('prominence_ratio'),
        time_resolution=params.get('time_resolution')
    )
    
    # Implement LRU eviction if cache is full
    if len(_peak_detection_cache) >= MAX_PEAK_DETECTION_CACHE:
        # Remove oldest entry (simple FIFO for now)
        oldest_key = next(iter(_peak_detection_cache))
        del _peak_detection_cache[oldest_key]
        logger.debug(f"Peak detection cache full, evicted key={oldest_key}")
    
    _peak_detection_cache[cache_key] = (peaks, properties)
    logger.debug(f"Peak detection result cached (key={cache_key}, size={len(_peak_detection_cache)})")


def get_peak_inspection_image(result_id: str, filtered_result_id: str, offset: int, params: Dict[str, Any]) -> Optional[str]:
    """
    Get cached peak inspection image if available.
    
    Args:
        result_id: ID of the original data result
        filtered_result_id: ID of the filtered data result
        offset: Peak offset for pagination
        params: Detection parameters
        
    Returns:
        Base64 encoded image string if cached, None otherwise
    """
    global _peak_inspection_cache_hits, _peak_inspection_cache_misses
    
    cache_key = compute_params_hash(
        result_id,
        filtered_result_id,
        offset,
        prominence_threshold=params.get('prominence_threshold'),
        distance=params.get('distance'),
        rel_height=params.get('rel_height'),
        width_ms=params.get('width_ms'),
        prominence_ratio=params.get('prominence_ratio'),
        time_resolution=params.get('time_resolution')
    )
    
    if cache_key in _peak_inspection_cache:
        _peak_inspection_cache_hits += 1
        logger.debug(f"Peak inspection cache HIT (key={cache_key}, hits={_peak_inspection_cache_hits}, misses={_peak_inspection_cache_misses})")
        return _peak_inspection_cache[cache_key]
    else:
        _peak_inspection_cache_misses += 1
        logger.debug(f"Peak inspection cache MISS (key={cache_key}, hits={_peak_inspection_cache_hits}, misses={_peak_inspection_cache_misses})")
        return None


def cache_peak_inspection_image(result_id: str, filtered_result_id: str, offset: int, params: Dict[str, Any], image: str) -> None:
    """
    Cache peak inspection image.
    
    Args:
        result_id: ID of the original data result
        filtered_result_id: ID of the filtered data result
        offset: Peak offset for pagination
        params: Detection parameters
        image: Base64 encoded image string
    """
    global _peak_inspection_cache
    
    cache_key = compute_params_hash(
        result_id,
        filtered_result_id,
        offset,
        prominence_threshold=params.get('prominence_threshold'),
        distance=params.get('distance'),
        rel_height=params.get('rel_height'),
        width_ms=params.get('width_ms'),
        prominence_ratio=params.get('prominence_ratio'),
        time_resolution=params.get('time_resolution')
    )
    
    # Implement LRU eviction if cache is full
    if len(_peak_inspection_cache) >= MAX_PEAK_INSPECTION_CACHE:
        # Remove oldest entry (simple FIFO for now)
        oldest_key = next(iter(_peak_inspection_cache))
        del _peak_inspection_cache[oldest_key]
        logger.debug(f"Peak inspection cache full, evicted key={oldest_key}")
    
    _peak_inspection_cache[cache_key] = image
    logger.debug(f"Peak inspection image cached (key={cache_key}, size={len(_peak_inspection_cache)})")


def get_auto_cutoff_result(result_id: str) -> Optional[float]:
    """
    Get cached auto-cutoff result if available.
    
    Args:
        result_id: ID of the data result
        
    Returns:
        Cutoff frequency if cached, None otherwise
    """
    global _auto_cutoff_cache_hits, _auto_cutoff_cache_misses
    
    if result_id in _auto_cutoff_cache:
        _auto_cutoff_cache_hits += 1
        logger.debug(f"Auto-cutoff cache HIT (id={result_id}, hits={_auto_cutoff_cache_hits}, misses={_auto_cutoff_cache_misses})")
        return _auto_cutoff_cache[result_id]
    else:
        _auto_cutoff_cache_misses += 1
        logger.debug(f"Auto-cutoff cache MISS (id={result_id}, hits={_auto_cutoff_cache_hits}, misses={_auto_cutoff_cache_misses})")
        return None


def cache_auto_cutoff_result(result_id: str, cutoff_freq: float) -> None:
    """
    Cache auto-cutoff result.
    
    Args:
        result_id: ID of the data result
        cutoff_freq: Calculated cutoff frequency
    """
    global _auto_cutoff_cache
    
    # Implement LRU eviction if cache is full
    if len(_auto_cutoff_cache) >= MAX_AUTO_CUTOFF_CACHE:
        # Remove oldest entry (simple FIFO for now)
        oldest_key = next(iter(_auto_cutoff_cache))
        del _auto_cutoff_cache[oldest_key]
        logger.debug(f"Auto-cutoff cache full, evicted key={oldest_key}")
    
    _auto_cutoff_cache[result_id] = cutoff_freq
    logger.debug(f"Auto-cutoff result cached (id={result_id}, cutoff={cutoff_freq}, size={len(_auto_cutoff_cache)})")


def clear_all_caches() -> None:
    """Clear all caches (useful for testing or memory management)."""
    global _peak_detection_cache, _peak_inspection_cache, _auto_cutoff_cache
    global _peak_detection_cache_hits, _peak_detection_cache_misses
    global _peak_inspection_cache_hits, _peak_inspection_cache_misses
    global _auto_cutoff_cache_hits, _auto_cutoff_cache_misses
    
    _peak_detection_cache.clear()
    _peak_inspection_cache.clear()
    _auto_cutoff_cache.clear()
    
    _peak_detection_cache_hits = 0
    _peak_detection_cache_misses = 0
    _peak_inspection_cache_hits = 0
    _peak_inspection_cache_misses = 0
    _auto_cutoff_cache_hits = 0
    _auto_cutoff_cache_misses = 0
    
    logger.info("All caches cleared")


def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about cache usage."""
    return {
        "peak_detection": {
            "size": len(_peak_detection_cache),
            "max_size": MAX_PEAK_DETECTION_CACHE,
            "hits": _peak_detection_cache_hits,
            "misses": _peak_detection_cache_misses,
            "hit_rate": _peak_detection_cache_hits / max(1, _peak_detection_cache_hits + _peak_detection_cache_misses)
        },
        "peak_inspection": {
            "size": len(_peak_inspection_cache),
            "max_size": MAX_PEAK_INSPECTION_CACHE,
            "hits": _peak_inspection_cache_hits,
            "misses": _peak_inspection_cache_misses,
            "hit_rate": _peak_inspection_cache_hits / max(1, _peak_inspection_cache_hits + _peak_inspection_cache_misses)
        },
        "auto_cutoff": {
            "size": len(_auto_cutoff_cache),
            "max_size": MAX_AUTO_CUTOFF_CACHE,
            "hits": _auto_cutoff_cache_hits,
            "misses": _auto_cutoff_cache_misses,
            "hit_rate": _auto_cutoff_cache_hits / max(1, _auto_cutoff_cache_hits + _auto_cutoff_cache_misses)
        }
    }


