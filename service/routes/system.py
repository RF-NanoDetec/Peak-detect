from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request

from service import cache as cache_module
from service.models import Params
from service.storage import InMemoryStore

from .dependencies import get_store

router = APIRouter()


@router.get("/cache/stats")
def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring performance improvements."""
    return cache_module.get_cache_stats()


@router.delete("/cache/clear")
def clear_caches() -> Dict[str, str]:
    """Clear all caches (useful for testing or memory management)."""
    cache_module.clear_all_caches()
    return {"message": "All caches cleared successfully"}


@router.get("/params", response_model=Params)
def get_params(request: Request) -> Params:
    return getattr(request.app.state, "params_default", Params())


@router.put("/params", response_model=Params)
def put_params(new_params: Params, request: Request) -> Params:
    request.app.state.params_default = new_params
    return new_params


@router.get("/performance/timing/{result_id}")
def get_timing_report(
    result_id: str, store: InMemoryStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Get performance timing report for a specific result ID.

    Returns detailed timing breakdown for data loading, filtering, or peak detection.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Getting timing data for result_id: {result_id}")

    timing = store.get_timing(result_id)
    if not timing:
        all_timings = store.get_all_timings()
        logger.debug(f"Available timing IDs: {list(all_timings.keys())}")
        raise HTTPException(
            status_code=404,
            detail=f"Timing data not found for result ID: {result_id}. Available IDs: {list(all_timings.keys())[:5]}",
        )
    return timing


@router.get("/performance/timing")
def get_all_timing_reports(
    store: InMemoryStore = Depends(get_store),
) -> Dict[str, Dict[str, Any]]:
    """
    Get all performance timing reports.

    Returns a dictionary mapping result IDs to their timing summaries.
    """
    logger = logging.getLogger(__name__)
    all_timings = store.get_all_timings()
    logger.debug(
        f"Returning {len(all_timings)} timing reports: {list(all_timings.keys())}"
    )
    return all_timings


@router.get("/performance/debug")
def debug_timing(store: InMemoryStore = Depends(get_store)) -> Dict[str, Any]:
    """Debug endpoint to check timing storage."""
    all_timings = store.get_all_timings()
    return {
        "total_timings": len(all_timings),
        "timing_ids": list(all_timings.keys()),
        "sample": all_timings.get(list(all_timings.keys())[0]) if all_timings else None,
    }
