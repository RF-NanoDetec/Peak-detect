from __future__ import annotations

import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional


class InMemoryStore:
    def __init__(self):
        self._results: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._timings: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_results = max(1, int(os.getenv("PEAK_TOOL_MAX_RESULTS", "12")))
        self._result_ttl_seconds = max(
            60, int(os.getenv("PEAK_TOOL_RESULT_TTL_SECONDS", "7200"))
        )

    def _prune_locked(self):
        now = time.monotonic()
        expired_ids = [
            result_id
            for result_id, entry in self._results.items()
            if now - entry["last_accessed_at"] > self._result_ttl_seconds
        ]
        for result_id in expired_ids:
            self._results.pop(result_id, None)
            self._timings.pop(result_id, None)

        while len(self._results) > self._max_results:
            result_id, _entry = self._results.popitem(last=False)
            self._timings.pop(result_id, None)

    def save_result(self, payload: Dict[str, Any]) -> str:
        result_id = str(uuid.uuid4())
        with self._lock:
            now = time.monotonic()
            self._prune_locked()
            self._results[result_id] = {
                "payload": payload,
                "last_accessed_at": now,
            }
            self._results.move_to_end(result_id)
            self._prune_locked()
        return result_id

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._prune_locked()
            entry = self._results.get(result_id)
            if not entry:
                return None
            entry["last_accessed_at"] = time.monotonic()
            self._results.move_to_end(result_id)
            return entry["payload"]

    def save_timing(self, result_id: str, timing_summary: Dict[str, Any]):
        """Save timing data for a result ID."""
        with self._lock:
            self._timings[result_id] = timing_summary

    def get_timing(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get timing data for a result ID."""
        with self._lock:
            self._prune_locked()
            return self._timings.get(result_id)

    def get_all_timings(self) -> Dict[str, Dict[str, Any]]:
        """Get all timing data."""
        with self._lock:
            self._prune_locked()
            return self._timings.copy()

