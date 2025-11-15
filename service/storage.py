from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, Optional


class InMemoryStore:
    def __init__(self):
        self._results: Dict[str, Dict[str, Any]] = {}
        self._timings: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def save_result(self, payload: Dict[str, Any]) -> str:
        result_id = str(uuid.uuid4())
        with self._lock:
            self._results[result_id] = payload
        return result_id

    def get_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._results.get(result_id)
    
    def save_timing(self, result_id: str, timing_summary: Dict[str, Any]):
        """Save timing data for a result ID."""
        with self._lock:
            self._timings[result_id] = timing_summary
    
    def get_timing(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Get timing data for a result ID."""
        with self._lock:
            return self._timings.get(result_id)
    
    def get_all_timings(self) -> Dict[str, Dict[str, Any]]:
        """Get all timing data."""
        with self._lock:
            return self._timings.copy()


