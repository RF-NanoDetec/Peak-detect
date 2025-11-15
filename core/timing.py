"""
Performance timing utilities for detailed operation profiling.

This module provides context managers and utilities to track timing
for different phases of data processing: loading, filtering, peak detection, etc.
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading

_logger = logging.getLogger(__name__)

class TimingTracker:
    """
    Tracks timing for different operations and provides summary reports.
    """
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        self.total_start_time: Optional[float] = None
        self.total_end_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def start(self, phase: Optional[str] = None):
        """Start timing the overall operation."""
        self.total_start_time = time.time()
        if phase:
            self.start_phase(phase)
    
    def stop(self):
        """Stop timing the overall operation."""
        self.total_end_time = time.time()
    
    @contextmanager
    def phase(self, phase_name: str):
        """Context manager for timing a specific phase."""
        self.start_phase(phase_name)
        try:
            yield
        finally:
            self.end_phase(phase_name)
    
    def start_phase(self, phase_name: str):
        """Start timing a specific phase."""
        now = time.time()
        with self._lock:
            self.start_times[phase_name] = now
    
    def end_phase(self, phase_name: str):
        """End timing a specific phase and record the duration."""
        with self._lock:
            start_time = self.start_times.pop(phase_name, None)
            if start_time is None:
                _logger.warning(f"Phase '{phase_name}' was ended without being started")
                return
            elapsed = time.time() - start_time
            self.timings[phase_name].append(elapsed)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the timing report."""
        with self._lock:
            self.metadata[key] = value
    
    def get_total_time(self) -> float:
        """Get total elapsed time."""
        if self.total_start_time is None:
            return 0.0
        end = self.total_end_time if self.total_end_time else time.time()
        return end - self.total_start_time
    
    def get_phase_times(self) -> Dict[str, float]:
        """Get average time for each phase."""
        with self._lock:
            timings_copy = {phase: list(times) for phase, times in self.timings.items()}
        return {
            phase: sum(times) / len(times) if times else 0.0
            for phase, times in timings_copy.items()
        }
    
    def get_phase_total_times(self) -> Dict[str, float]:
        """Get total time for each phase (sum of all calls)."""
        with self._lock:
            timings_copy = {phase: list(times) for phase, times in self.timings.items()}
        return {
            phase: sum(times)
            for phase, times in timings_copy.items()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all timings."""
        total_time = self.get_total_time()
        phase_times = self.get_phase_total_times()
        
        # Calculate percentages
        percentages = {}
        if total_time > 0:
            percentages = {
                phase: (time / total_time) * 100
                for phase, time in phase_times.items()
            }
        
        # Find the slowest phase
        slowest_phase = None
        slowest_time = 0.0
        if phase_times:
            slowest_phase = max(phase_times.items(), key=lambda x: x[1])[0]
            slowest_time = phase_times[slowest_phase]
        
        summary = {
            "operation": self.operation_name,
            "total_time_seconds": total_time,
            "phases": {},
            "slowest_phase": {
                "name": slowest_phase,
                "time_seconds": slowest_time,
                "percentage": percentages.get(slowest_phase, 0.0) if slowest_phase else 0.0
            },
            "metadata": {}
        }
        
        with self._lock:
            for phase, time_value in phase_times.items():
                summary["phases"][phase] = {
                    "total_seconds": time_value,
                    "percentage": percentages.get(phase, 0.0),
                    "call_count": len(self.timings.get(phase, []))
                }
            summary["metadata"] = dict(self.metadata)
        
        return summary
    
    def get_formatted_summary(self) -> str:
        """Get a human-readable formatted summary."""
        summary = self.get_summary()
        lines = [
            f"\n{'='*60}",
            f"Performance Timing Report: {summary['operation']}",
            f"{'='*60}",
            f"Total Time: {summary['total_time_seconds']:.3f} seconds",
            "",
            "Phase Breakdown:"
        ]
        
        # Sort phases by time (descending)
        sorted_phases = sorted(
            summary['phases'].items(),
            key=lambda x: x[1]['total_seconds'],
            reverse=True
        )
        
        for phase, data in sorted_phases:
            lines.append(
                f"  {phase:30s} {data['total_seconds']:8.3f}s "
                f"({data['percentage']:5.1f}%) [{data['call_count']} calls]"
            )
        
        if summary['slowest_phase']['name']:
            lines.append("")
            lines.append(
                f"Slowest Phase: {summary['slowest_phase']['name']} "
                f"({summary['slowest_phase']['time_seconds']:.3f}s, "
                f"{summary['slowest_phase']['percentage']:.1f}%)"
            )
        
        if summary['metadata']:
            lines.append("")
            lines.append("Metadata:")
            for key, value in summary['metadata'].items():
                lines.append(f"  {key}: {value}")
        
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)
    
    def log_summary(self, logger: Optional[logging.Logger] = None):
        """Log the formatted summary."""
        log = logger or _logger
        log.info(self.get_formatted_summary())


# Global tracker instance for easy access
_current_tracker: Optional[TimingTracker] = None

def get_tracker() -> Optional[TimingTracker]:
    """Get the current global timing tracker."""
    return _current_tracker

def set_tracker(tracker: Optional[TimingTracker]):
    """Set the current global timing tracker."""
    global _current_tracker
    _current_tracker = tracker

def create_tracker(operation_name: str) -> TimingTracker:
    """Create and set a new timing tracker."""
    tracker = TimingTracker(operation_name)
    set_tracker(tracker)
    return tracker


