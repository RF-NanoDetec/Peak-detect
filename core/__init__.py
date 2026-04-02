"""
Core modules for the Peak Analysis Tool.

These modules provide the fundamental data processing, analysis,
and manipulation capabilities for the application.
"""

# Import main functionality from modules
# Import new refactored modules
from core.data_analysis import (
    calculate_auto_threshold,
    calculate_peak_areas,
    calculate_peak_intervals,
)
from core.data_utils import (
    decimate_for_plot,
    find_nearest,
    get_width_range,
    reset_application_state,
    timestamps_to_seconds,
)
from core.file_handler import load_single_file
from core.peak_analysis_utils import (
    adjust_lowpass_cutoff,
    find_peaks_with_window,
    timestamps_array_to_seconds,
)
from core.peak_detection import PeakDetector
from core.performance import get_memory_usage, profile_function

# Define public API
__all__ = [
    # Data analysis functions
    "calculate_auto_threshold",
    "calculate_peak_areas",
    "calculate_peak_intervals",
    # Data utilities
    "decimate_for_plot",
    "find_nearest",
    "get_width_range",
    "reset_application_state",
    "timestamps_to_seconds",
    # File handling
    "load_single_file",
    # Peak analysis utilities
    "adjust_lowpass_cutoff",
    "find_peaks_with_window",
    "timestamps_array_to_seconds",
    # Peak detection
    "PeakDetector",
    # Performance utilities
    "get_memory_usage",
    "profile_function",
]
