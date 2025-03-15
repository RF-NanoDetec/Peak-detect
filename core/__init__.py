"""
Core functionality for the Peak Analysis Tool.

This package contains the essential analysis and processing algorithms:
- peak_detection: Peak detection algorithms and filtering
- peak_analysis_utils: Utility functions for peak analysis
- data_processing: Core data processing functions
"""

# These imports will be uncommented as we move the modules
from .peak_detection import PeakDetector, calculate_auto_threshold
from .peak_analysis_utils import profile_function, find_peaks_with_window, find_nearest, timestamps_to_seconds, get_memory_usage, adjust_lowpass_cutoff
# from .peak_analysis_utils import * 