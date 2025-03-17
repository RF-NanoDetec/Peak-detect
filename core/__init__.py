"""
Core modules for the Peak Analysis Tool.

These modules provide the fundamental data processing, analysis, 
and manipulation capabilities for the application.
"""

# Import main functionality from modules
from core.peak_detection import PeakDetector
from core.file_handler import browse_files
from core.peak_analysis_utils import find_peaks_with_window, adjust_lowpass_cutoff, timestamps_array_to_seconds

# Import new refactored modules
from core.data_analysis import (
    calculate_peak_areas, 
    calculate_peak_intervals, 
    calculate_auto_threshold
)
from core.data_utils import (
    decimate_for_plot, 
    get_width_range, 
    reset_application_state,
    find_nearest,
    timestamps_to_seconds
)
from core.file_export import (
    export_plot,
    save_peak_information_to_csv
)
from core.performance import (
    profile_function,
    get_memory_usage
)

# These imports will be uncommented as we move the modules
# from .peak_analysis_utils import * 