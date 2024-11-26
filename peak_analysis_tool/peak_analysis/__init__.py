"""
Peak Analysis Tool
-----------------
A tool for analyzing peak data in scientific measurements.

This package provides functionality for:
- Signal processing
- Peak detection
- Data analysis
- Visualization
"""

import logging
from .config import Config
from .data_processing import DataProcessor
from .signal_processing import (
    find_nearest,
    apply_butterworth_filter,
    adjust_lowpass_cutoff
)
from .peak_detection import (
    find_peaks_with_window,
    estimate_peak_widths,
    calculate_peak_areas
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Version info
__version__ = '1.0.0'
__author__ = 'Lucjan & Silas'
__email__ = 'your.email@example.com'

# Define what should be imported with "from peak_analysis import *"
__all__ = [
    'Config',
    'DataProcessor',
    'find_nearest',
    'apply_butterworth_filter',
    'adjust_lowpass_cutoff',
    'find_peaks_with_window',
    'estimate_peak_widths',
    'calculate_peak_areas'
]
