"""
Core data analysis functionality for the Peak Analysis Tool.

This module contains functions for analyzing and processing data,
particularly focused on peak detection and measurements.
"""

import numpy as np
import traceback
import logging
from scipy.signal import find_peaks, peak_widths
from core.performance import profile_function
from core.peak_analysis_utils import find_peaks_with_window, adjust_lowpass_cutoff
from functools import wraps
# Import the calculate_auto_threshold function from peak_detection
from core.peak_detection import calculate_auto_threshold as peak_detection_auto_threshold

# Configure logging
logger = logging.getLogger(__name__)

@profile_function
def calculate_peak_areas(detector, signal, time_values, height_lim, distance, rel_height, width_values, time_resolution=1e-4):
    """
    Calculate areas under detected peaks.
    
    Args:
        detector (PeakDetector): Instance of PeakDetector class
        signal (numpy.ndarray): The filtered signal data
        time_values (numpy.ndarray): Corresponding time values
        height_lim (float): Minimum height threshold for peaks
        distance (int): Minimum number of samples between peaks
        rel_height (float): Relative height at which peak width is measured (0-1)
        width_values (list): List of min and max width values for filtering
        time_resolution (float, optional): Time resolution in seconds per unit.
            Defaults to 1e-4 (0.1 milliseconds per unit).
        
    Returns:
        tuple: (peak_areas, start_indices, end_indices) if peaks are detected, None otherwise
    """
    try:
        # Detect peaks if not already detected
        if detector.peaks_indices is None:
            detector.detect_peaks(
                signal,
                time_values,
                height_lim,
                distance,
                rel_height,
                width_values,
                time_resolution=time_resolution
            )
        
        # Calculate areas using the PeakDetector
        peak_area, start, end = detector.calculate_peak_areas(signal)
        
        return peak_area, start, end 

    except Exception as e:
        logger.error(f"Error calculating peak areas: {str(e)}\n{traceback.format_exc()}")
        return None

@profile_function
def calculate_peak_intervals(t_value, peaks_indices):
    """
    Calculate time intervals between consecutive peaks.
    
    Parameters
    ----------
    t_value : array-like
        Time values
    peaks_indices : array-like
        Indices of detected peaks
        
    Returns
    -------
    list
        List of time intervals between consecutive peaks
    """
    try:
        if len(peaks_indices) < 2:
            return []
            
        # Extract time values at peak indices
        peak_times = t_value[peaks_indices]
        
        # Calculate differences between consecutive peaks
        intervals = np.diff(peak_times)
        
        return intervals.tolist()
        
    except Exception as e:
        logger.error(f"Error calculating peak intervals: {str(e)}\n{traceback.format_exc()}")
        return []

@profile_function
def calculate_auto_threshold(signal, percentile=95):
    """
    Wrapper for the auto threshold calculation that uses percentile method.
    This function maintains compatibility with existing code but delegates
    to the implementation in peak_detection.py.
    
    Parameters
    ----------
    signal : array-like
        Signal data
    percentile : float, optional
        Percentile to use for threshold calculation (default: 95)
        
    Returns
    -------
    float
        Calculated threshold value
    """
    try:
        # Use sigma multiplier equivalent to percentile value
        # For 95th percentile, ~1.65 sigma is roughly equivalent
        sigma_multiplier = 1.65 if percentile == 95 else 5
        return peak_detection_auto_threshold(signal, sigma_multiplier)
        
    except Exception as e:
        logger.error(f"Error calculating auto threshold: {str(e)}\n{traceback.format_exc()}")
        return 20.0  # Return a default value


def with_error_handling(error_msg):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.status_indicator.set_state('error')
                self.status_indicator.set_text(error_msg)
                self.show_error(error_msg, e)
                return None
        return wrapper
    return decorator 