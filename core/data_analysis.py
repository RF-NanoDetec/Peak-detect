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
def calculate_peak_areas(detector, filtered_signal, t_value, height_lim_factor, 
                         distance, rel_height, width_values):
    """
    Calculate the areas of detected peaks.
    
    Parameters
    ----------
    detector : PeakDetector
        Instance of PeakDetector for peak detection
    filtered_signal : array-like
        The filtered signal data
    t_value : array-like
        Time values
    height_lim_factor : float
        Height limit factor for peak detection
    distance : int
        Minimum distance between peaks
    rel_height : float
        Relative height for peak width calculation
    width_values : list
        Minimum and maximum width values for filtering peaks
        
    Returns
    -------
    tuple
        (peak_area, start, end) if successful, None otherwise
    """
    try:
        # Detect peaks using the PeakDetector
        detector.detect_peaks(
            filtered_signal,
            t_value,
            height_lim_factor,
            distance,
            rel_height,
            width_values
        )
        
        # Calculate areas using the PeakDetector
        peak_area, start, end = detector.calculate_peak_areas(filtered_signal)
        
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

@profile_function
def calculate_auto_cutoff_frequency(t_value, signal, factor=0.1):
    """
    Automatically calculate a suitable cutoff frequency for filtering.
    
    Parameters
    ----------
    t_value : array-like
        Time values
    signal : array-like
        Signal data
    factor : float, optional
        Factor to adjust the calculated frequency (default: 0.1)
        
    Returns
    -------
    float
        Calculated cutoff frequency
    """
    try:
        # Estimate sampling frequency
        if len(t_value) > 1:
            # Calculate average sampling rate
            sampling_rate = 1.0 / (t_value[1] - t_value[0])
            
            # Calculate cutoff as a fraction of Nyquist frequency
            nyquist = 0.5 * sampling_rate
            cutoff = factor * nyquist
            
            return cutoff
        else:
            return 10.0  # Default if insufficient data
            
    except Exception as e:
        logger.error(f"Error calculating auto cutoff frequency: {str(e)}\n{traceback.format_exc()}")
        return 10.0  # Return a default value 

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