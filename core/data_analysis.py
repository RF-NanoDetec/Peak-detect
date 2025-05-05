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
def calculate_peak_areas(detector, signal, time_values, height_lim, distance, rel_height, width_values, time_resolution=1e-4, prominence_ratio=0.8):
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
        prominence_ratio (float, optional): Threshold for the ratio of prominence
            to peak height. Defaults to 0.8 (80%).
        
    Returns:
        tuple: (peak_areas, start_indices, end_indices) if peaks are detected, None otherwise
    """
    try:
        if signal is None:
            return None
            
        # Convert width from ms to samples
        sampling_rate = 1 / time_resolution
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        
        # Find peaks with the current parameters
        peaks, properties = find_peaks_with_window(
            signal,
            width=width_p,
            prominence=height_lim,
            distance=distance,
            rel_height=rel_height,
            prominence_ratio=prominence_ratio
        )
        
        if len(peaks) == 0:
            return None
            
        # Calculate areas under peaks
        areas = []
        start_indices = []
        end_indices = []
        
        for i, peak in enumerate(peaks):
            left_idx = int(properties["left_ips"][i])
            right_idx = int(properties["right_ips"][i])
            
            if left_idx < right_idx:
                area = np.trapz(signal[left_idx:right_idx])
                areas.append(area)
                start_indices.append(left_idx)
                end_indices.append(right_idx)
            
        return areas, start_indices, end_indices
        
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
def analyze_time_resolved(app):
    """
    Analyze time-resolved data using current parameters and update the visualization.
    
    This function performs time-resolved analysis of peak data, including:
    - Peak detection with current parameters
    - Area calculations
    - Interval measurements
    - Statistical analysis
    
    Parameters
    ----------
    app : Application
        The main application instance containing the data and parameters
        
    Returns
    -------
    tuple or None
        (peaks, areas, intervals) if analysis is successful, None otherwise
        - peaks: array of peak indices
        - areas: array of peak areas
        - intervals: array of peak intervals
    """
    try:
        if app.filtered_signal is None:
            return None
            
        # Get current parameters
        width_values = app.width_p.get().strip().split(',')
        time_res = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution
        if time_res <= 0:
            time_res = 0.0001  # Default to 0.1ms if invalid
        sampling_rate = 1 / time_res
        
        # Convert width from ms to samples
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        
        # Get the prominence ratio threshold
        prominence_ratio = app.prominence_ratio.get()
        
        # Find peaks with current parameters
        peaks, properties = find_peaks_with_window(
            app.filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get(),
            prominence_ratio=prominence_ratio
        )
        
        if len(peaks) == 0:
            return None
            
        # Calculate areas under peaks
        areas = []
        for i, peak in enumerate(peaks):
            left_idx = int(properties["left_ips"][i])
            right_idx = int(properties["right_ips"][i])
            if left_idx < right_idx:
                area = np.trapz(app.filtered_signal[left_idx:right_idx])
                areas.append(area)
            else:
                areas.append(0)
        areas = np.array(areas)
        
        # Calculate intervals between peaks
        intervals = calculate_peak_intervals(app.t_value, peaks)
        
        return peaks, areas, intervals
        
    except Exception as e:
        logger.error(f"Error in analyze_time_resolved: {str(e)}\n{traceback.format_exc()}")
        return None

@profile_function
def calculate_auto_threshold(signal, percentile=95, sigma_multiplier=None):
    """
    Wrapper for the auto threshold calculation.
    
    This function maintains compatibility with existing code but delegates
    to the implementation in peak_detection.py. It now supports direct sigma
    specification, which takes precedence over percentile-based selection.
    
    Parameters
    ----------
    signal : array-like
        Signal data
    percentile : float, optional
        Percentile to use for threshold calculation (default: 95)
    sigma_multiplier : float, optional
        If provided, this value is used directly as the sigma multiplier,
        overriding the percentile-based selection.
        
    Returns
    -------
    float
        Calculated threshold value
    """
    try:
        # If sigma_multiplier is directly provided, use it
        if sigma_multiplier is not None:
            sigma = sigma_multiplier
        else:
            # Otherwise use percentile-based selection as before
            sigma = 1.65 if percentile == 95 else 5
            
        return peak_detection_auto_threshold(signal, sigma)
        
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