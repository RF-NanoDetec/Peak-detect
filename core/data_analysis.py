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
from core.peak_analysis_utils import find_peaks_with_window, adjust_lowpass_cutoff, calculate_lowpass_cutoff
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
def analyze_time_resolved(filtered_signal, t_value, width_p_str, time_resolution, 
                          height_lim, distance, rel_height, prominence_ratio):
    """
    Analyze time-resolved data using provided parameters.
    
    This function performs time-resolved analysis of peak data, including:
    - Peak detection with current parameters
    - Area calculations
    - Interval measurements
    
    Parameters
    ----------
    filtered_signal : numpy.ndarray
        The filtered signal data.
    t_value : numpy.ndarray
        Time values corresponding to the signal.
    width_p_str : str
        String defining peak width range in ms (e.g., "0.1,50").
    time_resolution : float
        Time resolution (dwell time) in seconds.
    height_lim : float
        Minimum height (prominence) threshold for peaks.
    distance : int
        Minimum distance between peaks in samples.
    rel_height : float
        Relative height for width measurement (0-1).
    prominence_ratio : float
        Minimum prominence/height ratio for filtering subpeaks.
        
    Returns
    -------
    tuple or None
        (peaks, areas, intervals) if analysis is successful, None otherwise
    """
    try:
        if filtered_signal is None:
            logger.warning("analyze_time_resolved called with no filtered signal.")
            return None
            
        # Get current parameters from arguments
        width_values = width_p_str.strip().split(',')
        time_res = time_resolution
        if time_res <= 0:
            logger.warning(f"Invalid time resolution ({time_res}), defaulting to 0.1ms.")
            time_res = 0.0001
        sampling_rate = 1 / time_res
        
        # Convert width from ms to samples
        try:
             width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        except (ValueError, IndexError) as e:
             logger.error(f"Invalid width string '{width_p_str}': {e}")
             return None # Cannot proceed without valid width
        
        # Get the prominence ratio threshold from argument
        prominence_ratio_thresh = prominence_ratio
        
        # Find peaks with current parameters
        peaks, properties = find_peaks_with_window(
            filtered_signal,
            width=width_p,
            prominence=height_lim,
            distance=distance,
            rel_height=rel_height,
            prominence_ratio=prominence_ratio_thresh
        )
        
        if len(peaks) == 0:
            logger.info("No peaks detected in analyze_time_resolved with current parameters.")
            return None
            
        # Calculate areas under peaks
        areas = []
        if "left_ips" in properties and "right_ips" in properties:
            for i, peak in enumerate(peaks):
                left_idx = int(properties["left_ips"][i])
                right_idx = int(properties["right_ips"][i])
                if left_idx < right_idx and right_idx <= len(filtered_signal):
                    area = np.trapz(filtered_signal[left_idx:right_idx])
                    areas.append(area)
                else:
                    logger.warning(f"Invalid indices for peak {peak} area calculation: [{left_idx}:{right_idx}]")
                    areas.append(0) # Append 0 or NaN? Let's use 0 for now.
            areas = np.array(areas)
        else:
            logger.warning("Width properties (ips) missing, cannot calculate areas.")
            areas = np.full_like(peaks, 0.0) # Return array of zeros
        
        # Calculate intervals between peaks
        if t_value is None or len(t_value) != len(filtered_signal):
             logger.warning("Invalid or missing t_value, cannot calculate intervals.")
             intervals = []
        else:
             intervals = calculate_peak_intervals(t_value, peaks)
        
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

@profile_function
def calculate_auto_cutoff(signal_data, time_resolution):
    """
    Calculate an appropriate low-pass cutoff frequency based on signal characteristics.

    This method finds the highest signal value and uses 70% of that value as a
    threshold for detecting prominent peaks. The average width of these peaks
    is then used to estimate a suitable cutoff frequency using the calculate_lowpass_cutoff utility.

    Args:
        signal_data (numpy.ndarray): The raw signal data (e.g., app.state.x_value).
        time_resolution (float): The time interval between data points in seconds.

    Returns:
        float: The suggested cutoff frequency in Hz.
               Returns a default value (e.g., 10.0 Hz) if calculation fails or no peaks are found.
    """
    try:
        if signal_data is None or len(signal_data) == 0 or time_resolution <= 0:
            logger.warning("Insufficient data for auto cutoff calculation.")
            return 10.0 # Default cutoff

        fs = 1.0 / time_resolution
        logger.debug(f"Auto Cutoff: Sampling rate (fs): {fs:.2f} Hz")

        # Find the highest signal value and calculate 70% threshold
        signal_max = np.max(signal_data)
        threshold = signal_max * 0.7
        logger.debug(f"Auto Cutoff: Max signal value: {signal_max:.4f}, 70% threshold: {threshold:.4f}")

        # Detect peaks above the 70% threshold to measure their widths
        # We don't need complex filtering here, just basic height
        peaks, _ = find_peaks(signal_data, height=threshold)

        if len(peaks) == 0:
            logger.warning("Auto Cutoff: No peaks found above 70% threshold, using default cutoff 10.0 Hz")
            return 10.0  # Default cutoff if no significant peaks found

        logger.debug(f"Auto Cutoff: Found {len(peaks)} peaks above 70% threshold.")

        # Use calculate_lowpass_cutoff utility function
        # It handles peak width calculation internally based on the threshold
        suggested_cutoff = calculate_lowpass_cutoff(
            signal_data, fs, threshold, 1.0, time_resolution=time_resolution
        )

        logger.info(f"Auto Cutoff: Calculated cutoff frequency: {suggested_cutoff:.2f} Hz")
        return suggested_cutoff

    except Exception as e:
        logger.error(f"Error calculating auto cutoff frequency: {str(e)}\n{traceback.format_exc()}")
        return 10.0 # Return default cutoff on error

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