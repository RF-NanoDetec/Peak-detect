"""
Peak Analysis Utilities Module

This module provides utility functions for signal processing, peak detection,
and performance measurement in time series data analysis.

Functions:
    apply_butterworth_filter: Apply a Butterworth filter to a signal
    find_nearest: Find the index of the nearest value in an array
    binary_search_nearest: Fast binary search to find nearest value
    timestamps_to_seconds: Convert timestamps to seconds from start
    find_peaks_with_window: Detect peaks with specified window parameters
    estimate_peak_widths: Estimate widths of peaks in a signal
    adjust_lowpass_cutoff: Adjust cutoff frequency based on signal characteristics
"""

import os
import time
import logging
import traceback
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import seaborn as sns
import psutil
from numba import njit
from scipy.signal import find_peaks, butter, filtfilt, peak_widths

# Import profiling utilities from the performance module
from core.performance import profile_function, get_memory_usage

# Set default seaborn style
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.2)
sns.set_context("notebook", rc={"lines.linewidth": 1.0})

print("Lade peak_analysis_utils.py")

# Butterworth filter application
def apply_butterworth_filter(order, Wn, btype, fs, x):
    """
    Apply a Butterworth filter to a signal.
    
    Parameters:
        order (int): The order of the filter
        Wn (float or tuple): The critical frequency or frequencies.
            For lowpass and highpass filters, Wn is a scalar.
            For bandpass and bandstop filters, Wn is a length-2 sequence.
        btype (str): The type of filter: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
        fs (float): The sampling frequency of the signal
        x (numpy.ndarray): The signal to be filtered
        
    Returns:
        numpy.ndarray: The filtered signal
        
    Example:
        >>> filtered_signal = apply_butterworth_filter(4, 0.1, 'lowpass', 100, raw_signal)
    """
    b, a = butter(order, Wn, btype=btype, fs=fs)
    x_f = filtfilt(b, a, x)
    print(f'Butterworth filter coefficients (b, a): {b}, {a}')
    print(f'Filtered signal: {x_f[:10]}...')  # Printing the first 10 values for brevity
    return x_f

# Find the nearest value in an array
@njit
def find_nearest(array, value):
    """
    Find the index of the nearest value in an array using linear search.
    
    This function is optimized with Numba for faster execution.
    
    Parameters:
        array (numpy.ndarray): The array to search in
        value (float): The value to find
        
    Returns:
        int: Index of the nearest value in the array
        
    Note:
        For large arrays, consider using binary_search_nearest instead
        as it has better performance for sorted arrays.
    """
    # For small arrays, linear search is faster
    if len(array) < 50:
        idx = 0
        min_diff = abs(array[0] - value)
        for i in range(1, len(array)):
            diff = abs(array[i] - value)
            if diff < min_diff:
                min_diff = diff
                idx = i
        return idx
    
    # For larger arrays, use binary search
    # First check if array is sorted
    is_sorted = True
    for i in range(1, len(array)):
        if array[i] < array[i-1]:
            is_sorted = False
            break
    
    if is_sorted:
        return binary_search_nearest(array, value)
    else:
        # If not sorted, sort a copy and map back to original indices
        indices = np.arange(len(array))
        sorted_indices = indices[np.argsort(array)]
        sorted_array = np.sort(array)
        
        nearest_idx_in_sorted = binary_search_nearest(sorted_array, value)
        return sorted_indices[nearest_idx_in_sorted]

# Helper function for binary search - significantly faster than linear search
@njit
def binary_search_nearest(array, value):
    """
    Find the index of the nearest value in a sorted array using binary search.
    
    This function is optimized with Numba for faster execution and provides
    better performance than linear search for large sorted arrays.
    
    Parameters:
        array (numpy.ndarray): The sorted array to search in
        value (float): The value to find
        
    Returns:
        int: Index of the nearest value in the array
        
    Note:
        The array must be sorted in ascending order for correct results.
        
    Example:
        >>> idx = binary_search_nearest(time_array, target_time)
        >>> nearest_time = time_array[idx]
    """
    if len(array) == 0:
        return -1
    
    # Special cases
    if value <= array[0]:
        return 0
    if value >= array[-1]:
        return len(array) - 1
    
    # Binary search
    low = 0
    high = len(array) - 1
    
    while low <= high:
        mid = (low + high) // 2
        
        if array[mid] == value:
            return mid
        
        if array[mid] < value:
            low = mid + 1
        else:
            high = mid - 1
    
    # At this point, low > high
    # Return the closest of the two
    if low >= len(array):
        return high
    if high < 0:
        return low
    
    if abs(array[low] - value) < abs(array[high] - value):
        return low
    else:
        return high

# Convert timestamps to seconds
def timestamps_array_to_seconds(timestamps, start_time):
    """
    Convert an array of timestamps to seconds elapsed from the start time.
    
    This function works on arrays of timestamps.
    For converting a single timestamp, use timestamps_to_seconds from data_utils.
    
    Parameters:
        timestamps (list or numpy.ndarray): Array of timestamps in "MM:SS" format
        start_time (str): Reference start time in "MM:SS" format
        
    Returns:
        list: Array of elapsed seconds from start_time
        
    Example:
        >>> time_in_seconds = timestamps_array_to_seconds(raw_timestamps, raw_timestamps[0])
    """
    try:
        seconds = []
        start_min, start_sec = map(int, start_time.split(':'))
        start_time_seconds = start_min * 60 + start_sec
        
        for timestamp in timestamps:
            minu, sec = map(int, timestamp.split(':'))
            seconds.append(minu * 60 + sec - start_time_seconds)
            
        return seconds
    except Exception as e:
        print(f"Error in timestamps_array_to_seconds: {e}")
        print(f"Timestamps: {timestamps}, Start time: {start_time}")
        raise ValueError(f"Error converting timestamps: {e}\n"
                        f"Format should be 'MM:SS' (e.g., '01:30')")

# Detect peaks with a sliding window and filter out invalid ones
@profile_function
def find_peaks_with_window(signal, width, prominence, distance, rel_height):
    """
    Detect peaks in a signal with specified window parameters.
    
    This function wraps scipy.signal.find_peaks with additional parameters
    for controlling the detection window and is performance-profiled.
    
    Parameters:
        signal (numpy.ndarray): The signal to analyze
        width (tuple): Tuple of (min_width, max_width) or None
        prominence (float): Minimum prominence of peaks
        distance (int): Minimum distance between peaks
        rel_height (float): Relative height at which width is measured
        
    Returns:
        tuple: (peaks, properties) where:
            - peaks is a numpy array of indices where peaks were found
            - properties is a dict containing peak properties (heights, widths, etc.)
            
    Example:
        >>> peaks, properties = find_peaks_with_window(
        ...     signal, width=(10, 50), prominence=0.1, distance=20, rel_height=0.5)
    """
    # Call the scipy function and return results
    # The find_peaks function is already highly optimized C code,
    # so we just ensure our inputs are optimized
    
    # Ensure signal is a contiguous array for better performance
    if not np.isfortran(signal) and not signal.flags.c_contiguous:
        signal = np.ascontiguousarray(signal)
    
    peaks, properties = find_peaks(signal, 
                                 width=width,
                                 prominence=prominence,
                                 distance=distance,
                                 rel_height=rel_height)
        
    return peaks, properties

# Estimate the average peak width
def estimate_peak_widths(signal, fs, big_counts):
    """
    Estimate appropriate peak widths based on signal characteristics.
    
    Parameters:
        signal (numpy.ndarray): The signal to analyze
        fs (float): Sampling frequency of the signal
        big_counts (int): Number of major peaks expected
        
    Returns:
        tuple: (min_width, max_width) suitable for peak detection
    """
    peaks, _ = find_peaks(signal, width=[1, 2000], prominence=big_counts, distance=1000)
    widths = peak_widths(signal, peaks, rel_height=0.5)[0]
    avg_width = np.mean(widths) / fs
    return avg_width

# Adjust the low-pass filter cutoff frequency
def adjust_lowpass_cutoff(signal, fs, big_counts, normalization_factor):
    """
    Adjust lowpass filter cutoff frequency based on signal characteristics.
    
    This function analyzes the signal spectrum and determines an appropriate
    cutoff frequency for lowpass filtering to preserve important features
    while reducing noise.
    
    Parameters:
        signal (numpy.ndarray): The signal to analyze
        fs (float): Sampling frequency of the signal
        big_counts (int): Number of major peaks expected
        normalization_factor (float): Factor to normalize the cutoff
        
    Returns:
        float: Recommended cutoff frequency for lowpass filtering
        
    Example:
        >>> cutoff = adjust_lowpass_cutoff(raw_signal, 100, 20, 0.5)
        >>> filtered = apply_butterworth_filter(4, cutoff, 'lowpass', 100, raw_signal)
    """
    print("\nDEBUG: Starting adjust_lowpass_cutoff")
    print(f"DEBUG: Input parameters:")
    print(f"- fs: {fs}")
    print(f"- big_counts: {big_counts}")
    print(f"- normalization_factor: {normalization_factor}")
    
    avg_width = estimate_peak_widths(signal, fs, big_counts)
    print(f'\nDEBUG: Average width of peaks: {avg_width} seconds')
    
    # Calculate base cutoff frequency from average width (in Hz)
    base_cutoff = 1 / avg_width  # Convert time width to frequency
    print(f'DEBUG: Base cutoff frequency (1/width): {base_cutoff:.2f} Hz')
    
    # Apply normalization factor to adjust the cutoff
    cutoff = base_cutoff * float(normalization_factor)  # Ensure normalization_factor is float
    print(f'DEBUG: After normalization:')
    print(f'- Base cutoff: {base_cutoff:.2f} Hz')
    print(f'- Normalization factor: {normalization_factor}')
    print(f'- Adjusted cutoff: {cutoff:.2f} Hz')
    
    # Limit the cutoff frequency to reasonable values
    original_cutoff = cutoff
    cutoff = max(min(cutoff, 10000), 50)
    if cutoff != original_cutoff:
        print(f'DEBUG: Cutoff was limited from {original_cutoff:.2f} to {cutoff:.2f} Hz')
    
    print(f'\nDEBUG: Final cutoff frequency: {cutoff:.2f} Hz')
    
    order = 2
    btype = 'lowpass'
    filtered_signal = apply_butterworth_filter(order, cutoff, btype, fs, signal)
    
    return filtered_signal, cutoff

