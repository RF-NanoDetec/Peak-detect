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

# Import profiling utilities from the utils package
from utils.performance import profile_function, get_memory_usage

# Set default seaborn style
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.2)
sns.set_context("notebook", rc={"lines.linewidth": 1.0})

print("Lade peak_analysis_utils.py")

# Butterworth filter application
def apply_butterworth_filter(order, Wn, btype, fs, x):
    b, a = butter(order, Wn, btype, fs=fs)
    x_f = filtfilt(b, a, x)
    print(f'Butterworth filter coefficients (b, a): {b}, {a}')
    print(f'Filtered signal: {x_f[:10]}...')  # Printing the first 10 values for brevity
    return x_f

# Find the nearest value in an array
@njit
def find_nearest(array, value):
    """
    Find the nearest value in an array using binary search if array is large,
    or linear search for small arrays
    
    Parameters
    ----------
    array : ndarray
        Array to search in
    value : float
        Value to find
        
    Returns
    -------
    int
        Index of the nearest value
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
    Find index of the nearest value in a sorted array using binary search
    
    Parameters
    ----------
    array : ndarray
        Sorted array to search in
    value : float
        Value to find
        
    Returns
    -------
    int
        Index of the nearest value
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
@njit
def timestamps_to_seconds(timestamps, start_time):
    """Convert timestamps to seconds from start time"""
    try:
        seconds = []
        start_min, start_sec = map(int, start_time.split(':'))
        start_time_seconds = start_min * 60 + start_sec
        
        for timestamp in timestamps:
            minu, sec = map(int, timestamp.split(':'))
            seconds.append(minu * 60 + sec - start_time_seconds)
            
        return seconds
    except Exception as e:
        raise ValueError(f"Error converting timestamps: {e}\n"
                        f"Format should be 'MM:SS' (e.g., '01:30')")

# Detect peaks with a sliding window and filter out invalid ones
@profile_function
def find_peaks_with_window(signal, width, prominence, distance, rel_height):
    """
    Optimized peak detection using scipy.signal.find_peaks
    
    Parameters
    ----------
    signal : ndarray
        Input signal to find peaks in
    width : list or tuple
        Expected peak width range [min, max]
    prominence : float
        Minimum peak prominence
    distance : int
        Minimum distance between peaks
    rel_height : float
        Relative height for width calculation
        
    Returns
    -------
    tuple
        (peaks, properties) containing peak indices and properties
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
    peaks, _ = find_peaks(signal, width=[1, 2000], prominence=big_counts, distance=1000)
    widths = peak_widths(signal, peaks, rel_height=0.5)[0]
    avg_width = np.mean(widths) / fs
    return avg_width

# Adjust the low-pass filter cutoff frequency
def adjust_lowpass_cutoff(signal, fs, big_counts, normalization_factor):
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

