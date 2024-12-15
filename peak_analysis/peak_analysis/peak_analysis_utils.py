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

# Set default seaborn style
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.2)
sns.set_context("notebook", rc={"lines.linewidth": 1.0})

print("Lade peak_analysis_utils.py")
# Add after the imports, before the first function definition
def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Get start time
        start_time = time.time()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Get end time
        end_time = time.time()
        
        # Get memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Calculate memory difference
        memory_diff = memory_after - memory_before
        
        # Log the performance data
        logging.info(f"{func.__name__} performance:")
        logging.info(f"  Time: {end_time - start_time:.2f} seconds")
        logging.info(f"  Memory before: {memory_before:.1f} MB")
        logging.info(f"  Memory after: {memory_after:.1f} MB")
        logging.info(f"  Memory difference: {memory_diff:.1f} MB")
        
        # Print to console as well
        print(f"\n{func.__name__} performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory usage: {memory_diff:+.1f} MB")
        
        return result
    return wrapper

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
    idx = 0
    min_diff = np.abs(array[0] - value)
    for i in range(1, len(array)):
        diff = np.abs(array[i] - value)
        if diff < min_diff:
            min_diff = diff
            idx = i
    return idx

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

# Add this memory tracking function
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

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

print("profile_function ist definiert:", 'profile_function' in globals())