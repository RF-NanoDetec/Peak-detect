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
from scipy.signal import find_peaks, butter, filtfilt, peak_widths, savgol_filter

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
def find_peaks_with_window(signal, width, prominence, distance, rel_height, prominence_ratio):
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
        prominence_ratio (float): Threshold for the ratio of prominence
            to peak height. Peaks with ratio < threshold are filtered out as subpeaks.
            Higher values (e.g., 0.9) are more strict.
        
    Returns:
        tuple: (peaks, properties) where:
            - peaks is a numpy array of indices where peaks were found
            - properties is a dict containing peak properties (heights, widths, etc.)
    """
    # Ensure signal is a contiguous array for better performance
    if not np.isfortran(signal) and not signal.flags.c_contiguous:
        signal = np.ascontiguousarray(signal)
    
    # Find peaks with specified parameters
    peaks, properties = find_peaks(signal, 
                                 width=width,
                                 prominence=prominence,
                                 distance=distance, 
                                 rel_height=rel_height)
    
    # Calculate peak widths and interpolated positions
    if len(peaks) > 0:
        # Calculate widths at the specified relative height
        width_results = peak_widths(signal, peaks, rel_height=rel_height)
        
        # Store all width-related properties
        properties['widths'] = width_results[0]  # Width in samples
        properties['width_heights'] = width_results[1]  # Height at which width is measured
        properties['left_ips'] = width_results[2]  # Left interpolated position
        properties['right_ips'] = width_results[3]  # Right interpolated position
        
        # Print debug information for the first few peaks
        print("\nPeak Width Analysis:")
        for i in range(min(3, len(peaks))):
            print(f"\nPeak {i+1}:")
            print(f"  Position: {peaks[i]}")
            print(f"  Value: {signal[peaks[i]]:.3f}")
            print(f"  Width: {width_results[0][i]:.3f} samples")
            print(f"  Width height: {width_results[1][i]:.3f}")
            print(f"  Left IP: {width_results[2][i]:.3f}")
            print(f"  Right IP: {width_results[3][i]:.3f}")
    else:
        # If no peaks found, initialize empty arrays
        properties['widths'] = np.array([])
        properties['width_heights'] = np.array([])
        properties['left_ips'] = np.array([])
        properties['right_ips'] = np.array([])
    
    # Filter out subpeaks (peaks that sit on top of larger peaks)
    peaks, properties = filter_subpeaks(signal, peaks, properties, prominence_ratio_threshold=prominence_ratio)
    
    return peaks, properties

def filter_subpeaks(signal, peaks, properties, prominence_ratio_threshold):
    """
    Filter out peaks that are likely subpeaks sitting on top of larger peaks.
    
    This function uses the ratio of peak prominence to peak height to identify
    subpeaks. Peaks with low prominence-to-height ratios are considered subpeaks
    and are filtered out.
    
    Parameters:
        signal (numpy.ndarray): The signal to analyze
        peaks (numpy.ndarray): Array of peak indices
        properties (dict): Dictionary of peak properties from find_peaks
        prominence_ratio_threshold (float): Threshold for the ratio of prominence
            to peak height. Peaks with ratio < threshold are filtered out as subpeaks.
            
            Higher threshold values (e.g., 0.8) will be more strict and filter out more peaks.
            Lower threshold values (e.g., 0.2) will be more permissive and keep more peaks.
            
            Examples:
            - With threshold=0.8: Only peaks with prominence ≥ 80% of their height are kept
            - With threshold=0.5: Peaks with prominence ≥ 50% of their height are kept
            - With threshold=0.1: Most peaks are kept, only those with prominence < 10% of height are filtered
    
    Returns:
        tuple: (filtered_peaks, filtered_properties) with subpeaks removed
    """
    if len(peaks) == 0 or 'prominences' not in properties:
        return peaks, properties
    
    # Track which peaks to keep
    keep_mask = np.ones(len(peaks), dtype=bool)
    
    # Check each peak
    for i, peak_idx in enumerate(peaks):
        # Get peak height and prominence
        peak_height = signal[peak_idx]
        prominence = properties['prominences'][i]
        
        # Calculate ratio of prominence to peak height
        if peak_height > 0:  # Avoid division by zero
            ratio = prominence / peak_height
            
            # If the prominence ratio is below the threshold, it's considered a subpeak and filtered out
            # Higher threshold = more peaks filtered; Lower threshold = fewer peaks filtered
            if ratio < prominence_ratio_threshold:
                keep_mask[i] = False
    
    # Apply the mask to filter peaks and properties
    filtered_peaks = peaks[keep_mask]
    
    # Filter all properties
    filtered_properties = {}
    for key, values in properties.items():
        if isinstance(values, np.ndarray) and len(values) == len(peaks):
            filtered_properties[key] = values[keep_mask]
        else:
            filtered_properties[key] = values
    
    # Logging for debugging information
    import logging
    _logger = logging.getLogger(__name__)
    total_peaks = len(peaks)
    filtered_out_peaks = total_peaks - len(filtered_peaks)
    _logger.debug(
        f"Filtered out {filtered_out_peaks} subpeaks out of {total_peaks} total peaks"
    )
    _logger.debug(
        f"Using prominence-to-height ratio threshold of {prominence_ratio_threshold:.2f}"
    )
    _logger.debug(
        f"Keeping {len(filtered_peaks)} peaks with prominence/height ratio >= {prominence_ratio_threshold:.2f}"
    )
    
    return filtered_peaks, filtered_properties

# Estimate the average peak width
def estimate_peak_widths(signal, fs, prominence_threshold=None, time_resolution=1e-4, big_counts=None, **kwargs):
    """
    Estimate the average width of peaks in a signal.
    
    This function finds significant peaks above the specified threshold and calculates 
    their average width at half prominence, which is useful for determining an appropriate
    cutoff frequency for low-pass filtering.
    
    Parameters:
        signal (numpy.ndarray): The signal to analyze
        fs (float): Sampling frequency of the signal
        prominence_threshold (float | None): Threshold for peak prominence or height. If None,
            an adaptive threshold of 5*std(signal) is used.
        time_resolution (float, optional): Deprecated. Not used for conversion; widths are
            converted to seconds using fs. Kept for backward compatibility.
        
    Returns:
        float: Average width of peaks in seconds
               
    Note:
        This function calculates peak widths in sample units and then
        converts to seconds using the provided time_resolution. For a typical 
        time resolution of 0.1ms (1e-4 seconds), a width of 10 samples 
        represents 1 millisecond.
    """
    import logging
    _logger = logging.getLogger(__name__)
    _logger.debug("---- Starting estimate_peak_widths ----")
    _logger.debug(f"Input signal shape: {signal.shape}")
    _logger.debug(f"Sampling frequency (fs): {fs} Hz")
    # Determine threshold if not provided
    if prominence_threshold is None or prominence_threshold <= 0:
        prominence_threshold = max(5.0 * np.std(signal), 0.1)
    _logger.debug(f"Prominence threshold: {prominence_threshold}")
    
    # Find significant peaks for width estimation
    peaks, _ = find_peaks(signal, width=[1, 20000], prominence=prominence_threshold, distance=1000)
    
    # If no peaks found, use default width
    if len(peaks) == 0:
        _logger.debug("No peaks found for width estimation, using default width")
        default_width = 0.001  # Default width if no peaks found (1 ms)
        _logger.debug(f"Returned default width: {default_width}")
        _logger.debug("---- Finished estimate_peak_widths ----")
        return default_width
    
    # Calculate widths at half-prominence
    width_results = peak_widths(signal, peaks, rel_height=0.5)
    widths = width_results[0]
    
    # Calculate average width and convert to seconds using sampling frequency
    avg_samples = np.mean(widths)
    # Prefer fs for conversion; fallback to time_resolution only if fs is invalid
    if fs and fs > 0:
        avg_width = avg_samples / fs
    else:
        avg_width = avg_samples * time_resolution
    
    # Print detailed debug info
    _logger.debug(f"Peaks found: {len(peaks)}")
    if len(peaks) < 20:  # Only print all widths if there aren't too many
        _logger.debug(f"Raw width values in samples: {widths}")
    else:
        _logger.debug(f"First 5 width values in samples: {widths[:5]}...")
    
    _logger.debug(f"Average width in samples: {avg_samples:.2f}")
    _logger.debug(f"Average width in seconds: {avg_width:.6f}")
    if avg_width > 0:
        _logger.debug(f"Inverse of avg_width (1/avg_width): {1/avg_width:.2f}")
        _logger.debug(f"For cutoff calculation (1/avg_width): {1/avg_width:.2f} Hz")
    _logger.debug("---- Finished estimate_peak_widths ----")
    
    return avg_width

# Adjust the low-pass filter cutoff frequency
def adjust_lowpass_cutoff(
    signal, fs, 
    filter_type='butterworth', 
    # --- Butterworth specific ---
    manual_cutoff_hz=0.0, # New parameter to accept a pre-calculated cutoff
    prominence_threshold_butter_cutoff_calc=0.1, 
    normalization_factor_butter=1.0,
    butter_order=2,
    # --- Savitzky-Golay specific ---
    savgol_window_length=None, 
    savgol_polyorder=None,   
    prominence_threshold_savgol_window_est=0.1, 
    # --- Common ---
    time_resolution=1e-4
):
    """
    Adjusts/determines filter parameters and applies the chosen filter (Butterworth or Savitzky-Golay).

    For Butterworth, it calculates an adaptive lowpass cutoff.
    For Savitzky-Golay, it uses provided parameters or estimates window_length.
    
    Parameters:
        signal (numpy.ndarray): The signal to analyze.
        fs (float): Sampling frequency of the signal.
        filter_type (str): 'butterworth' or 'savgol'. Defaults to 'butterworth'.
        
        manual_cutoff_hz (float): Pre-calculated cutoff frequency for Butterworth.
            If >0, use it directly. Else, use existing auto-calc logic.
        prominence_threshold_butter_cutoff_calc (float): Prominence threshold used in 
            `estimate_peak_widths` for Butterworth cutoff calculation. Defaults to 0.1.
        normalization_factor_butter (float): Factor to normalize the Butterworth cutoff.
            Defaults to 1.0.
        butter_order (int): Order of the Butterworth filter. Defaults to 2.

        savgol_window_length (int, optional): Window length for Savitzky-Golay filter. 
            Must be odd. If None, it's estimated based on signal characteristics.
        savgol_polyorder (int, optional): Polynomial order for Savitzky-Golay filter.
            Must be less than savgol_window_length. If None, a default (e.g., 2 or 3) is used.
        prominence_threshold_savgol_window_est (float): Prominence threshold for peak 
            detection when estimating `savgol_window_length`. Defaults to 0.1.
            
        time_resolution (float, optional): Time resolution in seconds per unit, used by 
            `estimate_peak_widths` for Butterworth. Defaults to 1e-4.
            
    Returns:
        tuple: (filtered_signal, applied_filter_params_dict)
            - filtered_signal is the filtered data.
            - applied_filter_params_dict is a dictionary containing the type of filter applied 
              and its parameters (e.g., {'type': 'butterworth', 'cutoff_hz': 10.0, 'order': 2} or
              {'type': 'savgol', 'window_length': 51, 'polyorder': 3}).
            
    Note:
        The return signature has changed from (filtered_signal, cutoff_frequency).
        The second element is now a dictionary of parameters.
    """
    # Backwards compatibility: support legacy signature where the third argument
    # is a numeric prominence threshold and the fourth is a normalization factor.
    if not isinstance(filter_type, str):
        try:
            legacy_prominence = float(filter_type)
        except Exception:
            legacy_prominence = 0.1
        try:
            legacy_norm = float(manual_cutoff_hz)
        except Exception:
            legacy_norm = 1.0
        cutoff_hz = calculate_lowpass_cutoff(
            signal, fs, legacy_prominence, legacy_norm, time_resolution
        )
        filtered_signal_legacy = apply_butterworth_filter(
            butter_order, cutoff_hz, 'lowpass', fs, signal
        )
        return filtered_signal_legacy, cutoff_hz

    print(f"\\n==== DEBUG: Starting adjust_lowpass_cutoff (Filter Type: {filter_type}) ====")
    print(f"DEBUG: Input parameters:")
    print(f"- fs: {fs}")
    if filter_type == 'butterworth':
        print(f"- manual_cutoff_hz (input): {manual_cutoff_hz}")
        print(f"- prominence_threshold_butter_cutoff_calc (for auto): {prominence_threshold_butter_cutoff_calc}")
        print(f"- normalization_factor_butter (for auto): {normalization_factor_butter}")
        print(f"- butter_order: {butter_order}")
    elif filter_type == 'savgol':
        print(f"- savgol_window_length (input): {savgol_window_length}")
        print(f"- savgol_polyorder (input): {savgol_polyorder}")
        print(f"- prominence_threshold_savgol_window_est: {prominence_threshold_savgol_window_est}")
    print(f"- time_resolution: {time_resolution}")

    applied_filter_params = {'type': filter_type}
    filtered_signal = signal # Default to original if filtering fails or not applicable

    if filter_type == 'butterworth':
        print(f"DEBUG: Butterworth mode selected.")
        final_cutoff_hz = 0.0

        if manual_cutoff_hz > 0:
            print(f"DEBUG: Using provided manual_cutoff_hz: {manual_cutoff_hz} Hz")
            final_cutoff_hz = manual_cutoff_hz
            # Store this in applied_params for clarity, even if no auto-estimation done
            applied_filter_params['estimation_method'] = 'manual_override'
        else:
            print(f"DEBUG: manual_cutoff_hz is 0 or not provided. Auto-calculating Butterworth cutoff.")
            avg_width_sec = estimate_peak_widths(signal, fs, prominence_threshold_butter_cutoff_calc, time_resolution)
            print(f'\nDEBUG: Average width of peaks (for Butterworth cutoff): {avg_width_sec} seconds')
            
            if avg_width_sec <= 1e-9: 
                print(f"Warning: avg_width_sec ({avg_width_sec:.2e}) is very small or zero. ")
                base_cutoff_hz = fs / 4.0
                print(f"Using a fallback base_cutoff_hz: {base_cutoff_hz:.2f} Hz due to small avg_width_sec.")
            else:
                base_cutoff_hz = 1.0 / avg_width_sec
            print(f'DEBUG: Base cutoff frequency (1/avg_width_sec): {base_cutoff_hz:.2f} Hz')
            
            normalized_cutoff = base_cutoff_hz * float(normalization_factor_butter)
            print(f'DEBUG: Normalized cutoff (base * norm_factor): {normalized_cutoff:.2f} Hz')
            final_cutoff_hz = normalized_cutoff
            applied_filter_params['estimated_avg_width_sec'] = avg_width_sec
            applied_filter_params['prominence_for_cutoff_calc'] = prominence_threshold_butter_cutoff_calc
            applied_filter_params['estimation_method'] = 'auto_peak_width'

        nyquist_hz = fs / 2.0
        final_cutoff_hz = min(final_cutoff_hz, nyquist_hz * 0.95)
        final_cutoff_hz = max(final_cutoff_hz, 0.01) 
        print(f'DEBUG: Final Butterworth cutoff to be applied: {final_cutoff_hz:.2f} Hz')

        filtered_signal = apply_butterworth_filter(butter_order, final_cutoff_hz, 'lowpass', fs, signal)
        print(f'DEBUG: Butterworth filter applied with order={butter_order}, cutoff_hz={final_cutoff_hz:.2f} Hz')
        
        applied_filter_params['cutoff_hz'] = final_cutoff_hz
        applied_filter_params['order'] = butter_order
        # Remove params specific to auto-estimation if manual was used, or ensure they are correctly set.
        # The logic above already sets 'estimated_avg_width_sec' etc. only in the auto path.

    elif filter_type == 'savgol':
        print(f"DEBUG: Savitzky-Golay mode selected.")
        current_savgol_window_length = savgol_window_length
        current_savgol_polyorder = savgol_polyorder

        if len(signal) < 3: # SavGol not applicable for very short signals
            print(f"ERROR: Signal length ({len(signal)}) is too short for Savitzky-Golay filter. Returning unfiltered signal.")
            filtered_signal = signal.copy() # Return original signal
            applied_filter_params['error'] = "Signal too short for Sav-Gol filter (min length 3)."
            applied_filter_params['window_length'] = None
            applied_filter_params['polyorder'] = None
        else:
            # Estimate window length if not provided
            if current_savgol_window_length is None:
                print(f"DEBUG: Sav-Gol window_length not provided. Estimating using prominence: {prominence_threshold_savgol_window_est}")
                s_peaks, _ = find_peaks(signal, prominence=prominence_threshold_savgol_window_est)
                if len(s_peaks) > 0:
                    s_width_results = peak_widths(signal, s_peaks, rel_height=0.5)
                    avg_s_width_samples = np.mean(s_width_results[0]) # Widths in samples
                    print(f"DEBUG: Estimated avg peak width for Sav-Gol: {avg_s_width_samples:.2f} samples")
                    
                    estimated_wl = int(np.ceil(avg_s_width_samples * 1.5)) # Heuristic: 1.5x avg peak width
                    current_savgol_window_length = estimated_wl
                    applied_filter_params['estimated_window_source'] = 'peak_width_heuristic'
                    applied_filter_params['estimated_avg_samples_width'] = avg_s_width_samples
                else: 
                    print(f"DEBUG: No peaks found for Sav-Gol window estimation (prominence {prominence_threshold_savgol_window_est}). Using fallback.")
                    fallback_wl = min(max(5, int(len(signal) * 0.05)), 101) # Fallback: 5% of signal, min 5, max 101
                    current_savgol_window_length = fallback_wl
                    applied_filter_params['estimated_window_source'] = 'fallback_no_peaks'
                print(f"DEBUG: Auto-determined Sav-Gol window_length: {current_savgol_window_length}")

            # Ensure window length is odd and positive
            if current_savgol_window_length < 3: current_savgol_window_length = 3
            if current_savgol_window_length % 2 == 0:
                current_savgol_window_length += 1
                print(f"Adjusted Sav-Gol window_length to be odd: {current_savgol_window_length}")

            # Determine polyorder if not provided
            if current_savgol_polyorder is None:
                # Default polyorder, ensure it's less than window_length. Common default is 2 or 3.
                current_savgol_polyorder = min(3, current_savgol_window_length - 1)
                current_savgol_polyorder = max(1, current_savgol_polyorder) # Must be at least 1
                print(f"DEBUG: Using default Sav-Gol polyorder: {current_savgol_polyorder}")
                applied_filter_params['polyorder_source'] = 'default'
            else: # User provided polyorder
                current_savgol_polyorder = max(1, int(savgol_polyorder))


            # Final validation of Sav-Gol parameters against signal length
            if current_savgol_window_length > len(signal):
                print(f"Warning: Sav-Gol window_length {current_savgol_window_length} > signal length {len(signal)}. Adjusting to signal length or slightly less.")
                current_savgol_window_length = len(signal) if len(signal) % 2 != 0 else len(signal) -1
                if current_savgol_window_length < 3 : current_savgol_window_length = 3 # if signal itself is very short
            
            if current_savgol_polyorder >= current_savgol_window_length:
                print(f"Warning: Sav-Gol polyorder {current_savgol_polyorder} >= window_length {current_savgol_window_length}. Adjusting polyorder.")
                current_savgol_polyorder = current_savgol_window_length - 1
            current_savgol_polyorder = max(1, current_savgol_polyorder) # Ensure polyorder is at least 1

            print(f"DEBUG: Final Sav-Gol params before filtering: window_length={current_savgol_window_length}, polyorder={current_savgol_polyorder}")
            
            if current_savgol_window_length > len(signal) or current_savgol_window_length <= current_savgol_polyorder or current_savgol_window_length < 1:
                 print(f"ERROR: Invalid Savitzky-Golay parameters for signal of length {len(signal)}: "
                       f"window_length={current_savgol_window_length}, polyorder={current_savgol_polyorder}. "
                       "Returning unfiltered signal.")
                 filtered_signal = signal.copy() # Return original signal
                 applied_filter_params['error'] = "Invalid Sav-Gol parameters (e.g., window too large, polyorder too large/small)."
                 applied_filter_params['window_length'] = current_savgol_window_length
                 applied_filter_params['polyorder'] = current_savgol_polyorder
            else:
                try:
                    filtered_signal = savgol_filter(signal, current_savgol_window_length, current_savgol_polyorder)
                    print(f'DEBUG: Savitzky-Golay filter applied.')
                except ValueError as e:
                    print(f"ERROR applying Savitzky-Golay filter: {e}. Returning unfiltered signal.")
                    filtered_signal = signal.copy() # Return original on error
                    applied_filter_params['error'] = str(e)

            applied_filter_params['window_length'] = current_savgol_window_length
            applied_filter_params['polyorder'] = current_savgol_polyorder
    
    else:
        print(f"ERROR: Unsupported filter_type: {filter_type}. Choose 'butterworth' or 'savgol'. Returning unfiltered signal.")
        filtered_signal = signal.copy() # Return original signal
        applied_filter_params['error'] = f"Unsupported filter_type: {filter_type}"

    print(f'DEBUG: Filter "{filter_type}" processing complete. Final effective params: {applied_filter_params}')
    print("==== DEBUG: Finished adjust_lowpass_cutoff ====\\n")

    return filtered_signal, applied_filter_params

def calculate_lowpass_cutoff(signal, fs, prominence_threshold, normalization_factor=1.0, time_resolution=1e-4):
    """
    Calculate appropriate cutoff frequency for lowpass filtering.
    
    Parameters:
        signal (numpy.ndarray): The signal to analyze
        fs (float): Sampling frequency of the signal
        prominence_threshold (float): Threshold for peak prominence or height
        normalization_factor (float, optional): Factor to normalize the cutoff. 
            Defaults to 1.0.
        time_resolution (float, optional): Time resolution in seconds per unit.
            Defaults to 1e-4 (0.1 milliseconds per unit).
            
    Returns:
        float: Calculated cutoff frequency in Hz
    """
    print("\n==== DEBUG: Starting calculate_lowpass_cutoff ====")
    print(f"DEBUG: Input parameters:")
    print(f"- fs: {fs}")
    print(f"- prominence_threshold: {prominence_threshold}")
    print(f"- normalization_factor: {normalization_factor}")
    print(f"- time_resolution: {time_resolution}")
    
    avg_width = estimate_peak_widths(signal, fs, prominence_threshold, time_resolution)
    print(f'\nDEBUG: Average width of peaks: {avg_width} seconds')
    
    # Calculate base cutoff frequency from average width (in Hz)
    base_cutoff = 1 / avg_width
    print(f'DEBUG: Base cutoff frequency (1/avg_width): {base_cutoff:.2f} Hz')
    
    # Apply normalization factor
    cutoff = base_cutoff * float(normalization_factor)
    print(f'DEBUG: After normalization:')
    print(f'- Base cutoff: {base_cutoff:.2f} Hz')
    print(f'- Normalization factor: {normalization_factor}')
    print(f'- Final cutoff: {cutoff:.2f} Hz')
    
    # Limit cutoff to Nyquist frequency
    nyquist = fs / 2.0
    cutoff = min(cutoff, nyquist * 0.95)  # Stay below 95% of Nyquist
    print(f'DEBUG: Nyquist frequency: {nyquist:.2f} Hz')
    print(f'DEBUG: Final cutoff (limited by Nyquist): {cutoff:.2f} Hz')
    print("==== DEBUG: Finished calculate_lowpass_cutoff ====\n")
    
    return cutoff

# =============================
# Baseline noise and SNR helpers
# =============================

def compute_baseline_mask(signal_length, peak_indices, widths_in_samples, multiplier=2.0, left_indices=None, right_indices=None):
    """
    Build a boolean mask marking baseline (non-peak) points by excluding windows around each peak.

    The excluded window for a peak is ±(multiplier × width) around the peak center, or
    an extension of (multiplier × width) beyond left/right indices if provided.

    Args:
        signal_length (int): Length of the signal array.
        peak_indices (np.ndarray): Indices of detected peaks.
        widths_in_samples (np.ndarray): Peak widths in samples (same length as peak_indices).
        multiplier (float, optional): Multiplier applied to width to define exclusion window. Defaults to 2.0.
        left_indices (np.ndarray, optional): Left interpolated positions for peaks.
        right_indices (np.ndarray, optional): Right interpolated positions for peaks.

    Returns:
        np.ndarray: Boolean mask of shape (signal_length,) where True indicates baseline points.
    """
    signal_length = int(signal_length)
    mask = np.ones(signal_length, dtype=bool)

    if peak_indices is None or widths_in_samples is None or len(peak_indices) == 0:
        return mask

    # Collect exclusion intervals [start, end) and merge overlaps
    intervals = []
    num_peaks = min(len(peak_indices), len(widths_in_samples))

    for i in range(num_peaks):
        center = int(peak_indices[i])
        width_samples = int(np.ceil(widths_in_samples[i]))
        if width_samples <= 0:
            width_samples = 1

        if left_indices is not None and right_indices is not None and i < len(left_indices) and i < len(right_indices):
            left_ip = int(np.floor(left_indices[i]))
            right_ip = int(np.ceil(right_indices[i]))
            span = max(1, right_ip - left_ip)
            extension = int(np.ceil(multiplier * span))
            start = max(0, left_ip - extension)
            end = min(signal_length, right_ip + extension)
        else:
            half_window = int(np.ceil(multiplier * width_samples))
            start = max(0, center - half_window)
            end = min(signal_length, center + half_window + 1)

        if start < end:
            intervals.append((start, end))

    if not intervals:
        return mask

    # Merge overlapping intervals
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:  # overlap
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    # Apply exclusions
    for s, e in merged:
        mask[s:e] = False

    return mask

def compute_noise_stats(baseline_signal):
    """
    Compute baseline noise statistics from a baseline-only signal segment.

    Returns standard deviation, robust MAD-based std, and baseline mean.

    Args:
        baseline_signal (np.ndarray): Signal values considered as baseline.

    Returns:
        tuple: (noise_std, noise_mad_std, baseline_mean)
    """
    if baseline_signal is None or len(baseline_signal) == 0:
        return 0.0, 0.0, 0.0

    baseline_signal = np.asarray(baseline_signal)
    baseline_mean = float(np.mean(baseline_signal))
    noise_std = float(np.std(baseline_signal))

    # Robust MAD-based estimator of sigma under normal assumption
    median_val = float(np.median(baseline_signal))
    mad = float(np.median(np.abs(baseline_signal - median_val)))
    noise_mad_std = 1.4826 * mad

    return noise_std, noise_mad_std, baseline_mean

def compute_snr_values(peak_heights, noise_value):
    """
    Compute per-peak SNR given peak heights and a scalar noise level.

    Args:
        peak_heights (np.ndarray): Vector of peak amplitudes (e.g., prominences).
        noise_value (float): Noise level (e.g., baseline std). Must be > 0.

    Returns:
        np.ndarray: Vector of SNR values (same shape as peak_heights). Empty if noise_value <= 0.
    """
    if peak_heights is None:
        return np.array([])
    if noise_value is None or noise_value <= 0:
        return np.array([])
    peak_heights = np.asarray(peak_heights)
    return peak_heights / float(noise_value)
