"""
Peak detection module for separating peak detection functionality from main application.

This module contains functions for detecting, analyzing, and processing
signal peaks in time series data.

Classes:
    PeakDetector: Main class for peak detection and analysis operations

Functions:
    calculate_auto_threshold: Automatically calculate a threshold for peak detection
"""

# Standard library
import os
import logging
import traceback
from functools import wraps

# Third-party libraries
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths

# Local imports
from .peak_analysis_utils import profile_function, find_peaks_with_window, find_nearest

class PeakDetector:
    """
    A class for detecting and analyzing peaks in time series data.
    
    This class provides methods for peak detection, analysis of peak characteristics
    (height, width, area), and conversion of peak data into structured formats.
    It implements various signal processing techniques to identify meaningful peaks
    and filter out noise.
    
    Attributes:
        logger (logging.Logger): Logger instance for recording operations
        peaks_indices (numpy.ndarray): Indices of detected peaks
        peaks_properties (dict): Properties of detected peaks including heights, widths
        peaks_data (pandas.DataFrame): Structured data containing peak information
        
    Example:
        >>> detector = PeakDetector()
        >>> detector.detect_peaks(signal, time_values, height_lim=0.5, distance=10)
        >>> results = detector.create_peak_dataframe(time_values)
    """
    
    def __init__(self, logger=None):
        """
        Initialize the PeakDetector with optional logger.
        
        Parameters:
            logger (logging.Logger, optional): Logger instance for recording operations.
                If None, a default logger will be used.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.reset()
    
    def reset(self):
        """
        Reset all detector state variables to their initial values.
        
        This method clears any previously detected peaks and their properties,
        allowing the detector to be reused for new data.
        """
        self.peaks_indices = None
        self.peaks_properties = {}
        self.peaks_data = None
        self.logger.debug("Peak detector reset")
    
    @profile_function
    def detect_peaks(self, signal, time_values, height_lim, distance, rel_height=0.5, width_range=None, time_resolution=1e-4, prominence_ratio=0.8):
        """
        Detect peaks in the provided signal data.
        
        This method identifies peaks in the signal that meet specified criteria
        for height, distance between peaks, and optionally width constraints.
        
        Parameters:
            signal (numpy.ndarray): The signal data to analyze
            time_values (numpy.ndarray): Corresponding time values for the signal
            height_lim (float): Minimum height threshold for peaks
            distance (int): Minimum number of samples between peaks
            rel_height (float, optional): Relative height at which peak width is measured.
                Defaults to 0.5 (half height).
            width_range (tuple, optional): Tuple of (min_width, max_width) in milliseconds to filter peaks.
                If None, no width filtering is applied.
            time_resolution (float, optional): Time resolution in seconds per unit.
                Defaults to 1e-4 (0.1 milliseconds per unit).
            prominence_ratio (float, optional): Threshold for the ratio of prominence to peak height.
                Peaks with ratio < threshold are filtered out as subpeaks.
                Higher values (e.g., 0.9) are more strict, keeping only very prominent peaks.
                Lower values (e.g., 0.5) are more permissive. Defaults to 0.8 (80%).
                
        Returns:
            tuple: (indices, properties) where:
                - indices is a numpy array containing the indices of detected peaks
                - properties is a dict containing peak properties (heights, widths, etc.)
                
        Notes:
            This method stores the detected peaks internally and they can be
            accessed via the peaks_indices and peaks_properties attributes.
        """
        try:
            # Use time_resolution directly instead of calculating from time differences
            rate = time_resolution  # Time between samples in seconds
            sampling_rate = 1 / rate  # Samples per second
            
            # Convert width range from milliseconds to samples
            if width_range:
                # Convert from milliseconds to samples using time resolution
                width_p = [int(float(value) * sampling_rate / 1000) for value in width_range]
                print(f"DEBUG - Width conversion in detect_peaks:")
                print(f"  Original width values (ms): {width_range}")
                print(f"  Time resolution: {time_resolution} seconds per unit")
                print(f"  Sampling rate: {sampling_rate:.1f} Hz")
                print(f"  Converted width_p (samples): {width_p}")
            else:
                width_p = None
            
            # Find peaks with specified parameters
            peaks, properties = find_peaks_with_window(
                signal, 
                width=width_p,
                prominence=height_lim,
                distance=distance, 
                rel_height=rel_height,
                prominence_ratio=prominence_ratio
            )
            
            # Store results
            self.peaks_indices = peaks
            self.peaks_properties = properties
            
            # Log results
            self.logger.info(f"Detected {len(peaks)} peaks")
            self.logger.info(f"Peak indices: {peaks[:10]}...")
            
            return peaks, properties
            
        except Exception as e:
            error_msg = f"Error in peak detection: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)
    
    @profile_function
    def calculate_peak_areas(self, signal, window_extension=40):
        """
        Calculate the area under each detected peak.
        
        This method computes the area under each peak by integrating the signal
        over a window around each peak. The window size is determined by the
        peak width and can be extended by the window_extension parameter.
        
        Parameters:
            signal (numpy.ndarray): The signal data to analyze
            window_extension (int, optional): Number of additional samples to include
                on each side of the peak for area calculation. Defaults to 40.
                
        Returns:
            numpy.ndarray: Array of calculated peak areas
            
        Raises:
            ValueError: If peaks have not been detected before calling this method
            
        Notes:
            The results are also stored in the peaks_properties dictionary
            under the key 'areas'.
        """
        if self.peaks_indices is None or self.peaks_properties is None:
            raise ValueError("No peaks detected. Run detect_peaks first.")
        
        try:
            peaks = self.peaks_indices
            events = len(peaks)
            
            # Extract peak widths and convert to integers for indexing
            window = np.round(self.peaks_properties['widths'], 0).astype(int) + window_extension
            
            # Initialize arrays for results
            peak_area = np.zeros(events)
            start = np.zeros(events)
            end = np.zeros(events)
            
            # Calculate area for each peak
            for i in range(events):
                # Determine window boundaries
                start_idx = max(0, peaks[i] - window[i])
                end_idx = min(len(signal), peaks[i] + window[i])
                
                # Extract data within window
                y_data = signal[start_idx:end_idx]
                
                # Find background level (minimum value in window)
                background = np.min(y_data)
                
                # Get start and end indices from peak properties
                st = int(self.peaks_properties["left_ips"][i])
                en = int(self.peaks_properties["right_ips"][i])
                
                # Store indices
                start[i] = st
                end[i] = en
                
                # Calculate area as sum of signal minus background
                peak_area[i] = np.sum(signal[st:en] - background)
            
            # Store results
            self.peaks_properties['areas'] = peak_area
            self.peaks_properties['start_indices'] = start
            self.peaks_properties['end_indices'] = end
            
            self.logger.info(f"Calculated {events} peak areas")
            
            return peak_area, start, end
            
        except Exception as e:
            error_msg = f"Error calculating peak areas: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)
    
    def calculate_peak_intervals(self, time_values, interval=10):
        """
        Calculate time intervals between consecutive peaks.
        
        Parameters:
            time_values (numpy.ndarray): Time values corresponding to the signal
            interval (float, optional): Not used in new implementation.
                Kept for backwards compatibility.
                
        Returns:
            numpy.ndarray: Array of peak times
            numpy.ndarray: Array of time intervals between consecutive peaks
            
        Notes:
            This method updates the peaks_properties dictionary with the calculated
            intervals under the key 'intervals'.
        """
        if self.peaks_indices is None:
            raise ValueError("No peaks detected. Run detect_peaks first.")
        
        try:
            # Get time values at peak positions
            peak_times = time_values[self.peaks_indices]
            
            # Calculate intervals between consecutive peaks
            # First interval is the time to first peak
            peaks_interval = np.zeros_like(peak_times)
            peaks_interval[0] = peak_times[0]
            
            # Remaining intervals are differences between consecutive peaks
            if len(peak_times) > 1:
                peaks_interval[1:] = np.diff(peak_times)
            
            # Store results
            self.peaks_properties['intervals'] = peaks_interval
            
            self.logger.info(f"Calculated {len(peaks_interval)} peak intervals")
            
            return peak_times, peaks_interval
            
        except Exception as e:
            error_msg = f"Error calculating peak intervals: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)
    
    def create_peak_dataframe(self, time_values, protocol_info=None, time_resolution=1e-4):
        """
        Create a DataFrame containing peak detection results.
        
        This method organizes all detected peak information into a structured DataFrame,
        optionally adding protocol information.
        
        Parameters:
            time_values (numpy.ndarray): Time values corresponding to the signal
            protocol_info (dict, optional): Additional protocol information to include
                in the DataFrame. Defaults to None.
            time_resolution (float, optional): Time resolution in seconds per unit.
                Defaults to 1e-4 (0.1 milliseconds per unit).
                
        Returns:
            pandas.DataFrame: DataFrame containing peak information including:
                - Peak Index: The index of each peak in the original signal
                - Time: The time at which each peak occurs (in minutes)
                - Height: Peak heights (prominences)
                - Width: Peak widths (in milliseconds)
                - Area: Area under each peak
                - Interval: Time intervals between consecutive peaks
                - Any additional protocol information provided
                
        Notes:
            This method requires that peak detection and area calculation have been
            performed beforehand.
        """
        # Validate inputs
        if self.peaks_indices is None:
            raise ValueError("No peaks detected. Run detect_peaks first.")
            
        if 'prominences' not in self.peaks_properties:
            raise ValueError("Peak prominences not available")
            
        if 'widths' not in self.peaks_properties:
            raise ValueError("Peak widths not available")
            
        if 'areas' not in self.peaks_properties:
            raise ValueError("Peak areas not available. Run calculate_peak_areas first.")
            
        if 'intervals' not in self.peaks_properties:
            self.calculate_peak_intervals(time_values)
            
        try:
            # Check if all arrays have the same length
            expected_length = len(self.peaks_indices)
            
            # Validate array lengths
            props_to_check = ['prominences', 'widths', 'areas', 'intervals']
            for prop in props_to_check:
                if prop in self.peaks_properties:
                    actual_length = len(self.peaks_properties[prop])
                    if actual_length != expected_length:
                        error_msg = f"Array length mismatch: {prop} has length {actual_length}, expected {expected_length}"
                        self.logger.error(error_msg)
                        raise ValueError(error_msg)
            
            # Use time_resolution directly instead of calculating from time differences
            rate = time_resolution  # Time between samples in seconds
            
            # Create basic DataFrame with peak measurements
            results_df = pd.DataFrame({
                "Peak Index": self.peaks_indices,
                "Time (s)": time_values[self.peaks_indices] / 60,  # Convert to minutes
                "Height": self.peaks_properties['prominences'],
                "Width (ms)": self.peaks_properties['widths'] * rate * 1000,  # Convert from samples to milliseconds
                "Area": self.peaks_properties['areas'],
                "Interval": self.peaks_properties['intervals']
            })
            
            # Add protocol information if provided
            if protocol_info is not None:
                for key, value in protocol_info.items():
                    results_df[key] = value
            
            # Store results
            self.peaks_data = results_df
            
            return results_df
            
        except Exception as e:
            error_msg = f"Error creating peak DataFrame: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)

# Function to calculate auto threshold
def calculate_auto_threshold(signal, sigma_multiplier=5):
    """
    Automatically calculate a threshold for peak detection based on signal statistics.
    
    This function calculates a threshold by finding the mean and standard deviation
    of the signal, then setting the threshold at mean + (sigma_multiplier * std_dev).
    
    Parameters:
        signal (numpy.ndarray): The signal data to analyze
        sigma_multiplier (float, optional): Number of standard deviations above
            the mean to set the threshold. Defaults to 5.
            
    Returns:
        float: The calculated threshold value
        
    Notes:
        This method is useful for automated peak detection when the appropriate
        threshold is not known in advance.
        
    Example:
        >>> threshold = calculate_auto_threshold(signal, sigma_multiplier=3)
        >>> detector.detect_peaks(signal, time_values, height_lim=threshold, distance=10)
    """
    if signal is None or len(signal) == 0:
        raise ValueError("Signal is empty or None")
        
    # Calculate standard deviation
    signal_std = np.std(signal)
    
    # Calculate threshold as sigma_multiplier times standard deviation
    suggested_threshold = sigma_multiplier * signal_std
    
    return suggested_threshold 