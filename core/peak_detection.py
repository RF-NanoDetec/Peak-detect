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
    def detect_peaks(self, signal, time_values, height_lim, distance, rel_height=0.5, width_range=None):
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
            width_range (tuple, optional): Tuple of (min_width, max_width) to filter peaks.
                If None, no width filtering is applied.
                
        Returns:
            tuple: (indices, properties) where:
                - indices is a numpy array containing the indices of detected peaks
                - properties is a dict containing peak properties (heights, widths, etc.)
                
        Notes:
            This method stores the detected peaks internally and they can be
            accessed via the peaks_indices and peaks_properties attributes.
        """
        try:
            # Convert width range to milliseconds
            width_p = [int(float(value) * 10) for value in width_range] if width_range else None
            
            # Find peaks with specified parameters
            peaks, properties = find_peaks_with_window(
                signal, 
                width=width_p,
                prominence=height_lim,
                distance=distance, 
                rel_height=rel_height
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
            interval (float, optional): Time interval for grouping peaks.
                Defaults to 10 seconds.
                
        Returns:
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
            
            # Create position array
            position = np.arange(1, int(max(time_values)), interval)
            points = len(position)
            
            # Initialize arrays for results
            peaks = np.zeros(points)
            peaks_interval = np.zeros(points)
            
            # Calculate peaks and intervals
            for i, pos in enumerate(position):
                peaks[i] = find_nearest(peak_times, pos)
                
                if i == 0:
                    peaks_interval[i] = peaks[i]
                else:
                    peaks_interval[i] = peaks[i] - peaks[i - 1]
            
            # Store results
            self.peaks_properties['intervals'] = peaks_interval
            
            self.logger.info(f"Calculated {points} peak intervals")
            
            return peaks, peaks_interval
            
        except Exception as e:
            error_msg = f"Error calculating peak intervals: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)
    
    def create_peak_dataframe(self, time_values, protocol_info=None):
        """
        Create a pandas DataFrame containing all peak information.
        
        This method compiles all detected peak properties into a structured
        DataFrame for easier analysis and export.
        
        Parameters:
            time_values (numpy.ndarray): Time values corresponding to the signal
            protocol_info (dict, optional): Additional protocol information to include
                in the DataFrame. Defaults to None.
                
        Returns:
            pandas.DataFrame: DataFrame containing peak information with columns:
                - 'Peak Index': Index of the peak in the original signal
                - 'Time (s)': Time of the peak occurrence
                - 'Height': Peak height
                - 'Width': Peak width at half-height
                - 'Area': Area under the peak
                - 'Interval': Time since previous peak
                - Additional columns from protocol_info if provided
                
        Raises:
            ValueError: If peaks have not been detected before calling this method
        """
        if self.peaks_indices is None or self.peaks_properties is None or self.peaks_properties.get('areas') is None:
            raise ValueError("Missing peak data. Ensure detect_peaks and calculate_peak_areas have been run.")
        
        try:
            # Create basic DataFrame with peak measurements
            results_df = pd.DataFrame({
                "Peak Index": self.peaks_indices,
                "Time (s)": time_values[self.peaks_indices] / 60,  # Convert to minutes
                "Height": self.peaks_properties['prominences'],
                "Width": self.peaks_properties['widths'] / 10,  # Convert to milliseconds
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