"""
Peak detection module for separating peak detection functionality from main application.

This module contains functions for detecting, analyzing, and processing
signal peaks in time series data.
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
    Class for detecting and analyzing peaks in signal data.
    
    This class encapsulates peak detection methods and maintains state
    between operations for improved performance and usability.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the peak detector.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger for recording operational information
        """
        self.logger = logger or logging.getLogger(__name__)
        self.reset()
    
    def reset(self):
        """Reset all internal state variables"""
        self.peaks = None
        self.properties = None
        self.peak_area = None
        self.start_indices = None
        self.end_indices = None
        self.intervals = None
        self.peak_results = None
    
    @profile_function
    def detect_peaks(self, signal, time_values, height_lim, distance, rel_height, width_range):
        """
        Detect peaks in the given signal.
        
        Parameters
        ----------
        signal : ndarray
            Signal data to analyze
        time_values : ndarray
            Time values corresponding to the signal data
        height_lim : float
            Threshold for peak detection
        distance : int
            Minimum distance between peaks
        rel_height : float
            Relative height for width calculation
        width_range : list
            Min and max width values in the format [min, max]
            
        Returns
        -------
        tuple
            (peaks_indices, properties) detected peaks and their properties
        """
        try:
            # Convert width range to milliseconds
            width_p = [int(float(value) * 10) for value in width_range]
            
            # Find peaks with specified parameters
            peaks, properties = find_peaks_with_window(
                signal, 
                width=width_p,
                prominence=height_lim,
                distance=distance, 
                rel_height=rel_height
            )
            
            # Store results
            self.peaks = peaks
            self.properties = properties
            
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
        Calculate areas under detected peaks.
        
        Parameters
        ----------
        signal : ndarray
            Signal data to analyze
        window_extension : int, optional
            Number of points to extend the window on each side
            
        Returns
        -------
        tuple
            (area, start_indices, end_indices) for each peak
        """
        if self.peaks is None or self.properties is None:
            raise ValueError("No peaks detected. Run detect_peaks first.")
        
        try:
            peaks = self.peaks
            events = len(peaks)
            
            # Extract peak widths and convert to integers for indexing
            window = np.round(self.properties['widths'], 0).astype(int) + window_extension
            
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
                st = int(self.properties["left_ips"][i])
                en = int(self.properties["right_ips"][i])
                
                # Store indices
                start[i] = st
                end[i] = en
                
                # Calculate area as sum of signal minus background
                peak_area[i] = np.sum(signal[st:en] - background)
            
            # Store results
            self.peak_area = peak_area
            self.start_indices = start
            self.end_indices = end
            
            self.logger.info(f"Calculated {events} peak areas")
            
            return peak_area, start, end
            
        except Exception as e:
            error_msg = f"Error calculating peak areas: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)
    
    def calculate_peak_intervals(self, time_values, interval=10):
        """
        Calculate time intervals between detected peaks.
        
        Parameters
        ----------
        time_values : ndarray
            Time values corresponding to the signal data
        interval : int, optional
            Interval size for analysis
            
        Returns
        -------
        tuple
            (peaks, intervals) peak positions and intervals
        """
        if self.peaks is None:
            raise ValueError("No peaks detected. Run detect_peaks first.")
        
        try:
            # Get time values at peak positions
            peak_times = time_values[self.peaks]
            
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
            self.intervals = peaks_interval
            
            self.logger.info(f"Calculated {points} peak intervals")
            
            return peaks, peaks_interval
            
        except Exception as e:
            error_msg = f"Error calculating peak intervals: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)
    
    def create_peak_dataframe(self, time_values, protocol_info=None):
        """
        Create a DataFrame with all peak information.
        
        Parameters
        ----------
        time_values : ndarray
            Time values corresponding to the signal data
        protocol_info : dict, optional
            Dictionary with protocol information
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with all peak information
        """
        if self.peaks is None or self.properties is None or self.peak_area is None:
            raise ValueError("Missing peak data. Ensure detect_peaks and calculate_peak_areas have been run.")
        
        try:
            # Create basic DataFrame with peak measurements
            results_df = pd.DataFrame({
                "Peak Time (min)": time_values[self.peaks] / 60,  # Convert to minutes
                "Peak Area": self.peak_area,
                "Peak Width (ms)": self.properties['widths'] / 10,  # Convert to milliseconds
                "Peak Height": self.properties['prominences'],
            })
            
            # Add protocol information if provided
            if protocol_info is not None:
                for key, value in protocol_info.items():
                    results_df[key] = value
            
            # Store results
            self.peak_results = results_df
            
            return results_df
            
        except Exception as e:
            error_msg = f"Error creating peak DataFrame: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise ValueError(error_msg)

# Function to calculate auto threshold
def calculate_auto_threshold(signal, sigma_multiplier=5):
    """
    Calculate automatic threshold for peak detection.
    
    Parameters
    ----------
    signal : ndarray
        Signal data to analyze
    sigma_multiplier : float, optional
        Multiplier for standard deviation
        
    Returns
    -------
    float
        Suggested threshold value
    """
    if signal is None or len(signal) == 0:
        raise ValueError("Signal is empty or None")
        
    # Calculate standard deviation
    signal_std = np.std(signal)
    
    # Calculate threshold as sigma_multiplier times standard deviation
    suggested_threshold = sigma_multiplier * signal_std
    
    return suggested_threshold 