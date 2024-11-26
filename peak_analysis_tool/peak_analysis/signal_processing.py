"""
Signal processing module for the Peak Analysis Tool.
Provides advanced signal processing functionality including filtering and analysis.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter
from typing import Tuple, Optional, Dict, Union
import logging
from dataclasses import dataclass
from numba import njit

from .config import config
from .utils.decorators import profile_function

logger = logging.getLogger(__name__)

@dataclass
class FilterParameters:
    """Parameters for signal filtering."""
    order: int
    cutoff_freq: float
    filter_type: str = 'lowpass'
    sampling_rate: float = 1000.0
    
    def validate(self):
        """Validate filter parameters."""
        if self.order < 1:
            raise ValueError("Filter order must be >= 1")
        if self.cutoff_freq <= 0:
            raise ValueError("Cutoff frequency must be > 0")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be > 0")
        if self.cutoff_freq >= self.sampling_rate / 2:
            raise ValueError("Cutoff frequency must be < sampling_rate/2")

class SignalProcessor:
    """Class for signal processing operations."""
    
    def __init__(self):
        """Initialize signal processor."""
        self.signal: Optional[np.ndarray] = None
        self.time: Optional[np.ndarray] = None
        self.sampling_rate: Optional[float] = None
        self.filtered_signal: Optional[np.ndarray] = None
    
    @staticmethod
    @njit
    def find_nearest(array: np.ndarray, value: float) -> int:
        """
        Find index of nearest value in array.
        
        Args:
            array: Input array
            value: Value to find
            
        Returns:
            Index of nearest value
        """
        idx = 0
        min_diff = np.abs(array[0] - value)
        for i in range(1, len(array)):
            diff = np.abs(array[i] - value)
            if diff < min_diff:
                min_diff = diff
                idx = i
        return idx
    
    @profile_function
    def apply_butterworth_filter(
        self,
        signal: np.ndarray,
        params: FilterParameters
    ) -> np.ndarray:
        """
        Apply Butterworth filter to signal.
        
        Args:
            signal: Input signal array
            params: Filter parameters
            
        Returns:
            Filtered signal array
        """
        try:
            # Validate parameters
            params.validate()
            
            # Design filter
            nyquist = params.sampling_rate / 2
            normalized_cutoff = params.cutoff_freq / nyquist
            b, a = butter(
                params.order,
                normalized_cutoff,
                btype=params.filter_type,
                analog=False
            )
            
            # Apply filter
            filtered = filtfilt(b, a, signal)
            
            logger.info(
                f"Applied Butterworth filter: "
                f"order={params.order}, "
                f"cutoff={params.cutoff_freq:.1f}Hz"
            )
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error in Butterworth filtering: {str(e)}")
            raise
    
    @profile_function
    def apply_savgol_filter(
        self,
        signal: np.ndarray,
        window_length: int,
        poly_order: int
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to signal.
        
        Args:
            signal: Input signal array
            window_length: Length of filter window
            poly_order: Order of polynomial
            
        Returns:
            Filtered signal array
        """
        try:
            if window_length % 2 == 0:
                window_length += 1
            
            filtered = savgol_filter(signal, window_length, poly_order)
            
            logger.info(
                f"Applied Savitzky-Golay filter: "
                f"window={window_length}, "
                f"order={poly_order}"
            )
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error in Savitzky-Golay filtering: {str(e)}")
            raise
    
    def estimate_optimal_cutoff(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        noise_level: Optional[float] = None
    ) -> float:
        """
        Estimate optimal cutoff frequency for filtering.
        
        Args:
            signal: Input signal array
            sampling_rate: Sampling rate of signal
            noise_level: Optional known noise level
            
        Returns:
            Estimated cutoff frequency
        """
        try:
            # Compute power spectrum
            freqs, psd = signal.welch(
                signal,
                fs=sampling_rate,
                nperseg=min(len(signal), 1024)
            )
            
            if noise_level is None:
                # Estimate noise level from high frequencies
                noise_level = np.mean(psd[-int(len(psd)/5):])
            
            # Find frequency where power drops to noise level
            signal_freqs = freqs[psd > 2*noise_level]
            if len(signal_freqs) > 0:
                cutoff = signal_freqs[-1]
            else:
                cutoff = sampling_rate / 4
            
            logger.info(f"Estimated optimal cutoff frequency: {cutoff:.1f}Hz")
            return cutoff
            
        except Exception as e:
            logger.error(f"Error estimating cutoff frequency: {str(e)}")
            raise
    
    @profile_function
    def remove_baseline(
        self,
        signal: np.ndarray,
        window_size: int = 1001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove signal baseline using rolling minimum.
        
        Args:
            signal: Input signal array
            window_size: Size of rolling window
            
        Returns:
            Tuple of (baseline-corrected signal, baseline)
        """
        try:
            if window_size % 2 == 0:
                window_size += 1
            
            # Estimate baseline
            from scipy.ndimage import minimum_filter1d
            baseline = minimum_filter1d(signal, size=window_size)
            
            # Remove baseline
            corrected = signal - baseline
            
            logger.info(f"Removed baseline with window size {window_size}")
            return corrected, baseline
            
        except Exception as e:
            logger.error(f"Error in baseline removal: {str(e)}")
            raise
    
    def calculate_snr(
        self,
        signal: np.ndarray,
        noise_region: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            signal: Input signal array
            noise_region: Optional tuple of (start, end) indices for noise estimation
            
        Returns:
            Calculated SNR value
        """
        try:
            if noise_region is None:
                # Use last 10% of signal for noise estimation
                noise_start = int(0.9 * len(signal))
                noise_region = (noise_start, len(signal))
            
            noise = signal[noise_region[0]:noise_region[1]]
            noise_power = np.mean(noise**2)
            signal_power = np.mean(signal**2)
            
            snr = 10 * np.log10(signal_power / noise_power)
            
            logger.info(f"Calculated SNR: {snr:.1f}dB")
            return snr
            
        except Exception as e:
            logger.error(f"Error calculating SNR: {str(e)}")
            raise
    
    @profile_function
    def process_signal(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sampling_rate: float,
        filter_params: Optional[FilterParameters] = None,
        remove_baseline: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Complete signal processing pipeline.
        
        Args:
            signal: Input signal array
            time: Time array
            sampling_rate: Sampling rate of signal
            filter_params: Optional filter parameters
            remove_baseline: Whether to remove baseline
            
        Returns:
            Dictionary containing processed signals
        """
        try:
            self.signal = signal
            self.time = time
            self.sampling_rate = sampling_rate
            
            results = {
                'original': signal,
                'time': time
            }
            
            # Remove baseline if requested
            if remove_baseline:
                corrected, baseline = self.remove_baseline(signal)
                results['baseline_corrected'] = corrected
                results['baseline'] = baseline
                working_signal = corrected
            else:
                working_signal = signal
            
            # Apply filtering if parameters provided
            if filter_params is None:
                # Estimate optimal parameters
                cutoff = self.estimate_optimal_cutoff(working_signal, sampling_rate)
                filter_params = FilterParameters(
                    order=2,
                    cutoff_freq=cutoff,
                    sampling_rate=sampling_rate
                )
            
            filtered = self.apply_butterworth_filter(working_signal, filter_params)
            results['filtered'] = filtered
            self.filtered_signal = filtered
            
            # Calculate SNR
            snr = self.calculate_snr(filtered)
            results['snr'] = snr
            
            logger.info("Completed signal processing pipeline")
            return results
            
        except Exception as e:
            logger.error(f"Error in signal processing pipeline: {str(e)}")
            raise
    
    def plot_frequency_response(self, params: FilterParameters) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate filter frequency response.
        
        Args:
            params: Filter parameters
            
        Returns:
            Tuple of (frequencies, response)
        """
        try:
            params.validate()
            
            w, h = signal.freqz(*butter(
                params.order,
                params.cutoff_freq / (params.sampling_rate/2),
                btype=params.filter_type,
                analog=False
            ))
            
            freqs = w * params.sampling_rate / (2 * np.pi)
            response = 20 * np.log10(np.abs(h))
            
            return freqs, response
            
        except Exception as e:
            logger.error(f"Error calculating frequency response: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create processor
    processor = SignalProcessor()
    
    # Generate example signal
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(10 * 2 * np.pi * t) + np.random.normal(0, 0.1, len(t))
    
    try:
        # Process signal
        results = processor.process_signal(
            signal=signal,
            time=t,
            sampling_rate=100,
            remove_baseline=True
        )
        
        print(f"SNR: {results['snr']:.1f}dB")
        
    except Exception as e:
        print(f"Error: {str(e)}")
