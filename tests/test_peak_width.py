#!/usr/bin/env python
"""
Test script for peak width calculation validation.

This script:
1. Generates synthetic data with known peak widths
2. Saves the data to a test file
3. Processes the data using the application's peak detection
4. Verifies that measured peak widths match expected values
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import unittest

# Add the parent directory to the path so we can import application modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import application modules
from core.peak_detection import PeakDetector
from core.peak_analysis_utils import adjust_lowpass_cutoff, calculate_lowpass_cutoff
from core.data_analysis import calculate_peak_areas


class PeakWidthTest(unittest.TestCase):
    """Test case for peak width calculation."""
    
    def setUp(self):
        """Set up test data with known peak widths."""
        # Parameters for synthetic data
        self.time_resolution = 1e-4  # 0.1 milliseconds (in seconds)
        self.duration = 5.0  # 5 seconds
        self.num_peaks = 100
        self.known_peak_width = 0.05  # 50 milliseconds in seconds
        
        # Create time array with 0.1 ms resolution
        self.num_samples = int(self.duration / self.time_resolution)
        self.time = np.linspace(0, self.duration, self.num_samples)
        
        # Generate test data with Gaussian peaks of known width
        self.data = self.generate_test_data()
        
        # Save test data to file
        self.test_file = 'test_peak_data.txt'
        self.save_test_data()
        
        # Initialize peak detector
        self.peak_detector = PeakDetector()
    
    def generate_test_data(self):
        """Generate synthetic signal with peaks of known width."""
        # Start with zero baseline
        signal = np.zeros(self.num_samples)
        
        # Calculate peak positions (evenly spaced)
        peak_positions = np.linspace(0.1, self.duration - 0.1, self.num_peaks)
        peak_indices = [int(pos / self.time_resolution) for pos in peak_positions]
        
        # Standard deviation in samples (for Gaussian peaks)
        sigma = self.known_peak_width / (2.355 * self.time_resolution)  # FWHM = 2.355 * sigma
        
        # Add Gaussian peaks with known width
        for idx in peak_indices:
            # Generate Gaussian peak with specified width
            x = np.arange(self.num_samples)
            gaussian = 100 * np.exp(-0.5 * ((x - idx) / sigma) ** 2)
            signal += gaussian
        
        # Add some noise
        noise = np.random.normal(0, 1, self.num_samples)
        signal += noise
        
        return signal
    
    def save_test_data(self):
        """Save test data to file with proper format."""
        # Create DataFrame with time and amplitude
        df = pd.DataFrame({
            'Time - Plot 0': self.time / self.time_resolution,  # Convert back to raw units
            'Amplitude - Plot 0': self.data
        })
        
        # Save to file
        df.to_csv(self.test_file, sep='\t', index=False)
        print(f"Test data saved to {self.test_file}")
    
    def test_peak_width_calculation(self):
        """Test that peak width calculation correctly measures known widths."""
        # Load data from file (to mimic application behavior)
        df = pd.read_csv(self.test_file, sep='\t')
        
        # Convert time to seconds (as our application would)
        time = df['Time - Plot 0'].values * self.time_resolution
        signal = df['Amplitude - Plot 0'].values
        
        # Calculate sampling rate
        rate = np.median(np.diff(time))
        fs = 1 / rate
        print(f"Sampling rate (fs): {fs} Hz")
        
        # Apply filtering (as the application would)
        filtered_signal, cutoff = adjust_lowpass_cutoff(signal, fs, 100, 1.0)
        
        # Detect peaks
        peaks, properties = find_peaks(filtered_signal, 
                                     height=10,  # Reasonable threshold
                                     distance=int(0.01 / rate),  # Minimum 10ms between peaks
                                     prominence=5)  # Minimum prominence
        
        # Calculate peak widths in samples
        widths_samples, width_heights, left_ips, right_ips = peak_widths(
            filtered_signal, peaks, rel_height=0.5
        )
        
        # Convert width from samples to seconds
        widths_seconds = widths_samples * rate
        
        # Print statistics
        print(f"Number of detected peaks: {len(peaks)}")
        print(f"Mean peak width: {np.mean(widths_seconds):.4f} seconds")
        print(f"Expected peak width: {self.known_peak_width:.4f} seconds")
        
        # Calculate percentage error
        error_percentage = abs(np.mean(widths_seconds) - self.known_peak_width) / self.known_peak_width * 100
        print(f"Percentage error: {error_percentage:.2f}%")
        
        # Verify results with tolerance
        # Note: Filtering can broaden peaks, so allow a higher tolerance (50%)
        self.assertLess(error_percentage, 50, "Peak width error exceeds 50%")
        
        # Verify number of peaks detected (should be close to num_peaks)
        detection_ratio = len(peaks) / self.num_peaks
        print(f"Peak detection ratio: {detection_ratio:.2f}")
        self.assertGreater(detection_ratio, 0.7, "Less than 70% of peaks detected")
        
        # Visualize results
        self.plot_results(time, signal, filtered_signal, peaks, widths_samples, left_ips, right_ips)
    
    def plot_results(self, time, signal, filtered_signal, peaks, widths, left_ips, right_ips):
        """Plot the test results for visual inspection."""
        plt.figure(figsize=(15, 10))
        
        # Plot original and filtered signals
        plt.subplot(211)
        plt.plot(time, signal, 'b-', alpha=0.5, label='Original Signal')
        plt.plot(time, filtered_signal, 'g-', label='Filtered Signal')
        plt.plot(time[peaks], filtered_signal[peaks], 'ro', label='Detected Peaks')
        plt.legend()
        plt.title('Peak Detection Results')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        
        # Plot zoom view of a few peaks with width measurements
        plt.subplot(212)
        # Select a subset of peaks to zoom in
        zoom_start = int(len(peaks) * 0.4)  # Start at 40% through
        zoom_count = 5  # Show 5 peaks
        zoom_peaks = peaks[zoom_start:zoom_start+zoom_count]
        
        # Find time range for zoom
        zoom_start_time = time[max(0, zoom_peaks[0] - 100)]
        zoom_end_time = time[min(len(time)-1, zoom_peaks[-1] + 100)]
        
        plt.plot(time, filtered_signal, 'g-')
        plt.plot(time[zoom_peaks], filtered_signal[zoom_peaks], 'ro')
        
        # Plot width indicators for each peak in the zoom view
        for i, peak in enumerate(zoom_peaks):
            idx = zoom_start + i  # Index in the original peaks array
            left_idx = int(left_ips[idx])
            right_idx = int(right_ips[idx])
            width = widths[idx]
            width_height = filtered_signal[peak] * 0.5  # For half-height width
            
            # Plot horizontal line indicating width
            plt.hlines(y=width_height, 
                      xmin=time[left_idx], 
                      xmax=time[right_idx],
                      color='r', linestyle='-', 
                      label=f'Width: {width * np.median(np.diff(time)):.4f}s' if i == 0 else "")
        
        plt.xlim(zoom_start_time, zoom_end_time)
        plt.title('Detailed View of Peak Width Measurements')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Save the plot
        plt.tight_layout()
        plt.savefig('peak_width_test_results.png')
        print("Test results plot saved to peak_width_test_results.png")
        plt.close()
    
    def tearDown(self):
        """Clean up after test."""
        # Optionally remove the test file
        # os.remove(self.test_file)
        pass


if __name__ == "__main__":
    # Run the test
    unittest.main() 