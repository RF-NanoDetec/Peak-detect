#!/usr/bin/env python
"""
Test script for time resolution handling.

This script tests if the time resolution is correctly applied when loading
data and carrying through to peak width calculations.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import unittest

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import application modules for direct testing
from core.peak_detection import PeakDetector
from core.peak_analysis_utils import adjust_lowpass_cutoff, calculate_lowpass_cutoff, estimate_peak_widths
from core.file_handler import load_single_file


class TimeResolutionTest(unittest.TestCase):
    """Test case for time resolution handling throughout the application."""
    
    def setUp(self):
        """Set up test data with different time resolutions."""
        # Generate test data with 3 different time resolutions
        self.resolutions = {
            'standard': 1e-4,  # 0.1 ms (default)
            'fine': 1e-5,      # 0.01 ms
            'coarse': 1e-3     # 1 ms
        }
        
        self.duration = 1.0  # 1 second duration
        self.peak_width = 0.05  # 50 ms peak width
        
        # Create test files with different resolutions
        self.test_files = {}
        
        for name, resolution in self.resolutions.items():
            self.test_files[name] = self.create_test_file(resolution)
    
    def create_test_file(self, time_resolution):
        """Create a test file with specified time resolution."""
        # Calculate number of samples
        num_samples = int(self.duration / time_resolution)
        time = np.linspace(0, self.duration, num_samples)
        
        # Create signal with a single peak of known width
        signal = np.zeros(num_samples)
        
        # Add a single peak at 0.5 seconds
        peak_center = int(0.5 / time_resolution)
        sigma = self.peak_width / (2.355 * time_resolution)  # FWHM = 2.355 * sigma
        
        x = np.arange(num_samples)
        gaussian = 100 * np.exp(-0.5 * ((x - peak_center) / sigma) ** 2)
        signal += gaussian
        
        # Add some noise
        noise = np.random.normal(0, 1, num_samples)
        signal += noise
        
        # Create test file name
        filename = f'test_time_res_{time_resolution:.1e}.txt'
        
        # Save to file (using the raw time units)
        df = pd.DataFrame({
            'Time - Plot 0': time / time_resolution,  # Convert to raw units
            'Amplitude - Plot 0': signal
        })
        df.to_csv(filename, sep='\t', index=False)
        
        print(f"Created test file {filename} with {num_samples} samples at {time_resolution:.1e}s resolution")
        return filename
    
    def test_time_resolution_loading(self):
        """Test that time resolution is correctly applied when loading data."""
        # For each resolution, test loading with correct and incorrect resolution values
        for name, resolution in self.resolutions.items():
            file_path = self.test_files[name]
            
            # Test loading with correct resolution
            correct_data = load_single_file(file_path, time_resolution=resolution)
            
            # Get time array and check ranges
            time_correct = correct_data['time']
            
            # Time should start near 0 and end near the duration
            self.assertLess(time_correct[0], 0.001, f"Time should start near 0 for {name} resolution")
            self.assertAlmostEqual(time_correct[-1], self.duration, delta=0.01, 
                                  msg=f"Time should end near {self.duration} for {name} resolution")
            
            # Test loading with incorrect resolution (10x too small)
            wrong_res = resolution / 10
            incorrect_data = load_single_file(file_path, time_resolution=wrong_res)
            time_incorrect = incorrect_data['time']
            
            # When using wrong resolution, the time values will be 10x smaller than they should be
            # This means the end time will be duration/10 instead of duration
            self.assertAlmostEqual(time_incorrect[-1], self.duration / 10, delta=0.01,
                                 msg=f"When using wrong resolution ({wrong_res}), time range should be smaller")
            
            print(f"\nResolution: {name} ({resolution:.1e})")
            print(f"  Correct time range: {time_correct[0]:.4f} - {time_correct[-1]:.4f} seconds")
            print(f"  Incorrect time range: {time_incorrect[0]:.4f} - {time_incorrect[-1]:.4f} seconds")
            
            # Calculate sampling rates
            correct_rate = np.median(np.diff(time_correct))
            incorrect_rate = np.median(np.diff(time_incorrect))
            
            print(f"  Correct sampling rate: {correct_rate:.8f} seconds")
            print(f"  Incorrect sampling rate: {incorrect_rate:.8f} seconds")
            
            # Verify that sampling rates differ by factor of 10
            self.assertAlmostEqual(incorrect_rate / correct_rate, 0.1, delta=0.01,
                                 msg="Sampling rates should differ by factor of 10")
    
    def test_peak_width_with_resolution(self):
        """Test that peak width calculation respects time resolution."""
        results = {}
        
        # For each resolution, measure the peak width
        for name, resolution in self.resolutions.items():
            file_path = self.test_files[name]
            
            # Load data with correct resolution
            data = load_single_file(file_path, time_resolution=resolution)
            time = data['time']
            signal = data['amplitude']
            
            # Calculate sampling rate
            rate = np.median(np.diff(time))
            fs = 1 / rate
            
            # Apply filtering
            filtered_signal, cutoff = adjust_lowpass_cutoff(signal, fs, 100, 1.0)
            
            # Find peak
            peaks, properties = find_peaks(filtered_signal, height=10, prominence=5)
            
            # Should find exactly one peak
            self.assertEqual(len(peaks), 1, f"Should find exactly 1 peak with {name} resolution")
            
            # Calculate peak width using scipy peak_widths
            widths_samples, width_heights, left_ips, right_ips = peak_widths(
                filtered_signal, peaks, rel_height=0.5
            )
            
            # Convert width from samples to seconds
            width_seconds = widths_samples[0] * rate
            
            # Calculate width using our application's function
            width_params = "50,100"  # Set a reasonable width parameter range
            app_width = estimate_peak_widths(
                filtered_signal, fs, big_counts=10  # Use appropriate big_counts value
            )
            
            # Store results
            results[name] = {
                'sampling_rate': fs,
                'scipy_width': width_seconds,
                'app_width': app_width,  # Now a single value
                'expected_width': self.peak_width
            }
        
        # Print and verify results
        print("\nPeak Width Measurement Results:")
        print("-" * 70)
        print(f"{'Resolution':<10} {'Sampling Rate':<15} {'SciPy Width':<15} {'App Width':<15} {'Expected':<10} {'Error %':<10}")
        print("-" * 70)
        
        for name, result in results.items():
            scipy_error = abs(result['scipy_width'] - self.peak_width) / self.peak_width * 100
            app_error = abs(result['app_width'] - self.peak_width) / self.peak_width * 100
            
            # Print results in table format
            print(f"{name:<10} {result['sampling_rate']:<15.1f} {result['scipy_width']:<15.4f} "
                  f"{result['app_width']:<15.4f} {self.peak_width:<10.4f} {app_error:<10.2f}")
            
            # Verify the results are within tolerance
            self.assertLess(scipy_error, 15, f"SciPy width error exceeds 15% for {name} resolution")
            self.assertLess(app_error, 15, f"App width error exceeds 15% for {name} resolution")
        
        # Verify consistency across resolutions
        width_values = [r['app_width'] for r in results.values()]
        max_diff = max(width_values) - min(width_values)
        
        print(f"\nMax width difference across resolutions: {max_diff:.4f} seconds "
              f"({max_diff/self.peak_width*100:.2f}% of expected width)")
        
        # The width measurements should be consistent regardless of resolution
        self.assertLess(max_diff, self.peak_width * 0.15, 
                      "Width measurements differ by more than 15% across resolutions")
        
        # Plot the results
        self.plot_results()
    
    def plot_results(self):
        """Plot the test results for visual inspection."""
        plt.figure(figsize=(15, 10))
        
        # Plot 3 resolutions side by side
        for i, (name, resolution) in enumerate(self.resolutions.items()):
            file_path = self.test_files[name]
            
            # Load data
            data = load_single_file(file_path, time_resolution=resolution)
            time = data['time']
            signal = data['amplitude']
            
            # Calculate sampling rate
            rate = np.median(np.diff(time))
            fs = 1 / rate
            
            # Apply filtering
            filtered_signal, cutoff = adjust_lowpass_cutoff(signal, fs, 100, 1.0)
            
            # Find peak
            peaks, properties = find_peaks(filtered_signal, height=10, prominence=5)
            
            # Calculate peak width using SciPy's peak_widths
            widths, heights, left_ips, right_ips = peak_widths(filtered_signal, peaks, rel_height=0.5)
            
            # Calculate peak width using our application's approach
            app_width = estimate_peak_widths(filtered_signal, fs, 10)
            
            # Plot in subplot
            plt.subplot(3, 1, i+1)
            plt.plot(time, filtered_signal, label=f'Signal ({name}, {fs:.1f} Hz)')
            
            # Plot peak and width
            if len(peaks) > 0:
                peak_idx = peaks[0]
                width = widths[0]
                left_idx = int(left_ips[0])
                right_idx = int(right_ips[0])
                width_height = heights[0]
                
                plt.plot(time[peak_idx], filtered_signal[peak_idx], 'ro', label='Peak')
                plt.hlines(y=width_height, xmin=time[left_idx], xmax=time[right_idx],
                          color='r', linestyle='-', 
                          label=f'Width: {width * rate:.4f}s (App: {app_width:.4f}s)')
            
            plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.3, label='Expected Peak Center')
            
            # Set plot limits to zoom on the peak
            plt.xlim(0.3, 0.7)
            plt.title(f'Resolution: {name} ({resolution:.1e}s, {fs:.1f} Hz)')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('time_resolution_test_results.png')
        print("Time resolution test results saved to time_resolution_test_results.png")
        plt.close()
    
    def tearDown(self):
        """Clean up after test."""
        # Optionally remove test files
        # for file_path in self.test_files.values():
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        pass


if __name__ == "__main__":
    unittest.main() 