#!/usr/bin/env python
"""
Runner script for peak width test.

This script runs the peak width test and displays the results.
"""

import os
import sys
import unittest
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the test
from tests.test_peak_width import PeakWidthTest

if __name__ == "__main__":
    print("=" * 80)
    print("Running Peak Width Validation Test")
    print("=" * 80)
    print("This test will:")
    print("1. Generate synthetic data with 100 peaks of exactly 50ms width")
    print("2. Save it to a test file with 0.1ms time resolution")
    print("3. Process the data using the application's peak detection")
    print("4. Verify that measured peak widths match expected values")
    print("=" * 80)
    
    # Run the test
    test_suite = unittest.TestLoader().loadTestsFromTestCase(PeakWidthTest)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # If test passed, show the results
    if test_result.wasSuccessful():
        print("\nTest PASSED! Peak width calculations are accurate.")
        
        # Display the saved plot if available
        if os.path.exists('peak_width_test_results.png'):
            print("Displaying results plot...")
            from PIL import Image
            img = Image.open('peak_width_test_results.png')
            img.show()
    else:
        print("\nTest FAILED! Please check the error messages above.")
    
    print("\nTest data file: test_peak_data.txt")
    print("Results plot: peak_width_test_results.png") 