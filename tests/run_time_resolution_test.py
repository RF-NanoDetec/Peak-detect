#!/usr/bin/env python
"""
Runner script for time resolution test.

This script runs the time resolution test to verify that the application
correctly handles different time resolutions throughout the processing pipeline.
"""

import os
import sys
import unittest
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the test
from tests.test_time_resolution import TimeResolutionTest

if __name__ == "__main__":
    print("=" * 80)
    print("Running Time Resolution Handling Test")
    print("=" * 80)
    print("This test will:")
    print("1. Generate test data with the same peak at 3 different time resolutions")
    print("2. Verify that loading with correct resolution gives consistent time values")
    print("3. Check that peak width calculation is consistent across resolutions")
    print("4. Validate that the time resolution changes are applied throughout the pipeline")
    print("=" * 80)
    
    # Run the test
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TimeResolutionTest)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # If test passed, show the results
    if test_result.wasSuccessful():
        print("\nTest PASSED! Time resolution handling is correct throughout the application.")
        
        # Display the saved plot if available
        if os.path.exists('time_resolution_test_results.png'):
            print("Displaying results plot...")
            from PIL import Image
            img = Image.open('time_resolution_test_results.png')
            img.show()
    else:
        print("\nTest FAILED! Please check the error messages above.")
    
    print("\nTest data files are saved with names like: test_time_res_1.0e-04.txt")
    print("Results plot: time_resolution_test_results.png") 