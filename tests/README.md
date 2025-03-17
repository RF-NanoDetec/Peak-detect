# Peak Analysis Tool Tests

This directory contains test scripts for validating the Peak Analysis Tool functionality. The tests are designed to verify that critical components of the application work as expected.

## Available Tests

### Peak Width Test

This test verifies that the peak width calculation is accurate by:
1. Generating synthetic data with peaks of known width (50ms)
2. Saving it to a file with 0.1ms time resolution 
3. Processing the data using the application's peak detection
4. Verifying that measured peak widths match expected values

To run the test:
```bash
python run_peak_width_test.py
```

### Time Resolution Test

This test verifies that time resolution is handled correctly throughout the application by:
1. Generating test data with the same peak at 3 different time resolutions
2. Verifying that loading with correct resolution gives consistent time values
3. Checking that peak width calculation is consistent across resolutions
4. Validating that time resolution changes are applied throughout the pipeline

To run the test:
```bash
python run_time_resolution_test.py
```

## Test Output

The tests generate:
1. Test data files (e.g., `test_peak_data.txt` or `test_time_res_1.0e-04.txt`)
2. Result plots (e.g., `peak_width_test_results.png` or `time_resolution_test_results.png`)

These outputs are helpful for visual inspection of the test results.

## Expected Results

The tests should indicate success with:
- No assertion errors
- Detailed output showing that measured values match expected values within tolerance
- Visual confirmation that peak widths are correctly identified in plots

If a test fails, it will print detailed error information explaining why.

## Adding New Tests

To add a new test:
1. Create a test class extending `unittest.TestCase`
2. Add test methods starting with `test_`
3. Create a runner script for easy execution 