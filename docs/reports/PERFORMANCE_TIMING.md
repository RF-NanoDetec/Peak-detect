# Performance Timing System

## Overview

The performance timing system tracks and reports detailed timing information for the three main operations in the peak analysis workflow:

1. **Data Loading** - Loading files from disk and processing them
2. **Filtering** - Applying preprocessing filters (Butterworth, Savitzky-Golay)
3. **Peak Detection** - Detecting peaks in the signal

## How It Works

The timing system uses a `TimingTracker` class that tracks different phases of each operation. Timing data is automatically collected and stored for each operation, allowing you to identify bottlenecks and optimize performance.

## Timing Breakdown

### Data Loading (`data_loading`)

Tracks:
- `file_validation` - Validating file paths
- `load_data_from_paths` - Overall data loading
  - `load_file_1`, `load_file_2`, ... - Individual file loading
  - `sort_results` - Sorting loaded files
  - `preallocate_arrays` - Memory allocation
  - `concatenate_data` - Combining multiple files
  - `dead_time_correction` - Applying photon correction (if enabled)
  - `create_dataframe` - Creating pandas DataFrame
- `save_result` - Saving to storage

### Filtering (`filtering`)

Tracks:
- `copy_data` - Copying input data
- `butterworth_filter` (if Butterworth selected)
  - `butterworth_design` - Designing filter coefficients
  - `butterworth_apply` - Applying filter (filtfilt)
- `savgol_filter` (if Savitzky-Golay selected)
- `save_result` - Saving filtered data

### Peak Detection (`peak_detection`)

Tracks:
- `load_data` - Loading base and filtered data
- `validate_params` - Validating peak detection parameters
- `auto_threshold_calc` - Calculating auto threshold (if using filtered data)
- `analyze_time_resolved` - Main peak detection algorithm
- `calculate_stats` - Computing peak statistics
- `compute_properties` - Computing peak properties (widths, prominences, etc.)

## Accessing Timing Reports

### Via API Endpoints

1. **Get timing for a specific result:**
   ```
   GET /api/performance/timing/{result_id}
   ```
   Returns detailed timing breakdown for that specific operation.

2. **Get all timing reports:**
   ```
   GET /api/performance/timing
   ```
   Returns all stored timing data.

### Via Logs

Timing summaries are automatically logged to the application log when operations complete. Look for messages like:

```
Data loading completed in 2.345s
Filtering completed in 1.234s
Peak detection completed in 3.456s
```

Followed by a detailed breakdown showing each phase.

## Timing Report Format

Each timing report includes:

```json
{
  "operation": "data_loading",
  "total_time_seconds": 2.345,
  "phases": {
    "load_file_1": {
      "total_seconds": 1.234,
      "percentage": 52.6,
      "call_count": 1
    },
    "concatenate_data": {
      "total_seconds": 0.456,
      "percentage": 19.4,
      "call_count": 1
    }
  },
  "slowest_phase": {
    "name": "load_file_1",
    "time_seconds": 1.234,
    "percentage": 52.6
  },
  "metadata": {
    "file_count": 1,
    "total_points": 1000000,
    "time_resolution": 0.0001
  }
}
```

## Identifying Bottlenecks

The timing system helps identify:

1. **Slow file I/O** - If `load_file_X` phases take a long time
2. **Filter computation overhead** - Compare `butterworth_design` vs `butterworth_apply`
3. **Peak detection complexity** - See how `analyze_time_resolved` compares to other phases
4. **Data processing overhead** - Check `concatenate_data` and `create_dataframe` times

## Performance Optimization Recommendations

Based on timing reports:

### If file loading is slow:
- Consider parallel file loading for multiple files
- Use faster storage (SSD vs HDD)
- Optimize file format (binary vs text)

### If filtering is slow:
- Butterworth `filtfilt` is O(n) but processes data twice (forward + backward)
- Savitzky-Golay is typically faster for similar results
- Consider downsampling before filtering if high resolution isn't needed

### If peak detection is slow:
- `analyze_time_resolved` is the main bottleneck
- Consider using filtered data (often faster than raw)
- Adjust detection parameters (distance, width constraints) to reduce search space
- For very large datasets, consider chunked processing

### If plotting/rendering is slow:
- The frontend uses uPlot which is highly optimized
- Data is automatically decimated for display
- Full-resolution data is only loaded when explicitly requested

## Example Usage

After running a data loading operation, you can check the timing:

```python
# In your application code or via API
import requests

# Get timing for a specific result
response = requests.get(f"http://localhost:8000/api/performance/timing/{result_id}")
timing_data = response.json()

print(f"Total time: {timing_data['total_time_seconds']:.3f}s")
print(f"Slowest phase: {timing_data['slowest_phase']['name']}")
print(f"  Time: {timing_data['slowest_phase']['time_seconds']:.3f}s")
print(f"  Percentage: {timing_data['slowest_phase']['percentage']:.1f}%")
```

## Notes

- Timing data is stored in memory and cleared when the server restarts
- Timing overhead is minimal (< 1ms per phase)
- All times are in seconds with millisecond precision
- Percentages are calculated relative to total operation time

