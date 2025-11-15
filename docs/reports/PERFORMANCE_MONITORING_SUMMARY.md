# Performance Monitoring Implementation Summary

## What Was Implemented

A comprehensive performance timing system has been added to track and analyze the time spent in different phases of data processing:

### 1. Timing Infrastructure (`core/timing.py`)

- **TimingTracker class**: Tracks timing for different phases of operations
- **Context managers**: Easy-to-use `phase()` context manager for timing code blocks
- **Summary reports**: Automatic generation of detailed timing reports with percentages
- **Metadata tracking**: Store additional context (file counts, data points, etc.)

### 2. Integration Points

#### Data Loading (`service/handlers.py` - `open_files`)
- Tracks file validation, loading, concatenation, and storage
- Breaks down individual file loading times
- Tracks dead-time correction if applied

#### Filtering (`service/handlers.py` - `preprocess_run`)
- Tracks filter design vs. filter application separately
- Distinguishes between Butterworth and Savitzky-Golay filters
- Measures data copying and storage overhead

#### Peak Detection (`service/handlers.py` - `detect_run`)
- Tracks data loading, validation, and detection phases
- Separates threshold calculation from peak detection
- Measures post-processing (stats, properties computation)

### 3. Storage and Retrieval (`service/storage.py`)

- Added `save_timing()` and `get_timing()` methods
- Timing data stored alongside results
- Can retrieve timing for specific operations or all operations

### 4. API Endpoints (`service/handlers.py`)

- `GET /api/performance/timing/{result_id}` - Get timing for specific operation
- `GET /api/performance/timing` - Get all timing reports

## What to Expect

### Typical Timing Breakdown

Based on typical data processing workflows, you should see:

**Data Loading (1M points, single file):**
- File I/O: 50-70% of total time
- Data concatenation: 10-20%
- DataFrame creation: 10-15%
- Other overhead: 5-10%

**Filtering (1M points, Butterworth):**
- Filter design: < 1%
- Filter application (filtfilt): 80-95%
- Data copying: 2-5%
- Storage: 2-5%

**Peak Detection (1M points):**
- Main detection algorithm: 60-80%
- Property computation: 15-25%
- Stats calculation: 5-10%
- Data loading/validation: 5-10%

### Identifying Bottlenecks

The timing system will help you identify:

1. **File I/O bottlenecks**: If `load_file_X` takes > 50% of loading time
2. **Filter computation**: If `butterworth_apply` or `savgol_filter` dominates
3. **Peak detection complexity**: If `analyze_time_resolved` is the main bottleneck
4. **Memory operations**: If `concatenate_data` or `preallocate_arrays` are slow

## High-Level Optimization Recommendations

### If Data Loading is Slow

**Likely causes:**
- Slow disk I/O (HDD vs SSD)
- Large file sizes
- Multiple files loaded sequentially

**Optimization strategies:**
1. **Parallel file loading**: Load multiple files concurrently
2. **File format optimization**: Use binary formats (HDF5, NumPy) instead of text
3. **Streaming**: For very large files, process in chunks
4. **Caching**: Cache frequently accessed files

### If Filtering is Slow

**Likely causes:**
- Large dataset size (O(n) complexity)
- Butterworth filtfilt processes data twice (forward + backward)
- Memory allocation overhead

**Optimization strategies:**
1. **Downsampling**: Filter downsampled data, then interpolate
2. **Filter choice**: Savitzky-Golay is often faster than Butterworth
3. **Chunked processing**: Process data in chunks for very large datasets
4. **Optimized libraries**: Use scipy.signal.lfilter with optimized backends

### If Peak Detection is Slow

**Likely causes:**
- Large dataset size
- Complex peak detection parameters (width constraints, prominence filtering)
- Multiple passes over data

**Optimization strategies:**
1. **Use filtered data**: Pre-filtered data often has fewer false peaks
2. **Parameter tuning**: More restrictive parameters reduce search space
3. **Chunked detection**: Process data in overlapping windows
4. **Early termination**: Stop after finding target number of peaks
5. **Downsampling**: Detect peaks on downsampled data, then refine

### If Plotting/Rendering is Slow

**Note**: The current implementation uses uPlot which is highly optimized. Plotting should be fast.

**If plotting is slow:**
1. **Data decimation**: Ensure automatic decimation is working
2. **Limit visible range**: Only load data for visible time range
3. **Reduce update frequency**: Throttle chart updates during interactions

## Next Steps

1. **Run your typical workflow** and collect timing data
2. **Identify the slowest phase** in each operation
3. **Focus optimization efforts** on the phases taking > 50% of time
4. **Compare different scenarios**:
   - Small vs. large datasets
   - Single vs. multiple files
   - Different filter types
   - Different peak detection parameters

## Example Analysis Workflow

1. Load a typical dataset
2. Check timing: `GET /api/performance/timing/{load_result_id}`
3. Apply a filter
4. Check timing: `GET /api/performance/timing/{filter_result_id}`
5. Detect peaks
6. Check timing: `GET /api/performance/timing/detect_{result_id}`
7. Compare percentages to identify bottlenecks
8. Focus optimization on the slowest phases

## Logging

Timing summaries are automatically logged. Look for messages like:

```
Data loading completed in 2.345s
============================================================
Performance Timing Report: data_loading
============================================================
Total Time: 2.345 seconds

Phase Breakdown:
  load_file_1                       1.234s  (52.6%) [1 calls]
  concatenate_data                  0.456s  (19.4%) [1 calls]
  create_dataframe                  0.234s  (10.0%) [1 calls]
  ...

Slowest Phase: load_file_1 (1.234s, 52.6%)
============================================================
```

This gives you immediate insight into where time is being spent.

