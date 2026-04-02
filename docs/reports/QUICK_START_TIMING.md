# Quick Start: Getting Performance Timing Data

## Step 1: Start Your Server

Make sure your server is running:

```bash
python -m service.app
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8765
```

## Step 2: Run Some Operations

Use your web UI or API to:
1. **Load data** - Load one or more data files
2. **Apply filter** (optional) - Apply a Butterworth or Savitzky-Golay filter
3. **Detect peaks** - Run peak detection

## Step 3: Get the Timing Data

### Option A: Use the Utility Script (Easiest)

```bash
# View all timing data
python tools/get_timing_data.py

# View timing for a specific result ID
python tools/get_timing_data.py <result_id>
```

### Option B: Test the System

Run the test script to verify everything works:

```bash
python tools/test_timing.py
```

This will:
- Create a test file
- Load it
- Show you the timing data
- Clean up

### Option C: Use the API Directly

```bash
# Get all timing data
curl http://127.0.0.1:8765/api/performance/timing

# Get timing for specific result
curl http://127.0.0.1:8765/api/performance/timing/<result_id>
```

## Step 4: Share the Data with Me

Once you have timing data, you can share it with me in several ways:

### Method 1: Copy the Output

Run the utility script and copy the output:

```bash
python tools/get_timing_data.py > timing_output.txt
```

Then share the contents of `timing_output.txt`.

### Method 2: Save as JSON

The script can save data as JSON. When prompted, type 'y':

```bash
python tools/get_timing_data.py
# ... shows report ...
Save to file? (y/n): y
```

Then share the JSON file.

### Method 3: Copy from API Response

If using the API directly, copy the JSON response and share it.

## What You'll See

The timing report shows:

```
======================================================================
Timing Report: data_loading
======================================================================
Total Time: 2.345 seconds

Phase Breakdown:
Phase Name                               Time (s)      %        Calls
----------------------------------------------------------------------
load_file_1                                 1.234s   52.6%         1
concatenate_data                            0.456s   19.4%         1
create_dataframe                            0.234s   10.0%         1
...

Slowest Phase: load_file_1
  Time: 1.234s
  Percentage: 52.6%

Metadata:
  file_count: 1
  total_points: 1000000
  time_resolution: 0.0001
======================================================================
```

## Finding Result IDs

Result IDs are returned when you:
- Load data → `resultId` in the response
- Apply filter → `resultId` in the task result
- Detect peaks → Use the base `resultId` (timing stored as `detect_{resultId}`)

You can also get all timing data without knowing IDs:

```bash
python tools/get_timing_data.py
```

This shows all available timing reports.

## Troubleshooting

**"Could not connect to server"**
- Make sure the server is running: `python -m service.app`

**"No timing data available"**
- Run some operations first (load data, filter, detect peaks)
- Timing data is stored in memory and cleared when server restarts

**"No timing data found for result ID"**
- Check that the result ID is correct
- Make sure the operation completed successfully
- For peak detection, use format: `detect_{resultId}`

## Next Steps

Once you have timing data, I can help you:
1. Identify bottlenecks (what's taking the most time)
2. Suggest optimizations
3. Compare different scenarios
4. Understand the performance characteristics

Just share the timing output or JSON data with me!

