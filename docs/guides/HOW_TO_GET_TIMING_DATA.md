# How to Get Performance Timing Data

## Overview

Performance timing data is stored **in memory** (not in files) and can be accessed in several ways:

1. **Via API endpoints** (programmatic access)
2. **Via utility script** (easy command-line access)
3. **Via logs** (automatic logging when operations complete)

## Method 1: Using the Utility Script (Easiest)

### Step 1: Make sure your server is running

```bash
python -m service.app
```

The server should be running at `http://127.0.0.1:8765`

### Step 2: Run some operations

1. Load some data files
2. Apply a filter (optional)
3. Detect peaks

### Step 3: Get timing data

**View all timing data:**
```bash
python tools/get_timing_data.py
```

**View timing for a specific result:**
```bash
python tools/get_timing_data.py <result_id>
```

The script will:
- Display a formatted report showing all phases and their timings
- Show percentages for each phase
- Identify the slowest phase
- Optionally save data to a JSON file

### Example Output

```
Performance Timing Data Viewer
======================================================================
Fetching all timing data...

======================================================================
Timing Report for: abc123-def456-...
======================================================================
Total Time: 2.345 seconds (2345.0 ms)

Phase Breakdown:
Phase Name                               Time (s)      %        Calls
----------------------------------------------------------------------
load_file_1                                 1.234s   52.6%         1
concatenate_data                            0.456s   19.4%         1
create_dataframe                            0.234s   10.0%         1
file_validation                             0.123s    5.2%         1
save_result                                  0.098s    4.2%         1

Slowest Phase: load_file_1
  Time: 1.234s
  Percentage: 52.6%

Metadata:
  file_count: 1
  total_points: 1000000
  time_resolution: 0.0001
======================================================================
```

## Method 2: Using API Endpoints Directly

### Get timing for a specific result

```bash
curl http://127.0.0.1:8765/api/performance/timing/<result_id>
```

Or in Python:
```python
import requests

result_id = "your-result-id-here"
response = requests.get(f"http://127.0.0.1:8765/api/performance/timing/{result_id}")
timing_data = response.json()
print(timing_data)
```

### Get all timing data

```bash
curl http://127.0.0.1:8765/api/performance/timing
```

Or in Python:
```python
import requests

response = requests.get("http://127.0.0.1:8765/api/performance/timing")
all_timings = response.json()
print(all_timings)
```

## Method 3: Check the Logs

Timing summaries are automatically logged when operations complete. Look in your terminal where the server is running, or check log files if logging is configured.

You'll see messages like:
```
INFO: Data loading completed in 2.345s
============================================================
Performance Timing Report: data_loading
============================================================
Total Time: 2.345 seconds

Phase Breakdown:
  load_file_1                       1.234s  (52.6%) [1 calls]
  concatenate_data                  0.456s  (19.4%) [1 calls]
  ...
```

## Finding Result IDs

### From the API Response

When you load data, apply a filter, or detect peaks, the API returns a `resultId`. Save this ID to query timing later.

**Example - Loading data:**
```python
import requests

# Load files
response = requests.post(
    "http://127.0.0.1:8765/api/files/open",
    json={
        "paths": ["/path/to/file.txt"],
        "time_resolution": 1e-4
    }
)
result = response.json()
result_id = result["resultId"]  # Save this!

# Later, get timing
timing = requests.get(f"http://127.0.0.1:8765/api/performance/timing/{result_id}")
```

### From the Web UI

If you're using the web UI, result IDs are typically stored in the browser's state. You can also check the browser's developer console (Network tab) to see the API responses.

## Timing Data Format

The timing data is returned as JSON with this structure:

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

## Saving Timing Data to Files

### Using the utility script

The script will prompt you to save data:
```bash
python tools/get_timing_data.py
# ... shows report ...
Save to file? (y/n): y
✅ Saved timing data to: timing_<result_id>.json
```

### Manually

```python
import requests
import json

# Get timing data
response = requests.get("http://127.0.0.1:8765/api/performance/timing/<result_id>")
timing_data = response.json()

# Save to file
with open("timing_report.json", "w") as f:
    json.dump(timing_data, f, indent=2)
```

## Important Notes

1. **Timing data is in memory** - It's cleared when the server restarts
2. **Timing is automatic** - No configuration needed, it tracks all operations
3. **Result IDs are unique** - Each operation (load, filter, detect) gets its own ID
4. **Peak detection timing** - Uses ID format: `detect_{result_id}`

## Troubleshooting

### "Could not connect to server"
- Make sure the server is running: `python -m service.app`
- Check the server is on port 8765

### "No timing data found"
- Run some operations first (load data, filter, detect peaks)
- Check that the result ID is correct
- Timing data is cleared when server restarts

### "No timing data available"
- This means no operations have been run yet
- Load some data, apply a filter, or detect peaks first

## Next Steps

Once you have timing data:

1. **Identify bottlenecks** - Look for phases taking >50% of time
2. **Compare operations** - Run the same operation with different parameters
3. **Optimize** - Focus on the slowest phases (see `docs/PERFORMANCE_MONITORING_SUMMARY.md`)

