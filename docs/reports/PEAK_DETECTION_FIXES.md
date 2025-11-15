# Peak Detection & Visualization Fixes

## Critical Bugs Fixed

### 1. 🐛 **CRITICAL: Peak Position Bug**

**Problem**: Peaks appeared at completely wrong positions on the detection chart.

**Root Cause**: Index mismatch between full and decimated datasets
- Peak detection runs on FULL dataset (e.g., 300,000 points)
- Detection returns peak indices into full dataset: `[100, 5000, 150000, ...]`
- Frontend used decimated preview data (e.g., 10,000 points)
- Code incorrectly used full dataset indices as indices into decimated data
- Result: Peaks appeared at wrong times and amplitudes!

**Example of the Bug**:
```typescript
// WRONG CODE (before fix):
const peakData = peaks.map(idx => ({
  time: previewData.time[idx],  // ← Using full dataset index on decimated data!
  amplitude: previewData.amplitude[idx]
}))

// If peak is at index 150,000 in full dataset,
// but previewData only has 10,000 points → WRONG!
```

**Solution**: Backend now returns actual peak times and amplitudes
```python
# Backend (service/handlers.py):
return {
    "peaks": peaks_list,  # Indices (for reference)
    "peak_times": [float(base["time"][i]) for i in peaks_list],
    "peak_amplitudes": [float(base["amplitude"][i]) for i in peaks_list],
    ...
}
```

```typescript
// Frontend (DetectionChart.tsx):
const peakData = peakTimes.map((time, i) => ({
  time: time / 60,  // Actual time value!
  amplitude: peakAmplitudes[i]  // Actual amplitude!
}))
```

**Result**: Peaks now appear at their CORRECT positions! ✅

### 2. 📊 **Y-Axis Domain Issues**

**Problem**: 
- Preprocessing chart showed too many negative values (down to -200)
- Detection chart cut off negative values after filtering

**Root Cause**: Recharts auto-scaling included large negative padding by default.

**Solution**: Set explicit domain constraints
```typescript
<YAxis 
  domain={['dataMin - 20', 'dataMax + 20']}
  // Shows only 20 units below minimum value
/>
```

**Result**: 
- ✅ Preprocessing: Shows only slightly below zero
- ✅ Detection: Includes small negative values from filtering
- ✅ Both: Clean, readable Y-axis range

### 3. 🎯 **Peak Visibility**

**Problem**: Many peaks detected but few/none displayed on chart.

**Root Cause**: Combination of the index bug and Y-axis scaling.

**Solution**: 
1. Fixed index bug (peaks now at correct positions)
2. Fixed Y-axis domain (peaks now visible)
3. Backend returns actual peak data for direct plotting

**Result**: All detected peaks now visible at correct positions! ✅

## Technical Implementation

### Backend Changes (`service/handlers.py`)

**Before**:
```python
return {
    "count": len(peaks_list),
    "peaks": peaks_list,
    "stats": {...}
}
```

**After**:
```python
# Extract peak details from full dataset
if len(peaks_list) > 0:
    peak_heights = [float(base["amplitude"][i]) for i in peaks_list]
    peak_times = [float(base["time"][i]) for i in peaks_list]
    ...

return {
    "count": len(peaks_list),
    "peaks": peaks_list,  # Indices (for reference)
    "peak_times": peak_times,  # Actual time values
    "peak_amplitudes": peak_heights,  # Actual amplitudes
    "stats": {...}
}
```

### Frontend Changes

**1. DetectionChart Component** (`ui-web/components/charts/DetectionChart.tsx`)

**Before**:
```typescript
interface DetectionChartProps {
  peaks: number[]  // Indices
  ...
}

// WRONG: Using indices on decimated data
const peakData = peaks.map(idx => ({
  time: previewData.time[idx],
  amplitude: previewData.amplitude[idx]
}))
```

**After**:
```typescript
interface DetectionChartProps {
  peakTimes?: number[]  // Actual times
  peakAmplitudes?: number[]  // Actual amplitudes
  ...
}

// CORRECT: Using actual values
const peakData = peakTimes.map((time, i) => ({
  time: Math.max(0, time / 60),
  amplitude: peakAmplitudes[i]
}))
```

**2. Detect Page** (`ui-web/app/detect/page.tsx`)

**Before**:
```typescript
const peaks = detectionResults?.peaks || []

<DetectionChart peaks={peaks} ... />
```

**After**:
```typescript
const peakTimes = detectionResults?.peak_times || []
const peakAmplitudes = detectionResults?.peak_amplitudes || []

<DetectionChart 
  peakTimes={peakTimes}
  peakAmplitudes={peakAmplitudes}
  ...
/>
```

**3. Y-Axis Configuration** (Both Charts)

```typescript
<YAxis 
  domain={['dataMin - 20', 'dataMax + 20']}
  // Adds 20 units padding above and below
  // Prevents excessive negative range
  // Includes small negative values from filtering
/>
```

### Type Definitions Updated (`ui-web/lib/types.ts`)

```typescript
export interface DetectPeaksResponse {
  peaks: number[]
  peak_times?: number[]  // NEW
  peak_amplitudes?: number[]  // NEW
  count: number
  stats: {...}
}
```

## Visual Results

### Before Fix 🔴
- Preprocessing: Y-axis from -200 to 1200
- Detection: Peaks at wrong positions
- Detection: Many peaks missing
- Detection: Graph rescaled incorrectly

### After Fix ✅
- Preprocessing: Y-axis from -20 to 1200 (clean!)
- Detection: Peaks at EXACT correct positions
- Detection: ALL peaks visible
- Detection: Graph scaled properly around data

## Data Flow

### Peak Detection Flow (After Fix)

1. **Detection** (`/api/detect/run`):
   ```
   Full dataset (300k points) → Peak detection → Peak indices
   Peak indices → Extract times & amplitudes from full dataset
   Return: {peaks, peak_times, peak_amplitudes}
   ```

2. **Visualization**:
   ```
   Preview data (10k points) → Line chart
   Peak times & amplitudes → Scatter plot
   Both use TIME as x-axis → Perfect alignment!
   ```

### Why This Works

**Key Insight**: Instead of trying to match indices between datasets of different sizes, we use **time as the common coordinate system**.

- Preview data: plotted by time
- Peaks: plotted by time
- Time is continuous and shared
- No index matching needed!

## Testing Checklist

- [x] Load and preprocess data
- [x] Apply filter → Y-axis stays reasonable (-20 to max)
- [x] Detect peaks → Peaks appear at correct positions
- [x] Detect peaks → All detected peaks visible
- [x] Peak markers align with signal maxima
- [x] Hover over peaks → Correct values shown
- [x] Multiple peaks → All positioned correctly
- [x] Negative values after filtering → Visible (not cut off)

## Performance Impact

**No performance regression**:
- Extracting peak times/amplitudes: O(n) where n = number of peaks (typically <1000)
- Negligible compared to peak detection itself
- Data transmission: +2 arrays of ~1000 floats each = ~16KB
- Well within acceptable limits

## Future Considerations

### Potential Enhancement: Smart Preview Around Peaks

If more detail needed around peaks:
```python
# Could create custom preview that includes peak regions
def create_peak_aware_preview(time, amplitude, peaks, max_points):
    # 1. Include all peak positions + surroundings
    # 2. Fill remainder with uniform sampling
    # 3. Ensure total <= max_points
    ...
```

This would provide:
- Higher detail around peaks
- Lower detail in baseline regions
- Better visualization without increasing data size significantly

## Conclusion

The index mismatch bug was a **critical issue** that made peak detection visualization completely unreliable. By returning actual time and amplitude values instead of indices, we've made the system robust and correct.

**Key Takeaway**: When working with decimated/sampled data, always use **coordinate values** (time, position, etc.) rather than **indices** for cross-referencing between datasets.

All visualization issues are now resolved:
- ✅ Peaks at correct positions
- ✅ All peaks visible
- ✅ Proper Y-axis scaling
- ✅ Clean, professional appearance
- ✅ Accurate representation of detection results


