# Preprocessing Visualization Improvements

## Overview
Implemented comprehensive improvements to the preprocessing page to provide better visualization and comparison of original vs filtered data.

## Completed Changes

### 1. Overlay Visualization ✅
**Original + Filtered Data Display**
- Original data is always visible with reduced opacity (40%) when filtered data exists
- Filtered data is overlaid on top with full opacity
- Both traces are clearly labeled in the legend
- Title changes from "Signal Preview" to "Signal Comparison" when filtered data is present

**Colors:**
- Original Data: Gray (`rgb(100, 116, 139)`) with 0.4 opacity when filtered
- Filtered Data: Blue (`rgb(59, 130, 246)`) with full opacity

### 2. Smart Time Axis Formatting ✅
**Automatic Unit Switching**
- If max time < 60 seconds: Display in seconds (s)
- If max time >= 60 seconds: Automatically convert to minutes (min)
- X-axis label updates dynamically: "Time (s)" or "Time (min)"

**Implementation:**
```typescript
const formatTimeAxis = (timeData: number[]) => {
  const maxTime = Math.max(...timeData)
  if (maxTime < 60) {
    return { title: 'Time (s)', values: timeData }
  } else {
    return { title: 'Time (min)', values: timeData.map(t => t / 60) }
  }
}
```

### 3. Proper Axis Labels ✅
**Y-Axis:**
- Changed from "Amplitude" to "Counts"
- More descriptive and domain-appropriate

**X-Axis:**
- Smart formatting as described above
- Clearer time representation

### 4. Improved Data Decimation ✅
**Enhanced Algorithm:**
- Increased default max_points from 5000 to 10000
- Minimum 1000 points preserved (was 100)
- Now preserves both peaks AND valleys (previously only peaks)
- Uses smaller stride to maintain more detail
- Two-stage decimation for better feature preservation

**Key Improvements:**
- `stride // 2` for base sampling (keeps more points)
- Includes local minima (valleys) in addition to maxima
- Final check to ensure result doesn't exceed max_points
- Better balance between performance and visual quality

**Before:**
```python
# Only preserved peaks with aggressive stride
mask[::stride] = True
peak_mask = (y > y_left) & (y > y_right)
```

**After:**
```python
# Preserves more points with smaller stride
mask[::max(1, stride // 2)] = True
peak_mask = (y > y_left) & (y > y_right)
valley_mask = (y < y_left) & (y < y_right)  # NEW
mask = mask | peak_mask | valley_mask
```

### 5. Data Store Enhancement ✅
**New State Management:**
- Added `originalPreviewData` to store unfiltered data
- Added `filteredPreviewData` to store filtered results
- Added `setOriginalPreviewData()` and `setFilteredPreviewData()` methods
- Preserved both datasets for comparison

**Data Flow:**
```
Load Files → setOriginalPreviewData → 
Apply Filter → setFilteredPreviewData →
Display Both Overlaid
```

## Files Modified

### Frontend
1. **ui-web/lib/types.ts**
   - Added `originalPreviewData: DataPreviewResponse | null`
   - Added `filteredPreviewData: DataPreviewResponse | null`
   - Added setter methods for both

2. **ui-web/lib/stores/dataStore.ts**
   - Implemented new state fields
   - Added setter methods
   - Maintained data persistence

3. **ui-web/app/load/page.tsx**
   - Sets `originalPreviewData` when files load
   - Clears `filteredPreviewData` on new load

4. **ui-web/app/preprocess/page.tsx**
   - Complete chart visualization rewrite
   - Dual-trace display implementation
   - Smart time formatting
   - Legend configuration
   - Opacity management

### Backend
1. **service/handlers.py**
   - Changed preview endpoint parameter from `limit` to `maxPoints`
   - Increased default from 5000 to 10000
   - Minimum points increased to 1000

2. **core/data_utils.py**
   - Enhanced `decimate_for_plot` algorithm
   - Added valley preservation
   - Improved stride calculation
   - Two-stage decimation process

## Technical Details

### Chart Configuration
```typescript
layout={{
  title: filteredPreviewData ? 'Signal Comparison' : 'Signal Preview',
  xaxis: { title: timeAxisTitle },  // Dynamic: "Time (s)" or "Time (min)"
  yaxis: { title: 'Counts' },
  showlegend: true,
  legend: {
    x: 1,
    y: 1,
    xanchor: 'right',
    yanchor: 'top',
  },
}}
```

### Trace Configuration
```typescript
// Original trace
{
  name: 'Original Data',
  line: { color: 'rgb(100, 116, 139)', width: 1.5 },
  opacity: filteredPreviewData ? 0.4 : 1.0,
}

// Filtered trace
{
  name: 'Filtered Data',
  line: { color: 'rgb(59, 130, 246)', width: 2 },
  opacity: 1.0,
}
```

## User Experience Improvements

### Before
- Only one trace visible at a time
- No comparison between original and filtered
- Generic axis labels ("Time (s)", "Amplitude")
- Aggressive decimation losing signal detail
- No indication of what data is being shown

### After
- Both original and filtered data visible simultaneously
- Clear visual comparison with opacity difference
- Smart time formatting (seconds/minutes)
- Domain-appropriate labels ("Counts")
- Better data preservation with improved decimation
- Clear legend identifying each trace

## Testing

### Test Scenarios

1. **Load and View Original**
   - Load data files
   - Navigate to Preprocess
   - Verify original data shows with full opacity
   - Verify y-axis shows "Counts"
   - Verify x-axis shows appropriate time unit

2. **Apply Butterworth Filter**
   - Select Butterworth filter
   - Set parameters (cutoff: 1000 Hz, order: 4)
   - Click "Apply Filter"
   - Verify original data becomes semi-transparent
   - Verify filtered data overlays in blue
   - Verify legend shows both traces

3. **Apply Savitzky-Golay Filter**
   - Select Savitzky-Golay filter
   - Set parameters (window: 51, polyorder: 3)
   - Click "Apply Filter"
   - Verify overlay works correctly
   - Verify smoothing is visible

4. **Time Formatting**
   - **Short signals (<60s)**: Verify "Time (s)" label
   - **Long signals (>=60s)**: Verify "Time (min)" label and values are divided by 60

5. **Data Decimation**
   - Load large dataset (>100k points)
   - Verify plot renders smoothly
   - Verify signal features (peaks, valleys) are preserved
   - Verify no drastic loss of visual information

## Known Limitations

1. **Memory Usage**: Storing both original and filtered data doubles memory usage for preview
2. **Time Conversion**: Only handles seconds and minutes (not hours)
3. **Color Scheme**: Fixed colors (not themeable yet)

## Future Enhancements

1. **Interactive Comparison**: Add toggle to show/hide each trace
2. **Zoom Synchronization**: Ensure both traces zoom together
3. **Statistical Overlay**: Show difference between original and filtered
4. **Export Both**: Option to export comparison visualization
5. **Theme Support**: Make colors respect dark/light theme
6. **Hours Support**: Add automatic conversion for signals > 60 minutes

## Performance Notes

- Decimation improvements maintain good performance even with larger max_points
- Dual-trace rendering has minimal performance impact
- Smart time formatting is computed once per render (cached)
- Legend rendering is optimized by Plotly

## Compatibility

- Works with all existing filter types (none, butterworth, savgol)
- Backward compatible with existing data
- No database/storage changes required
- Frontend-only state management

## Conclusion

The preprocessing page now provides a professional, informative visualization that allows users to:
- Clearly see the effect of filtering
- Compare original vs filtered data side-by-side
- Understand time scales intuitively
- View data with appropriate domain labels
- See better preserved signal details

All improvements maintain performance while significantly enhancing user experience and data interpretation capabilities.



