# Full Data Display Debugging Guide

## Changes Made

### Backend (`service/handlers.py`)
- Added `fullData` boolean parameter to `/api/data/preview` endpoint
- When `fullData=True`, returns ALL data points without decimation
- Added logging to track what's being sent
- Returns metadata: `decimated`, `count`, `original_count`

### Frontend (`ui-web/app/preprocess/page.tsx`)
- Added checkbox: "Show all data points (no decimation)"
- When checked, fetches data with `fullData=true`
- Increased timeout to 2 minutes for large datasets
- Added console logging and toast notifications
- Shows info about decimation status and point counts

### API Client (`ui-web/lib/apiClient.ts`)
- Updated timeout: 30s for decimated, 120s for full data
- Added console logging for debugging

## How to Debug

### Step 1: Restart Backend
```bash
python -m service.app
```

### Step 2: Open Browser DevTools
- Press F12
- Go to Console tab
- Go to Network tab

### Step 3: Load Data and Check the Box
1. Load your data files
2. Go to Preprocess page
3. **Check the box**: "Show all data points (no decimation)"

### Step 4: Watch the Logs

**Browser Console should show:**
```
Reloading data with fullData=true
Fetching data preview: fullData=true, maxPoints=10000
Received 1000000 data points, decimated=false
Successfully loaded 1000000 points
```

**Backend Terminal should show:**
```
INFO: get_data_preview: fullData=True, total_points=1000000, maxPoints=10000
INFO: Returning ALL 1000000 data points (no decimation)
INFO: 127.0.0.1:XXXXX - "GET /api/data/preview?resultId=...&fullData=true" 200 OK
```

**UI Display should show:**
```
✓ Showing all 1,000,000 points
Decimated: No
```

## Potential Issues

### Issue 1: Response Size Too Large
**Symptom**: Request times out or fails
**Cause**: Million+ points = 50-100MB JSON
**Solution**: 
- Check Network tab for the request size
- May need to increase FastAPI response limits
- Consider using binary format or compression

### Issue 2: Browser Memory
**Symptom**: Browser becomes slow or crashes
**Cause**: Plotting 1M+ points uses lots of RAM
**Solution**:
- Plotly has built-in optimizations but may still struggle
- Consider setting a reasonable max (e.g., 100k points)
- Use WebGL mode in Plotly for better performance

### Issue 3: Parameter Not Passed
**Symptom**: Still shows decimated data even when checked
**Browser Console shows**: `fullData=false` 
**Solution**: 
- Check if checkbox state is updating
- Verify useEffect is triggering
- Check Network tab to see actual request parameters

### Issue 4: Data Transfer Time
**Symptom**: Long wait time, then data appears
**Cause**: Transferring large JSON takes time
**Solution**:
- This is expected for large datasets
- Toast notification shows "Loading all data points..."
- Wait for success message

## Expected Behavior

### With 1 Million Data Points

**Decimated (unchecked):**
- Request: ~1-2 seconds
- Response size: ~200-500 KB
- Points displayed: ~10,000
- Performance: Smooth and fast

**Full Data (checked):**
- Request: ~5-15 seconds
- Response size: ~50-100 MB
- Points displayed: 1,000,000
- Performance: May be slow, depends on browser

## Troubleshooting Commands

### Check Data Size in Backend
```python
# Add to handlers.py temporarily
import sys
total_size = sys.getsizeof(result["time"].tolist())
logger.info(f"Response size: {total_size / 1024 / 1024:.2f} MB")
```

### Force Full Data via API Test
```bash
# Test endpoint directly
curl "http://127.0.0.1:8765/api/data/preview?resultId=YOUR_ID&fullData=true"
```

### Check Browser Network Request
1. F12 → Network tab
2. Find request to `/api/data/preview`
3. Check Query String Parameters: `fullData: true`
4. Check Response size
5. Check Response time

## FastAPI Configuration

If response is too large, you may need to configure uvicorn:

```python
# In service/app.py
uvicorn.run(
    "service.app:app",
    host="127.0.0.1",
    port=8765,
    reload=False,
    log_level="info",
    limit_concurrency=1000,
    limit_max_requests=10000,
    timeout_keep_alive=120,  # Increase timeout
)
```

## Plotly Performance Tips

If rendering is slow with full data, you can optimize Plotly:

```typescript
// In preprocess/page.tsx
layout={{
  // ... existing layout
  hovermode: false,  // Disable hover for better performance
}}

// Or use WebGL mode
{
  type: 'scattergl',  // Instead of 'scatter'
  mode: 'lines',
  // ...
}
```

## Next Steps Based on Results

### If Full Data Shows Correctly
→ Problem was decimation algorithm
→ Need to improve decimation to preserve more features

### If Full Data Still Shows ~10k Points
→ Check backend logs for actual count sent
→ Check network response size
→ Issue is in data transfer or frontend parsing

### If Full Data Loads but Browser Crashes
→ Dataset is too large for browser rendering
→ Need better decimation, not full data display
→ Consider setting max at 50k-100k points

### If Request Times Out
→ Dataset is too large to transfer
→ Need server-side rendering or different approach
→ Consider pagination or chunking

## Success Criteria

✅ Backend log shows: "Returning ALL X data points (no decimation)"
✅ Browser console shows: "Received X data points, decimated=false"
✅ UI shows: "Showing all X points" and "Decimated: No"
✅ Chart renders (even if slow)
✅ All peaks and valleys are visible

If all criteria are met, the full data display is working correctly!



