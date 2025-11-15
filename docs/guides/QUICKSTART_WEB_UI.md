# Quick Start Guide - Web UI

## Overview
This guide will help you quickly start the web UI and test the new features:
1. File picker for multiple file selection
2. Working preprocessing with real-time progress tracking

## Prerequisites

- Python 3.8+ installed
- Node.js 16+ and npm installed
- All dependencies installed

## Step 1: Install Backend Dependencies

```bash
# From project root
pip install -r requirements.txt
```

Key dependencies for the new features:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `scipy` - For signal processing filters
- `numpy` - For numerical operations

## Step 2: Install Frontend Dependencies

```bash
cd ui-web
npm install
cd ..
```

## Step 3: Start the Backend Server

```bash
# From project root
python -m service.app
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765
```

Keep this terminal open!

## Step 4: Test Backend (Optional)

In a new terminal:
```bash
python tools/test_web_ui.py
```

Expected output:
```
Web UI Backend Test Suite
============================================================
...
✅ Health check passed
✅ Version check passed
✅ Params check passed
✅ File upload endpoint exists
...
🎉 All tests passed!
```

## Step 5: Start the Frontend Development Server

In a new terminal:
```bash
cd ui-web
npm run dev
```

You should see:
```
- ready started server on 0.0.0.0:3000
- info Loaded env from ...
- event compiled client and server successfully
```

## Step 6: Open in Browser

Navigate to: http://localhost:3000

## Testing the File Picker

1. Click on **Load Data** in the navigation
2. You should see a new "File Selection" card
3. Click the **"Select Files"** button
4. Choose one or multiple `.txt`, `.xls`, or `.xlsx` files from your computer
5. You'll see the selected files listed
6. Click **"Load Files"**
7. Wait for success message: "Loaded X file(s) successfully"
8. You should automatically navigate to the preprocess page

### Alternative: Manual Path Entry

You can still enter file paths manually:
1. Scroll down to "File Paths" section
2. Enter paths (one per line):
   ```
   C:\path\to\data1.txt
   C:\path\to\data2.txt
   ```
3. Click **"Load Files"**

## Testing the Preprocessing

After loading data, you'll be on the Preprocess page:

### Test 1: No Filter
1. Select **"None"** from the Filter dropdown
2. Click **"Apply Filter"**
3. Watch the progress: "Processing... 10%" → "Processing... 100%"
4. Success message: "Preprocessing completed"
5. Signal preview updates (should look the same as raw data)

### Test 2: Butterworth Filter
1. Select **"Butterworth"** from the Filter dropdown
2. Set parameters:
   - Cutoff Frequency: `1000` Hz
   - Order: `4`
3. Click **"Apply Filter"**
4. Watch progress bar increase: 10% → 30% → 70% → 90% → 100%
5. Success message: "Preprocessing completed"
6. Signal preview updates (should show smoothed data)

### Test 3: Savitzky-Golay Filter
1. Select **"Savitzky-Golay"** from the Filter dropdown
2. Set parameters:
   - Window Length: `51` (must be odd)
   - Polynomial Order: `3`
3. Click **"Apply Filter"**
4. Watch progress bar increase
5. Success message: "Preprocessing completed"
6. Signal preview updates (should show smoothed data)

## Troubleshooting

### Backend won't start
- **Error**: `Address already in use`
  - **Fix**: Kill process on port 8765 or change port in `service/app.py`
  
- **Error**: `ModuleNotFoundError: No module named 'scipy'`
  - **Fix**: `pip install scipy`

### Frontend won't start
- **Error**: `Cannot find module 'next'`
  - **Fix**: `cd ui-web && npm install`

### File picker shows no files
- Check browser console for errors (F12)
- Make sure files are `.txt`, `.xls`, or `.xlsx` format
- Try manual path entry instead

### Preprocessing hangs at 0%
- **Old issue**: This should now be fixed!
- Check backend terminal for errors
- Check browser console (F12) → Network tab → look for WebSocket connection
- WebSocket should connect to: `ws://127.0.0.1:8765/ws/progress?taskId=...`

### WebSocket not connecting
- Make sure backend is running
- Check CORS settings in `service/app.py`
- Check browser console for WebSocket errors
- Try refreshing the page

### "Failed to load files" error
- Check that file paths are valid (for manual entry)
- Check that files exist and are readable
- Check backend terminal for detailed error messages

## Console Debugging

### Backend Console (Terminal)
Look for:
```
INFO:     "POST /api/files/upload HTTP/1.1" 200 OK
INFO:     "POST /api/preprocess/run HTTP/1.1" 200 OK
INFO:     "WebSocket /ws/progress" opened
```

### Frontend Console (Browser DevTools)
Open with F12, check:
- **Console tab**: JavaScript errors
- **Network tab**: 
  - API calls should return 200 OK
  - WebSocket should show "101 Switching Protocols"
- **Sources tab**: Check if files loaded correctly

## Next Steps

After successful testing:
1. Continue to **Detect** page to find peaks
2. Use **Analyze** page for peak analysis
3. Check **Export** page to save results

## Features Summary

### ✅ Working Features
- ✅ File picker with multiple file selection
- ✅ Manual file path entry (legacy method)
- ✅ File upload to backend
- ✅ Preprocessing with filters:
  - Butterworth low-pass filter
  - Savitzky-Golay smoothing filter
  - None (pass-through)
- ✅ Real-time progress tracking via WebSocket
- ✅ Live signal preview
- ✅ Automatic navigation between steps

### 📝 Known Limitations
- File upload size not limited (may be slow for very large files)
- No cancel button for long-running operations
- No real-time preview while filtering (only after completion)

## Support

If you encounter issues not covered here:
1. Check `docs/WEB_UI_FIXES.md` for technical details
2. Review backend logs in terminal
3. Check browser console for frontend errors
4. Check `logs/app.log` for detailed backend logs



