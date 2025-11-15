# Web UI Fixes - File Picker and Preprocessing

## Summary
Fixed two major issues in the Next.js web interface:
1. Added file picker functionality for selecting multiple files
2. Fixed preprocessing endpoint to process data asynchronously with progress tracking

## Changes Made

### 1. File Picker Functionality

#### Frontend Changes (`ui-web/app/load/page.tsx`)
- Added file input element with multiple file selection support
- Created `handleFileSelect` function to manage file selection
- Added UI to display selected files
- Implemented "Select Files" button with Upload icon
- Added "OR" separator between file picker and manual path entry
- Files can be selected via picker OR typed manually (mutually exclusive)

#### API Client (`ui-web/lib/apiClient.ts`)
- Added `uploadFiles()` method to handle file uploads
- Uses FormData to send files to backend
- Increased timeout to 60 seconds for large file uploads

#### Backend (`service/handlers.py`)
- Added `/api/files/upload` endpoint
- Accepts multiple files via multipart/form-data
- Saves uploaded files to temporary directory
- Processes files using existing `load_data_from_paths` logic
- Cleans up temporary files after processing

### 2. Preprocessing Fixes

#### Backend (`service/handlers.py`)
- **Complete rewrite of `/api/preprocess/run` endpoint**
- Now returns `taskId` for WebSocket progress tracking (was returning `resultId` before)
- Processes data asynchronously using TaskManager
- Implements actual filtering logic:
  - **Butterworth Filter**: Low-pass filter with configurable cutoff frequency and order
  - **Savitzky-Golay Filter**: Smoothing filter with configurable window length and polynomial order
  - **None**: Pass-through (no filtering)
- Progress updates sent via WebSocket (10%, 30%, 70%, 90%, 100%)
- Returns new `resultId` with filtered data

#### Backend Models (`service/models.py`)
- Added filter parameters to `Params` model:
  - `filter_enabled`: bool (default: True)
  - `filter_type`: "none" | "butterworth" | "savgol" (default: "none")
  - `filter_cutoff_freq`: float (default: 1000.0)
  - `butter_order`: int (default: 4)
  - `savgol_window`: int (default: 51)
  - `savgol_polyorder`: int (default: 3)

#### Frontend (`ui-web/app/preprocess/page.tsx`)
- Updated `useEffect` to handle completed preprocessing
- Updates `resultId` with filtered data when processing completes
- Fetches and displays preview of filtered data
- Progress bar now shows actual progress from WebSocket

## How to Test

### Testing File Picker

1. **Start the backend server**:
   ```bash
   cd service
   python -m service.app
   ```

2. **Start the Next.js dev server**:
   ```bash
   cd ui-web
   npm run dev
   ```

3. **Navigate to Load Data page** (http://localhost:3000/load)

4. **Test file selection**:
   - Click "Select Files" button
   - Choose one or multiple `.txt`, `.xls`, or `.xlsx` files
   - Verify selected files are displayed
   - Click "Load Files"
   - Should see success message and navigate to preprocess page

5. **Test manual path entry** (still works):
   - Enter file paths in textarea (one per line)
   - Click "Load Files"
   - Should work as before

### Testing Preprocessing

1. **After loading data**, you'll be on the preprocess page

2. **Test different filter types**:
   
   **A. No Filter**:
   - Select "None" from filter dropdown
   - Click "Apply Filter"
   - Should complete quickly with "Preprocessing completed" message
   
   **B. Butterworth Filter**:
   - Select "Butterworth" from filter dropdown
   - Set Cutoff Frequency (e.g., 1000 Hz)
   - Set Order (e.g., 4)
   - Click "Apply Filter"
   - Watch progress bar increase from 0% → 100%
   - Should see filtered signal in preview
   
   **C. Savitzky-Golay Filter**:
   - Select "Savitzky-Golay" from filter dropdown
   - Set Window Length (e.g., 51, must be odd)
   - Set Polynomial Order (e.g., 3)
   - Click "Apply Filter"
   - Watch progress bar increase from 0% → 100%
   - Should see smoothed signal in preview

3. **Verify progress tracking**:
   - During processing, button should show "Processing... X%"
   - Progress should update in real-time via WebSocket
   - Should NOT hang at 0% anymore

4. **Check console for errors**:
   - Open browser DevTools
   - Check Network tab for WebSocket connection
   - Check Console for any errors

## Technical Details

### WebSocket Flow
1. Frontend calls `/api/preprocess/run` with parameters
2. Backend returns `taskId`
3. Frontend connects to WebSocket: `/ws/progress?taskId={taskId}`
4. Backend worker thread processes data and sends progress updates
5. Frontend receives updates and displays progress
6. On completion, backend sends result with new `resultId`
7. Frontend updates state and fetches preview data

### Filter Implementation
- Uses `scipy.signal` for filter implementations
- Butterworth: `butter()` and `filtfilt()` for zero-phase filtering
- Savitzky-Golay: `savgol_filter()` with automatic window validation
- Filters are applied to amplitude data, time data remains unchanged

## Files Modified

### Frontend
- `ui-web/app/load/page.tsx` - Added file picker UI
- `ui-web/lib/apiClient.ts` - Added uploadFiles method
- `ui-web/app/preprocess/page.tsx` - Fixed result handling

### Backend
- `service/handlers.py` - Added upload endpoint, rewrote preprocess endpoint
- `service/models.py` - Added filter parameters to Params model

### Documentation
- `docs/WEB_UI_FIXES.md` - This file

## Known Issues / Future Improvements

1. **File Upload Size**: Currently no size limit on uploads, may need to add
2. **Filter Preview**: Could add real-time preview before applying filter
3. **Filter Validation**: Could add more parameter validation (e.g., Nyquist frequency check)
4. **Error Handling**: Could improve error messages for invalid filter parameters
5. **Cancel Operation**: Could add ability to cancel long-running preprocessing tasks

## Dependencies

Ensure these are installed:
- Backend: `scipy` (already in requirements.txt)
- Frontend: `axios`, `react`, `next` (already in package.json)

## Deployment Notes

When deploying to production:
1. Set proper CORS origins in `service/app.py`
2. Configure upload size limits in FastAPI/uvicorn
3. Set `NEXT_PUBLIC_API_URL` environment variable in Next.js
4. Enable HTTPS for WebSocket connections (wss://)



