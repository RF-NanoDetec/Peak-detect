# UI Simplification - Test Checklist

## Overview
This document provides a checklist for testing the UI simplification changes made to the web application.

## Changes Implemented

### 1. Manual File Path Entry Removed
- Manual file path textarea has been removed
- File picker is now the only way to load files
- "OR" separator has been removed

### 2. Recent Files Now Shows Sessions
- Recent files are stored as file groups/sessions
- Display format:
  - Single file: "data.txt"
  - Multiple files: "data1.txt + 3" (shows first file + count of remaining files)
- Clicking a recent session opens the file picker for re-selection
- Maximum 5 recent sessions are stored

### 3. Time Resolution Display in Milliseconds
- UI now displays time resolution in milliseconds
- Default: 0.1 ms (was 0.0001 seconds)
- Backend still uses seconds internally
- Conversion is automatic

## Test Checklist

### File Picker Tests

- [ ] **Single File Selection**
  - Click "Select Files" button
  - Choose one file
  - Verify file name appears in selected files list
  - Click "Load Files"
  - Verify success message
  - Verify navigation to preprocess page

- [ ] **Multiple File Selection**
  - Click "Select Files" button
  - Choose multiple files (e.g., 5 files)
  - Verify all file names appear in selected files list
  - Verify count shows correctly (e.g., "5 file(s) selected")
  - Click "Load Files"
  - Verify success message
  - Verify navigation to preprocess page

- [ ] **No Files Selected**
  - Do NOT select any files
  - Try to click "Load Files"
  - Verify button is disabled

### Recent Files/Sessions Tests

- [ ] **Single File Session**
  - Load a single file
  - Navigate back to Load Data page
  - Verify "Recent Sessions" card appears
  - Verify session shows just the filename (e.g., "data.txt")

- [ ] **Multiple Files Session**
  - Load multiple files (e.g., 3 files: "file1.txt", "file2.txt", "file3.txt")
  - Navigate back to Load Data page
  - Verify session shows "file1.txt + 2" format

- [ ] **Recent Sessions Limit**
  - Load files 6 different times (6 different sessions)
  - Verify only the last 5 sessions are displayed
  - Verify oldest session is removed

- [ ] **Clicking Recent Session**
  - Click on a recent session entry
  - Verify file picker opens
  - Verify toast message appears: "Loading session with X file(s). Please re-select the files."

- [ ] **Duplicate Session Handling**
  - Load same files in same order twice
  - Verify only one entry for that session appears in recent files
  - Verify it's at the top (most recent)

### Time Resolution Tests

- [ ] **Default Value**
  - Open Load Data page
  - Verify time resolution shows 0.1 (milliseconds)
  - Verify label says "Time Resolution (milliseconds)"

- [ ] **Change Value**
  - Change time resolution to 0.5 ms
  - Load files
  - Verify files load successfully
  - Backend should receive 0.0005 seconds

- [ ] **Edge Cases**
  - Try 0.01 ms (very small)
  - Try 10 ms (larger value)
  - Try 1.0 ms
  - Verify all values work correctly

- [ ] **Value Persistence**
  - Set time resolution to a custom value
  - Reload the page
  - Verify value persists (should be saved in localStorage)

### Integration Tests

- [ ] **Complete Workflow**
  1. Set time resolution to 0.2 ms
  2. Select 3 files using file picker
  3. Click Load Files
  4. Verify files load successfully
  5. Go to Preprocess page
  6. Apply a filter
  7. Verify preprocessing works
  8. Navigate back to Load Data
  9. Verify recent session shows "file1.txt + 2"
  10. Click on the recent session
  11. Verify file picker opens

- [ ] **Browser Compatibility**
  - Test in Chrome
  - Test in Firefox
  - Test in Edge
  - Verify file picker works in all browsers

### Regression Tests

- [ ] **File Upload Endpoint**
  - Verify files are uploaded correctly to backend
  - Check backend logs for successful uploads

- [ ] **WebSocket Progress**
  - After loading files, go to Preprocess
  - Apply filter
  - Verify progress updates in real-time

- [ ] **Data Preview**
  - After loading files
  - Verify data preview appears on Preprocess page

## Expected Results

### UI Changes
- Cleaner, simpler Load Data page
- No textarea for manual path entry
- Only file picker button visible
- Recent Sessions shows file groups nicely formatted

### Functionality
- All file operations work correctly
- Time resolution conversion is transparent to user
- Recent sessions are stored and displayed correctly
- Clicking recent sessions opens file picker

## Known Limitations

1. **Recent Files Click Behavior**: When clicking a recent session, users must re-select files manually. We cannot reconstruct File objects from stored filenames for security reasons.

2. **File Order**: If users select the same files in a different order, it will create a new session entry.

3. **File Paths Not Stored**: Only filenames are stored, not full paths, for privacy and portability.

## Troubleshooting

### Issue: Recent sessions not appearing
- Check browser localStorage (key: "peak-tool-data")
- Clear localStorage and try again
- Verify files were loaded successfully

### Issue: Time resolution shows wrong value
- Check params store (key: "peak-tool-params")
- Verify default is 0.0001 seconds (= 0.1 ms)

### Issue: File picker not opening
- Check browser console for errors
- Verify browser supports file input
- Try different browser

## Test Environment

- **Frontend**: http://localhost:3000
- **Backend**: http://127.0.0.1:8765
- **Browser**: Chrome/Firefox/Edge latest versions

## Sign-off

- [ ] All tests passed
- [ ] No regressions found
- [ ] Ready for deployment

**Tested by**: _______________
**Date**: _______________
**Notes**: _______________



