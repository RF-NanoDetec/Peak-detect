# Peak Analysis Tool - Cutover Guide
## Transitioning from Tkinter to Web UI

### Overview

This guide documents the transition from the legacy Tkinter desktop UI to the modern web-based interface. The new architecture consists of:

1. **FastAPI Backend** - Handles all data processing, peak detection, and analysis
2. **Next.js Web UI** - Modern, responsive interface accessible via browser
3. **Standalone Deployment** - Single executable with embedded web UI

---

## For Users

### What's New

✅ **Modern Interface** - Clean, responsive design that works on any screen size  
✅ **Better Performance** - Optimized charts and data handling  
✅ **Browser-Based** - No installation required, just run the EXE  
✅ **Same Features** - All analysis capabilities from the old UI are preserved  

### How to Use the New Version

#### Option 1: Run the Standalone Executable (Recommended)

1. Double-click `PeakService.exe`
2. Your browser will open automatically to `http://127.0.0.1:8765`
3. Begin your analysis!

#### Option 2: Use the Launcher (Optional)

1. Double-click `launch_ui.py` (requires Python installed)
2. The launcher will start the service and open your browser

### Quick Start

1. **Load Data**: Click "Load" in the sidebar, select your data files
2. **Preprocess**: Apply filters (Butterworth, Savitzky-Golay, or none)
3. **Detect Peaks**: Configure detection parameters and run analysis
4. **Analyze**: View comprehensive statistics and charts
5. **Export**: Save results as CSV or export plots as images

---

## For Developers

### Architecture Changes

#### Old Architecture (Tkinter)
```
main.py → Tkinter UI → Core Functions → Results
```

#### New Architecture (Web)
```
Browser → Next.js UI → FastAPI Backend → Core Functions → Results
                ↓
          WebSocket for Progress
```

### Directory Structure

```
peak_analysis/
├── service/              # FastAPI backend
│   ├── app.py           # Main server + static file serving
│   ├── handlers.py      # API endpoints
│   ├── models.py        # Request/response models
│   ├── jobs.py          # Background task management
│   ├── events.py        # WebSocket progress broadcasting
│   └── storage.py       # In-memory data store
│
├── ui-web/              # Next.js frontend
│   ├── app/             # Page components
│   ├── components/      # Reusable UI components
│   ├── lib/             # API client, stores, utilities
│   └── out/             # Built static files (after build)
│
├── core/                # Pure business logic (UI-agnostic)
│   ├── data_analysis.py
│   ├── peak_detection.py
│   ├── file_handler.py
│   └── service_functions.py  # API-friendly wrappers
│
├── launcher/            # Optional Python launcher
│   └── launch_ui.py
│
└── tools/               # Build scripts
    ├── build_all.bat    # Complete build process
    └── build_service.bat # Backend-only build
```

### API Endpoints

#### File Operations
- `POST /api/files/open` - Load data files
- `POST /api/files/upload` - Upload files via browser
- `GET /api/results/{id}` - Get loaded data

#### Data Preview
- `GET /api/data/preview` - Get decimated data for plotting
- `GET /api/data/plot-image` - Get matplotlib-rendered plot

#### Preprocessing
- `POST /api/preprocess/run` - Apply filters (async with progress)
- `POST /api/preprocess/auto-cutoff` - Auto-calculate filter cutoff

#### Peak Detection
- `POST /api/detect/run` - Detect peaks
- `POST /api/detect/auto-threshold` - Auto-calculate threshold
- `POST /api/detect/inspect-peaks` - Generate peak inspection plots

#### Double Peak Analysis
- `POST /api/double-peak/analyze` - Analyze double peak patterns

#### Export
- `GET /api/export/peaks/csv` - Export peaks to CSV
- `POST /api/export/double-peaks/csv` - Export double peaks to CSV
- `GET /api/export/plot/image` - Export plot as image (PNG/SVG/PDF)

#### System
- `GET /api/health` - Health check
- `GET /api/version` - Get version info
- `GET /api/params` - Get default parameters
- `PUT /api/params` - Update default parameters

#### WebSocket
- `WS /ws/progress?taskId={id}` - Real-time progress updates

### Building from Source

#### Prerequisites
- Python 3.8+
- Node.js 16+
- PyInstaller (`pip install pyinstaller`)

#### Full Build Process

```bash
# Run the comprehensive build script
tools\build_all.bat
```

This will:
1. Install npm dependencies
2. Build the Next.js UI to static files
3. Package the backend with embedded UI into a single EXE
4. Create distribution folder with all necessary files

#### Manual Build Steps

```bash
# 1. Build the web UI
cd ui-web
npm install
npm run build  # Creates ui-web/out/

# 2. Build the backend
pyinstaller --onefile --name PeakService \
    --add-data "config;config" \
    --add-data "ui-web/out;ui-web/out" \
    --hidden-import "uvicorn.logging" \
    service/app.py

# 3. Test
dist\PeakService.exe
```

### Development Workflow

#### Running in Development Mode

```bash
# Terminal 1: Start the backend
python -m service.app

# Terminal 2: Start the frontend dev server
cd ui-web
npm run dev
```

Navigate to `http://localhost:3000` for hot-reload development.

#### Testing

```bash
# Backend tests
pytest tests/

# Frontend (if you add tests)
cd ui-web
npm test
```

---

## Migration Checklist

### Phase 1: Preparation ✓
- [x] Extract all business logic to `core/` modules
- [x] Create FastAPI backend with all endpoints
- [x] Build Next.js UI with all features
- [x] Test end-to-end workflow

### Phase 2: Deployment ✓
- [x] Configure Next.js for static export
- [x] Update backend to serve static files
- [x] Create build scripts
- [x] Update launcher

### Phase 3: Legacy Cleanup ✓
- [x] Remove Tkinter UI files (`ui/`, `main.py`)
- [x] Remove Tkinter dependencies from `requirements.txt`
- [x] Update installer to only include new components
- [x] Update documentation to reflect web UI only

---

## Troubleshooting

### Service Won't Start

**Problem**: "Address already in use" error  
**Solution**: Another process is using port 8765. Check Task Manager or change the port in `service/app.py`

**Problem**: Browser doesn't open automatically  
**Solution**: Manually navigate to `http://127.0.0.1:8765`

### UI Not Loading

**Problem**: Blank page or "Cannot connect"  
**Solution**: Ensure the backend service is running. Check logs for errors.

**Problem**: Static files not found  
**Solution**: Rebuild the UI with `npm run build` in the `ui-web` directory

### Data Processing Issues

**Problem**: Processing hangs or times out  
**Solution**: Check the browser console (F12) and backend logs for specific errors

**Problem**: WebSocket connection fails  
**Solution**: Ensure WebSocket protocol is supported by your browser (all modern browsers support it)

---

---

## Performance Notes

### Improvements
- ✅ Faster chart rendering (optimized decimation)
- ✅ Responsive progress tracking (WebSocket)
- ✅ Caching for expensive operations (auto-cutoff, peak inspection)
- ✅ Parallel processing where applicable

### Known Limitations
- First-time startup may be slower (EXE unpacking)
- Large datasets (>1M points) are automatically decimated for visualization
- Browser cache may need clearing after updates

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review backend logs (console output)
3. Check browser console (F12 → Console tab)
4. Refer to the API documentation (`http://127.0.0.1:8765/api/docs`)

---

## Changelog

### Version 2.0.0 - Web UI Release
- ✨ Complete web-based interface
- ✨ Real-time progress tracking
- ✨ Improved chart performance
- ✨ Modern, responsive design
- ✨ Standalone executable deployment
- ✨ All original features preserved and enhanced

---

*Last Updated: November 2025*

