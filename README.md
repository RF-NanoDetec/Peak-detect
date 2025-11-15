# Peak Analysis Tool

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-Proprietary-red.svg)

A robust web-based application for scientific peak analysis in time series data. Designed for researchers and analysts who need to detect, measure, and characterize signal peaks with precision.

![Application Screenshot](resources/images/screenshot.png)

## 🚀 Features

- **Modern Web Interface**: Browser-based UI with responsive design
- **Flexible Data Import**: Load single files or process batch datasets
- **Advanced Signal Processing**: Apply customizable filters to reduce noise
- **Photon Counter Correction**: Dead-time correction for accurate photon counting measurements
- **Protocol Metadata**: Capture and store experiment conditions for reproducibility
- **Intelligent Peak Detection**: Automatic and manual peak detection with configurable parameters
- **Comprehensive Analysis**: Calculate and visualize key peak metrics:
  - Peak height and prominence
  - Width at various relative heights
  - Area under the curve
  - Inter-peak intervals
- **Interactive Visualization**: Explore results with interactive plots and detailed peak views
- **Real-time Progress Tracking**: WebSocket-based progress updates
- **Data Export**: Save results as CSV files or high-resolution plots (includes protocol metadata)
- **Performance Optimized**: Efficient processing of large datasets with async processing
- **Standalone Deployment**: Single executable with embedded web UI

## 📋 Requirements

- **Python 3.8+**
- **Core Dependencies**:
  - NumPy (1.20+)
  - Pandas (1.3+)
  - Matplotlib (3.4+)
  - SciPy (1.7+)
  - FastAPI (0.115+)
  - Uvicorn (0.30+)
  - Seaborn (0.11+)
  - Numba (required for optimized peak math)
- **Web UI**:
  - Node.js 16+ (for building the UI)
  - Next.js 14+ (React framework)

## 🔧 Installation

### For End Users

#### Option 1: Run Standalone Executable (Recommended)

1. Download `PeakService.exe` from the releases
2. Double-click `PeakService.exe`
3. Your browser will automatically open to `http://127.0.0.1:8765`
4. Begin your analysis!

#### Option 2: Run from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/RF-NanoDetec/Peak-detect.git
   cd peak-analysis-tool
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the service:
   ```bash
   python -m service.app
   ```

5. Open your browser to `http://127.0.0.1:8765`

### For Developers

#### Development Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install Node.js dependencies for the web UI
cd ui-web
npm install
```

#### Running in Development Mode

```bash
# Terminal 1: Start the backend
python -m service.app

# Terminal 2: Start the frontend dev server
cd ui-web
npm run dev
```

Navigate to `http://localhost:3000` for hot-reload development.

#### Building from Source

```bash
# Complete build process (Windows)
tools\build_all.bat
```

This creates a standalone executable in the `dist/` directory.

#### Running Tests

```bash
# Backend tests
pytest

# Frontend tests (if available)
cd ui-web
npm test
```


## Project Structure

```
peak_analysis_tool/
|-- config/                # Environment + settings helpers
|   |-- __init__.py
|   `-- environment.py
|-- core/                  # Pure analysis + data utilities
|   |-- data_analysis.py
|   |-- data_utils.py
|   |-- file_handler.py
|   |-- peak_analysis_utils.py
|   |-- peak_detection.py
|   |-- performance.py
|   |-- photon_correction.py
|   |-- service_functions.py
|   `-- timing.py
|-- service/               # FastAPI backend + plotting/export glue
|   |-- app.py
|   |-- handlers.py
|   |-- jobs.py
|   |-- events.py
|   |-- models.py
|   |-- plotting.py
|   |-- cache.py
|   `-- storage.py
|-- ui-web/                # Next.js front-end
|   |-- app/
|   |-- components/
|   |-- hooks/
|   |-- lib/
|   `-- public/
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- data fixtures (.txt)
|-- docs/                  # Documentation hub (see docs/README.md)
|   |-- figures/
|   |-- guides/
|   |-- legacy/
|   |-- references/
|   `-- reports/
|-- tools/                 # Build/test helpers
|-- installer/             # Inno Setup scripts
|-- launcher/              # Convenience launch scripts
|-- resources/             # Static imagery/icons
|-- scripts/               # Documentation tooling
|-- data/                  # Sample datasets (git-ignored by default)
|-- requirements.txt
|-- PeakService.spec
`-- README.md
```

## 📖 Usage Guide

### Basic Usage

1. **Launch the Application**:
   - Run `PeakService.exe` (standalone)
   - Or `python -m service.app` (from source)
   - Your browser will open to `http://127.0.0.1:8765`

2. **Load Data**:
   - Click "Load" in the sidebar
   - Select your data files or upload via drag-and-drop
   - View the loaded data in the interactive plot

3. **Preprocess Data**:
   - Choose a filter type (Butterworth, Savitzky-Golay, or none)
   - Set filter parameters or use auto-cutoff
   - Apply preprocessing and view results in real-time

4. **Detect Peaks**:
   - Configure detection parameters (threshold, minimum distance, minimum width)
   - Use auto-threshold for automatic threshold calculation
   - Run peak detection
   - View detected peaks highlighted on the plot

5. **Analyze Results**:
   - Review detected peaks in the interactive table
   - Explore peak statistics and distributions
   - Use the peak inspector to examine individual peaks
   - Analyze double peak patterns if present

6. **Export Results**:
   - Export peak data to CSV
   - Save plots as PNG, SVG, or PDF
   - Download analysis results for further processing

### Advanced Features

#### Real-time Progress Tracking

Long-running operations show real-time progress through WebSocket connections, allowing you to monitor:
- Preprocessing progress
- Peak detection status
- Analysis completion

#### API Access

The backend provides a REST API accessible at `http://127.0.0.1:8765/api/docs` for:
- Programmatic data loading
- Automated analysis workflows
- Custom integrations

#### Custom Analysis Scripts

You can use the core analysis modules directly in your Python scripts:

```python
from core.peak_detection import PeakDetector
from core.data_analysis import analyze_peaks

# Your custom analysis code here
detector = PeakDetector(data, time)
peaks = detector.detect_peaks(threshold=0.5)
results = analyze_peaks(data, time, peaks)
```

## 🔄 Performance Considerations

- Large datasets (>1M points) are automatically decimated for visualization
- The web UI now uses a dedicated data worker that owns typed-array buffers, streams parsed data off the main thread, and serves zoom windows without blocking React
- Full-resolution time/amplitude arrays are delivered through a binary `/api/results/{id}/binary` endpoint so the frontend can stream typed arrays directly into workers without JSON overhead
- Original precision is maintained for all calculations
- Async processing prevents UI blocking during long operations
- Caching optimizes repeated operations
- WebSocket progress updates provide real-time feedback

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

Please include tests for any new functionality and ensure documentation is updated.

## 📜 License

© 2025 Dr. Lucjan Grzegorzewski All rights reserved.

This software is proprietary and confidential. Unauthorized copying, transfer, or use in any medium is strictly prohibited without prior written consent.

## 📧 Contact

For questions or support, please contact:
- Email: lgrzegor@physnet.uni-hamburg.de
- GitHub Issues: Submit issues through the repository's issue tracker 
