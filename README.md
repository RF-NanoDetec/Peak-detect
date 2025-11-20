# Peak Analysis Tool

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-Proprietary-red.svg)

A robust web-based application for scientific peak analysis in time series data. Designed for researchers and analysts who need to detect, measure, and characterize signal peaks with precision.

![Application Screenshot](resources/images/screenshot.png)

## 🚀 Features

- **Modern Web Interface**: Browser-based UI with responsive design.
- **Flexible Data Import**: Load single files or process batch datasets with protocol metadata support.
- **Advanced Signal Processing**: Apply customizable filters (Butterworth, Savitzky-Golay) to reduce noise.
- **Photon Counter Correction**: Dead-time correction for accurate photon counting measurements.
- **Intelligent Peak Detection**: Automatic and manual peak detection with configurable parameters (Prominence, Width, Distance).
- **Double Peak Analysis**: Specialized workflow for analyzing paired peak events.
- **Interactive Visualization**: Explore results with high-performance plots powered by uPlot.
- **Client-Side Image Export**: Save high-quality charts directly from the browser.
- **Data Export**: Save results as CSV/Excel files including all experiment metadata.
- **Standalone Deployment**: Single executable with embedded web UI.

## 📖 User Manual

For detailed usage instructions, please refer to the [User Manual](docs/USER_MANUAL.md).

## 🔧 Installation

### For End Users

#### Option 1: Run Standalone Executable (Recommended)
1. Download `PeakService.exe` from the releases.
2. Double-click `PeakService.exe`.
3. Your browser will automatically open to `http://127.0.0.1:8765`.

#### Option 2: Run from Source
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the service: `python -m service.app`.

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

## 🔄 Performance Considerations

- Large datasets (>1M points) are automatically decimated for visualization.
- The web UI uses a dedicated data worker that owns typed-array buffers for smooth performance.
- Full-resolution arrays are delivered via binary endpoints.
- WebSocket progress updates provide real-time feedback.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## 📜 License

© 2025 Dr. Lucjan Grzegorzewski All rights reserved.

This software is proprietary and confidential. Unauthorized copying, transfer, or use in any medium is strictly prohibited without prior written consent.

## 📧 Contact

For questions or support, please contact:
- Email: lgrzegor@physnet.uni-hamburg.de
- GitHub Issues: Submit issues through the repository's issue tracker
