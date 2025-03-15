# Peak Analysis Tool

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-Proprietary-red.svg)

A robust Python application for scientific peak analysis in time series data. Designed for researchers and analysts who need to detect, measure, and characterize signal peaks with precision.

![Application Screenshot](resources/images/screenshot.png)

## 🚀 Features

- **Flexible Data Import**: Load single files or process batch datasets
- **Advanced Signal Processing**: Apply customizable filters to reduce noise
- **Intelligent Peak Detection**: Automatic and manual peak detection with configurable parameters
- **Comprehensive Analysis**: Calculate and visualize key peak metrics:
  - Peak height and prominence
  - Width at various relative heights
  - Area under the curve
  - Inter-peak intervals
- **Interactive Visualization**: Explore results with interactive plots and detailed peak views
- **Data Export**: Save results as CSV files or high-resolution plots
- **Performance Optimized**: Efficient processing of large datasets with multi-threading support
- **Modern UI**: Clean and responsive interface with light/dark theme support

## 📋 Requirements

- **Python 3.8+**
- **Core Dependencies**:
  - NumPy (1.20+)
  - Pandas (1.3+)
  - Matplotlib (3.4+)
  - SciPy (1.7+)
  - Tkinter (8.6+)
  - Seaborn (0.11+)
  - Numba (optional, for performance acceleration)

## 🔧 Installation

### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/peak-analysis-tool.git
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

4. Run the application:
   ```bash
   python main.py
   ```

### Installation for Development

For those who want to contribute to development:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

## 📊 Project Structure

```
peak_analysis_tool/
│
├── core/                  # Core analysis functionality
│   ├── __init__.py        # Package initialization
│   ├── peak_detection.py  # Peak detection algorithms and classes
│   └── peak_analysis_utils.py  # Signal processing utilities
│
├── plotting/              # Data visualization components
│   ├── __init__.py
│   ├── raw_data.py        # Raw data visualization
│   ├── data_processing.py # Data processing visualization
│   ├── peak_visualization.py # Peak visualization functions
│   └── analysis_visualization.py # Statistical visualization
│
├── ui/                    # User interface components
│   ├── __init__.py
│   ├── theme.py           # Theme management (light/dark)
│   ├── tooltips.py        # Enhanced tooltips functionality
│   └── status_indicator.py # Status indicator widget
│
├── config/                # Application configuration
│   ├── __init__.py
│   ├── environment.py     # Environment settings
│   └── settings.py        # Application settings
│
├── utils/                 # General utilities
│   ├── __init__.py
│   └── performance.py     # Performance monitoring and optimization
│
├── resources/             # Static resources
│   └── images/            # Application images and icons
│
├── main.py                # Main application entry point
├── app.py                 # Application initialization
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── CHANGES.md             # Changelog
├── INSTALL.md             # Detailed installation instructions
└── README.md              # This file
```

## 📖 Usage Guide

### Basic Usage

1. **Launch the Application**:
   ```bash
   python main.py
   ```

2. **Load Data**:
   - Click "Load File" to import a single data file
   - Or use "Batch Mode" to process multiple files

3. **Preprocess Data**:
   - Apply filters to reduce noise (Low-pass, High-pass)
   - Use the auto-cutoff feature for optimal filtering

4. **Detect Peaks**:
   - Set the detection threshold or use auto-threshold
   - Adjust minimum peak distance and width parameters
   - Run peak detection

5. **Analyze Results**:
   - Review the detected peaks in the grid view
   - Examine peak characteristics in the data table
   - View statistical summaries and distributions

6. **Export Results**:
   - Save peak data to CSV
   - Export plots as high-resolution images
   - Take screenshots of specific views

### Advanced Usage

#### Custom Filters

Customize the filter parameters for specific signal types:

```python
# Example for optimizing filter settings
cutoff = adjust_lowpass_cutoff(signal, sampling_rate, peak_count=20, normalization=0.5)
filtered_signal = apply_butterworth_filter(4, cutoff, 'lowpass', sampling_rate, raw_signal)
```

#### Batch Processing

For processing multiple files:

1. Enable Batch Mode in the interface
2. Select a directory containing data files
3. Configure common parameters for all files
4. Start batch processing
5. Review aggregated results

## 🔄 Performance Considerations

- For large datasets (>1M points), consider using decimation options
- Enable multi-threading for batch processing by increasing MAX_WORKERS in config/settings.py
- Monitor memory usage through the app's status bar

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

Please include tests for any new functionality and ensure documentation is updated.

## 📜 License

© 2024 Lucjan & Silas. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, transfer, or use in any medium is strictly prohibited without prior written consent.

## 📧 Contact

For questions or support, please contact:
- Email: lgrzegor@physnet.uni-hamburg.de
- GitHub Issues: Submit issues through the repository's issue tracker 