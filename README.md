# Peak Analysis Tool

A Python application for analyzing signal data to detect and characterize peaks.

## Features

- Load single or batch files for analysis
- Apply filtering to reduce noise
- Detect peaks with customizable parameters
- Analyze peak properties (width, height, area)
- Generate visualizations and reports

## Project Structure

```
peak_analysis_tool/
│
├── core/                  # Core functionality
│   ├── __init__.py
│   ├── peak_detection.py  # Peak detection algorithms
│   └── peak_analysis_utils.py  # Analysis utilities
│
├── plotting/              # Plotting functionality
│   ├── __init__.py
│   ├── raw_data.py        # Raw data visualization
│   ├── data_processing.py # Data processing functions
│   ├── peak_visualization.py # Peak visualization
│   └── analysis_visualization.py # Analysis visualization
│
├── ui/                    # UI components
│   ├── __init__.py
│   ├── theme.py           # Theme management
│   ├── tooltips.py        # Enhanced tooltips
│   └── status_indicator.py # Status indicator widget
│
├── config/                # Configuration
│   ├── __init__.py
│   └── settings.py        # Application settings
│
├── utils/                 # General utilities
│   ├── __init__.py
│   └── performance.py     # Performance monitoring
│
├── resources/             # Static resources
│   └── images/            # Application images
│
├── main.py                # Main application entry point
└── requirements.txt       # Dependencies
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python main.py
   ```

## Usage

1. Load a data file using the "Load File" button
2. Apply filtering to reduce noise
3. Detect peaks with customizable parameters
4. Analyze peak properties and generate visualizations
5. Export results to CSV or images

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Tkinter
- Seaborn

## License

© 2024 All rights reserved. 