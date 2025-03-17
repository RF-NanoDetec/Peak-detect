# Peak Analysis Tool User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [User Interface](#user-interface)
4. [Data Loading](#data-loading)
5. [Signal Processing](#signal-processing)
6. [Peak Detection](#peak-detection)
7. [Analysis Workflows](#analysis-workflows)
8. [Visualization Options](#visualization-options)
9. [Data Export](#data-export)
10. [Batch Processing](#batch-processing)
11. [Advanced Features](#advanced-features)
12. [Troubleshooting](#troubleshooting)
13. [Technical Reference](#technical-reference)

## Introduction

The Peak Analysis Tool is a sophisticated signal processing and analysis software designed for detecting, measuring, and analyzing peaks in time-series data. This tool is particularly useful for scientific research, laboratory analysis, and signal processing applications where identifying signal peaks and their characteristics is crucial.

### Key Features

- **High-performance signal processing** with optimized algorithms for large datasets
- **Advanced peak detection** with customizable parameters for different signal types
- **Interactive visualization** with zooming, panning, and peak inspection capabilities
- **Comprehensive analysis** of peak characteristics including height, width, area, and intervals
- **Batch processing** for analyzing multiple files with consistent parameters
- **Data export** in various formats for further analysis or reporting
- **Customizable user interface** with light and dark themes for comfort during extended use

### Applications

- Scientific signal analysis
- Experimental data processing
- Time-series analysis
- Particle detection systems
- Spectroscopy data analysis
- Chromatography peak detection
- Sensor data analysis

## Getting Started

### System Requirements

- Windows, macOS, or Linux operating system
- 4GB RAM minimum (8GB recommended for large datasets)
- 1GB free disk space
- 1920x1080 screen resolution (minimum 1366x768)

### Installation

1. Download the installer from the official website
2. Run the installer and follow the on-screen instructions
3. Launch the application from the Start menu or desktop shortcut

### First Launch

Upon first launch, the application will display a blank interface with control panels on the left and an empty plot area on the right. Follow these steps to begin:

1. Click the "Browse" button to load a data file
2. Use the "Plot Raw Data" button to visualize the loaded data
3. Adjust processing parameters in the control panel
4. Click "Start Analysis" to begin processing

## User Interface

The user interface is divided into several key areas:

### Control Panel (Left Side)

Contains all controls for data loading, processing parameters, and analysis options:

- **File Selection** - Load and manage data files
- **Signal Processing** - Parameters for filtering and normalizing data
- **Peak Detection** - Controls for peak detection algorithms
- **Protocol Information** - Metadata about the current analysis session
- **Action Buttons** - Execute various analysis and visualization steps

### Visualization Area (Right Side)

Displays plots and results in a tabbed interface:

- **Raw Data** - Original unprocessed signal
- **Filtered Signal** - Processed signal after filtering
- **Peaks View** - Detected peaks with highlighting
- **Analysis** - Statistical plots and measurements of peak characteristics

### Menu Bar

Provides access to additional functions:

- **File Menu** - Open, save, and export options
- **Edit Menu** - Application settings and preferences
- **View Menu** - Visualization options and theme settings
- **Analysis Menu** - Advanced analysis features
- **Help Menu** - Documentation, about information, and support options

### Status Indicators

- **Status Bar** - Shows current application state and operation status
- **Progress Bar** - Visual indication of long-running operations
- **Preview Label** - Contextual information about the current view

## Data Loading

### Supported File Formats

The application supports multiple data file formats:

- CSV (Comma-Separated Values)
- TSV (Tab-Separated Values)
- TXT (Text files with delimited values)
- XLS/XLSX (Excel spreadsheets)
- Custom formats with configurable parameters

### Single File Mode

For analyzing one data file at a time:

1. Set file mode to "Single" in the dropdown menu
2. Click "Browse" to select a data file
3. Configure file format options if needed
4. Click "Load" to import the data

### Batch Mode

For processing multiple files with the same parameters:

1. Set file mode to "Batch" in the dropdown menu
2. Click "Browse" to select multiple data files
3. Set timestamp handling options for batch processing
4. Click "Load" to import all files

### Protocol Information

Enter metadata about your analysis for documentation purposes:

- **Start Time** - When the experiment or data collection began
- **Particle** - Type of particle or sample being analyzed
- **Concentration** - Sample concentration information
- **Stamp** - Unique identifier for the analysis session
- **Laser Power** - Relevant for optical measurements
- **Setup** - Equipment or experimental setup used
- **Notes** - Additional information about the data or analysis

## Signal Processing

### Normalization

The normalization factor adjusts the amplitude scaling of your signal:

- **Higher values** increase the signal amplitude
- **Lower values** decrease the signal amplitude
- **Value of 1.0** maintains the original signal scale
- **Auto-detection** can be used to suggest an optimal value

### Filtering Options

#### Cutoff Frequency

Controls the low-pass filter applied to your data:

- **Higher values** allow more high-frequency components
- **Lower values** create smoother signals with less noise
- **Value of 0** enables automatic cutoff frequency detection
- **Typical range:** 1-50 Hz depending on your signal characteristics

#### Filter Implementation

The application uses a Butterworth filter implementation:

- **Order:** Determines the steepness of the frequency rolloff
- **Passband:** Frequencies below the cutoff are preserved
- **Stopband:** Frequencies above the cutoff are attenuated

### Advanced Processing

Additional signal processing options include:

- **Savitzky-Golay smoothing** for noise reduction while preserving peak shapes
- **Baseline correction** to remove drift and offset
- **Signal decimation** for handling extremely large datasets efficiently
- **Zero-phase filtering** to prevent peak position shifts during filtering

## Peak Detection

### Detection Parameters

#### Height Threshold

Minimum amplitude required for a peak to be detected:

- **Higher values** detect only the most prominent peaks
- **Lower values** find more peaks including smaller ones
- **Auto-threshold** calculates an optimal value based on signal statistics
- **Typical range:** 10-30% of the maximum signal amplitude

#### Minimum Distance

Minimum separation between adjacent peaks in data points:

- **Higher values** prevent detection of closely spaced peaks
- **Lower values** allow detection of peaks that are close together
- **Recommended:** Set to approximately the expected peak width
- **Typical range:** 20-100 data points depending on sampling rate

#### Relative Height

Controls the determination of peak width:

- **Value range:** 0.0 to 1.0
- **Higher values** (closer to 1.0) result in narrower peak width measurements
- **Lower values** result in wider peak width measurements
- **Default of 0.85** works well for most applications

#### Width Range

Filtering peaks by width (in data points):

- **Format:** Minimum,Maximum (e.g., "1,200")
- **Empty value** disables width filtering
- **Narrow range** selects peaks of specific width characteristics
- **Wide range** includes various peak shapes

### Peak Detection Algorithm

The application employs a sophisticated multi-stage peak detection approach:

1. **Preprocessing** - Signal filtering and normalization
2. **Initial detection** - Finding all potential peak candidates
3. **Parameter filtering** - Applying height, distance, and width constraints
4. **Validation** - Statistical checks to reject false positives
5. **Measurement** - Calculating peak characteristics

### Advanced Detection Options

- **Adaptive thresholding** adjusts detection sensitivity based on local signal statistics
- **Peak prominence** consideration for distinguishing true peaks from noise
- **Shape analysis** for rejecting asymmetric or irregular peaks
- **Baseline detection** for accurate peak height and area measurement

## Analysis Workflows

### Basic Analysis Workflow

1. **Load Data** - Import your signal data file
2. **Plot Raw Data** - Visualize the unprocessed signal
3. **Start Analysis** - Apply filtering with current parameters
4. **Run Peak Detection** - Detect peaks in the filtered signal
5. **View Results** - Examine detected peaks and measurements

### Advanced Analysis Workflow

1. **Load Data** - Import your signal data file
2. **Plot Raw Data** - Visualize the unprocessed signal
3. **Adjust Parameters** - Fine-tune processing parameters
4. **Calculate Auto-Threshold** - Determine optimal detection threshold
5. **Calculate Auto-Cutoff** - Find optimal filter cutoff frequency
6. **Start Analysis** - Apply filtering with optimized parameters
7. **Run Peak Detection** - Detect peaks in the filtered signal
8. **Plot Filtered Peaks** - View detected peaks with highlighting
9. **Calculate Peak Areas** - Measure area under each peak
10. **Plot Data** - Generate statistical visualizations
11. **Export Results** - Save measurements and plots

### Batch Processing Workflow

1. **Switch to Batch Mode** - Select "Batch" from the file mode dropdown
2. **Load Multiple Files** - Import several data files
3. **Set Common Parameters** - Configure processing settings for all files
4. **Process Sequentially** - Analyze each file with the same parameters
5. **Compare Results** - View statistics across the batch
6. **Export Batch Results** - Save comprehensive analysis for all files

## Visualization Options

### Plot Types

#### Raw Data Plot

- **X-axis:** Time or sample index
- **Y-axis:** Signal amplitude
- **Features:** Zoomable, pannable view of original unprocessed data
- **Interactions:** Click and drag to zoom, right-click to reset view

#### Filtered Signal Plot

- **X-axis:** Time or sample index
- **Y-axis:** Processed signal amplitude
- **Features:** Overlay of original and filtered signals for comparison
- **Interactions:** Toggle signal visibility, adjust view range

#### Peaks View

- **X-axis:** Time or sample index
- **Y-axis:** Signal amplitude
- **Features:** Detected peaks highlighted with markers
- **Interactions:** Click peaks to view details, navigate between peak groups

#### Analysis Plots

- **Peak Height Distribution** - Histogram of peak heights
- **Peak Width Distribution** - Histogram of peak widths
- **Peak Area Distribution** - Histogram of integrated peak areas
- **Interval Analysis** - Distribution of time between adjacent peaks
- **Scatter Plots** - Relationships between peak characteristics
- **Time Series Analysis** - Changes in peak properties over time

### Visualization Controls

- **Zoom** - Magnify specific regions of interest
- **Pan** - Navigate through different parts of the signal
- **Rectangle Selection** - Select and analyze specific regions
- **Axis Scaling** - Linear or logarithmic axis options
- **Color Maps** - Different color schemes for better visualization
- **Grid Lines** - Toggle visibility of reference grid
- **Legend** - Identify different data series and markers

### Theme Options

The application supports light and dark themes:

- **Light Theme** - Bright background with dark text (default)
- **Dark Theme** - Dark background with light text (reduces eye strain)
- **Theme Switching** - Toggle between themes via the View menu or shortcut

## Data Export

### Export Formats

#### CSV Export

Save detected peak information to CSV format:

- **Peak indices** - Sample position of each peak
- **Peak timestamps** - Time position of each peak
- **Peak heights** - Maximum amplitude of each peak
- **Peak widths** - Width of each peak at the specified relative height
- **Peak areas** - Integrated area under each peak
- **Intervals** - Time between adjacent peaks

#### Plot Export

Save visualizations as image files:

- **PNG** - Lossless raster format (default)
- **JPG** - Compressed raster format
- **PDF** - Vector format for high-quality printing
- **SVG** - Editable vector format
- **Resolution options** - Standard or high DPI

#### Data Export

Export processed signals and measurements:

- **Raw signal** - Original unprocessed data
- **Filtered signal** - Processed data after filtering
- **Peak measurements** - All calculated peak characteristics
- **Analysis results** - Statistical summaries and distributions

### Export Options

- **Include protocol information** in exports
- **Automatic file naming** based on analysis parameters
- **Custom export templates** for specific reporting needs
- **Batch export** for multiple files or plots at once

## Batch Processing

### Batch Configuration

- **File Selection** - Choose multiple files for sequential processing
- **Parameter Sharing** - Apply the same settings to all files
- **Timestamp Handling** - Options for time synchronization between files
- **Process Controls** - Run, pause, or stop batch processing

### Batch Results

- **Individual Results** - Separate analysis for each file
- **Aggregate Statistics** - Combined metrics across all files
- **Comparative Visualization** - Overlay or grid plots for comparison
- **Summary Reports** - Statistical overview of the entire batch

### Batch Export

- **Consolidated Report** - Combined results from all files
- **Individual Reports** - Separate detailed analysis for each file
- **Comparative Metrics** - Statistical comparison between files
- **Bulk Export** - Save all plots and data in one operation

## Advanced Features

### Region of Interest Analysis

1. Use the rectangle selector tool to define a region
2. Right-click and select "Analyze Region"
3. View specialized analysis for just the selected area
4. Compare statistics with the overall signal

### Custom Filter Design

1. Access the Filter Designer from the Analysis menu
2. Design custom filter responses (Butterworth, Chebyshev, etc.)
3. Preview the filter effect on your signal
4. Apply the custom filter to your analysis

### Statistical Analysis

- **Descriptive Statistics** - Mean, median, standard deviation of peak properties
- **Distribution Fitting** - Fit statistical distributions to peak characteristics
- **Outlier Detection** - Identify anomalous peaks in the dataset
- **Trend Analysis** - Detect changes in peak properties over time

### Performance Optimization

- **Data Decimation** - Reduce points for faster processing of large datasets
- **Multi-threading** - Utilize multiple CPU cores for parallel processing
- **Memory Management** - Options for handling very large files efficiently
- **Processing Profiles** - Save and load optimization settings for different data types

## Troubleshooting

### Common Issues

#### No Peaks Detected

Possible causes and solutions:

- **Threshold too high** - Reduce the height threshold parameter
- **Filter cutoff too low** - Increase the cutoff frequency
- **Minimum distance too large** - Reduce the minimum distance parameter
- **Width range too narrow** - Expand or remove the width filtering range
- **Signal quality issues** - Check your data for acquisition problems

#### Program Slowdown

Optimization recommendations:

- **Reduce data points** - Use decimation options for very large files
- **Close other applications** - Free up system resources
- **Use batch mode carefully** - Process fewer files at once
- **Consider hardware upgrades** - More RAM or faster CPU

#### Unexpected Results

Troubleshooting steps:

1. Reset parameters to defaults and start again
2. Try auto-calculation for threshold and cutoff
3. Examine raw data plot for data quality issues
4. Check file format and data structure
5. Consult the technical reference for parameter recommendations

### Error Messages

- **"No filtered signal available"** - Run the Start Analysis function first
- **"Invalid parameter value"** - Check input values are within allowed ranges
- **"File format error"** - Ensure your data file matches expected format
- **"Memory error"** - Reduce file size or close other applications
- **"Processing error"** - Check the log file for detailed information

### Getting Help

- **In-app documentation** - Available through the Help menu
- **Tooltips** - Hover over parameters for contextual information
- **Log files** - Check `performance.log` for detailed error information
- **Support contact** - Email support@peakanalysistool.com for assistance

## Technical Reference

### Algorithm Details

#### Signal Filtering

The application implements a Butterworth low-pass filter:

```
H(s) = 1 / (1 + (s/ωc)^2n)
```

Where:
- H(s) is the transfer function
- ωc is the cutoff frequency
- n is the filter order (default: 4)

#### Peak Detection

The peak finding algorithm uses:

```
peaks, _ = find_peaks(
    filtered_signal,
    height=height_threshold,
    distance=min_distance,
    width=width_range,
    rel_height=relative_height
)
```

#### Peak Area Calculation

Peak areas are calculated by numerical integration:

```
peak_area = ∫ signal(t) dt
```

Where integration occurs over the full width at the specified relative height.

### Parameter Guidelines

| Parameter | Typical Range | Effect of Increase | Effect of Decrease |
|-----------|---------------|-------------------|-------------------|
| Normalization | 0.1 - 10.0 | Amplifies signal | Attenuates signal |
| Cutoff Frequency | 1 - 2000 Hz | More detail, more noise | Smoother signal, less detail |
| Height Threshold | 10 - 50 | Fewer peaks detected | More peaks detected |
| Minimum Distance | 20 - 100 | Fewer closely-spaced peaks | More closely-spaced peaks |
| Relative Height | 0.5 - 0.95 | Narrower peak width | Wider peak width |

### Performance Considerations

- **File Size Impact**: Processing time scales approximately linearly with file size
- **Memory Usage**: Peak memory usage is approximately 5x the size of the raw data file
- **CPU Utilization**: Multi-threading is used for operations on large datasets
- **GPU Acceleration**: Not currently implemented

### File Format Specifications

CSV files should conform to the following structure:

```
timestamp,value
1,0.123
2,0.456
...
```

Where:
- timestamp is the time in 0.1 milisec
- value is the signal amplitude

Alternative formats require appropriate column mapping in the import dialog.

---

*This documentation is for Peak Analysis Tool version 1.0. Last updated: May 2024.* 