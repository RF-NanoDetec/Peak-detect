# Peak Analysis Tool User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Practical Examples](#practical-examples)
3. [Getting Started](#getting-started)
4. [User Interface](#user-interface)
5. [Data Loading](#data-loading)
6. [Signal Processing](#signal-processing)
7. [Peak Detection](#peak-detection)
8. [Analysis Workflows](#analysis-workflows)
9. [Visualization Options](#visualization-options)
10. [Data Export](#data-export)
11. [Batch Processing](#batch-processing)
12. [Advanced Features](#advanced-features)
13. [Troubleshooting](#troubleshooting)
14. [Technical Reference](#technical-reference)

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

## Practical Examples

### Example 1: Basic Peak Detection

Let's walk through a simple peak detection analysis:

1. **Load Your Data**
   - Click "Browse" and select your data file
   - The file should contain two columns: timestamp and signal value
   - Example data format:
     ```
     timestamp,value
     0.0001,0.123
     0.0002,0.456
     ...
     ```

2. **Visualize Raw Data**
   - Click "Plot Raw Data" to see your signal
   - Use the zoom tool to examine regions of interest
   - Look for any obvious noise or artifacts

3. **Configure Basic Parameters**
   - Set normalization to 1.0 (default)
   - Enable filtering (checkbox)
   - Set cutoff frequency to 0 (auto-detect)
   - Set height threshold to 20 (default)
   - Set minimum distance to 5 (default)
   - Set relative height to 0.8 (default)
   - Set width range to "0.1,50" (default)

4. **Run Analysis**
   - Click "Start Analysis" to apply filtering
   - Click "Run Peak Detection" to find peaks
   - Review the detected peaks in the visualization

5. **Review Results**
   - Check the peak markers on the plot
   - Look at the peak statistics in the results panel
   - Export results if needed

### Example 2: Advanced Peak Analysis

For more complex signals, follow these steps:

1. **Data Preparation**
   - Load your data file
   - Plot raw data to inspect signal quality
   - Note any baseline drift or noise

2. **Optimize Parameters**
   - Click "Calculate Auto-Threshold" for optimal peak detection
   - Click "Calculate Auto-Cutoff" for optimal filtering
   - Adjust parameters based on results:
     - If too many peaks: increase height threshold
     - If peaks too close: increase minimum distance
     - If noise remains: decrease cutoff frequency

3. **Fine-tune Detection**
   - Use the rectangle selector to focus on specific regions
   - Adjust parameters while watching real-time updates
   - Save parameter sets for future use

4. **Advanced Analysis**
   - Calculate peak areas for quantitative analysis
   - Generate statistical plots
   - Export comprehensive results

### Example 3: Batch Processing

For analyzing multiple files:

1. **Prepare Files**
   - Organize your data files in a folder
   - Ensure consistent naming convention
   - Check file formats match

2. **Configure Batch Mode**
   - Switch to "Batch" mode in the dropdown
   - Select your folder of data files
   - Set common parameters for all files

3. **Process Files**
   - Start batch processing
   - Monitor progress in the status bar
   - Review results as they come in

4. **Analyze Results**
   - Compare statistics across files
   - Generate comparative plots
   - Export batch report

### Common Parameter Combinations

Here are some recommended parameter sets for different types of signals:

1. **Clean Signals**
   ```
   Normalization: 1.0
   Cutoff Frequency: 0 (auto)
   Height Threshold: 20
   Minimum Distance: 5
   Relative Height: 0.8
   Width Range: 0.1,50
   ```

2. **Noisy Signals**
   ```
   Normalization: 1.0
   Cutoff Frequency: 10 Hz
   Height Threshold: 30
   Minimum Distance: 10
   Relative Height: 0.7
   Width Range: 0.2,100
   ```

3. **Closely Spaced Peaks**
   ```
   Normalization: 1.0
   Cutoff Frequency: 0 (auto)
   Height Threshold: 15
   Minimum Distance: 3
   Relative Height: 0.9
   Width Range: 0.1,30
   ```

### Tips for Best Results

1. **Data Quality**
   - Ensure clean data acquisition
   - Remove obvious artifacts before analysis
   - Check for baseline drift

2. **Parameter Selection**
   - Start with default parameters
   - Use auto-calculation features
   - Make small adjustments
   - Document successful parameter sets

3. **Visualization**
   - Use zoom for detailed inspection
   - Compare raw and filtered signals
   - Check peak markers carefully
   - Use different plot types for verification

4. **Export and Documentation**
   - Save parameter sets
   - Export results in multiple formats
   - Document analysis steps
   - Keep protocol information updated

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

### Double Peak Analysis

The Peak Analysis Tool includes a specialized feature for detecting and analyzing double peaks, which are common in certain types of signals. This feature helps identify and characterize pairs of peaks that occur close together in time.

#### Enabling Double Peak Analysis

1. **Activate the Feature**
   - Select "Double Peak" mode from the analysis options
   - The interface will update to show double peak parameters
   - Additional visualization tabs will appear

2. **Configure Parameters**
   - **Minimum Distance**: Time between peaks (default: 0.001s)
   - **Maximum Distance**: Maximum allowed separation (default: 0.010s)
   - **Minimum Amplitude Ratio**: Ratio of second to first peak (default: 0.1)
   - **Maximum Amplitude Ratio**: Maximum allowed ratio (default: 5.0)
   - **Minimum Width Ratio**: Ratio of peak widths (default: 0.1)
   - **Maximum Width Ratio**: Maximum allowed ratio (default: 5.0)

#### Analysis Workflow

1. **Initial Detection**
   - Run standard peak detection first
   - The tool will identify potential double peak pairs
   - Results appear in the "Double Peak Selection" tab

2. **Review and Selection**
   - Examine each potential double peak pair
   - Use the selection interface to mark valid pairs
   - Navigate through results using next/previous buttons

3. **Grid View Analysis**
   - Switch to the "Double Peak Grid" tab
   - View multiple pairs simultaneously
   - Compare characteristics across pairs

4. **Statistical Analysis**
   - Review distribution of peak separations
   - Analyze amplitude and width ratios
   - Export detailed measurements

#### Common Applications

1. **Particle Detection**
   - Identify double peaks from particle pairs
   - Measure time between particle arrivals
   - Analyze particle size relationships

2. **Spectroscopy**
   - Detect split peaks in spectral data
   - Analyze peak broadening effects
   - Study peak overlap phenomena

3. **Sensor Analysis**
   - Identify double-trigger events
   - Analyze sensor response characteristics
   - Study timing relationships

#### Best Practices

1. **Parameter Selection**
   - Start with default values
   - Adjust based on your signal characteristics
   - Consider physical meaning of parameters

2. **Validation**
   - Manually verify detected pairs
   - Check for false positives
   - Document selection criteria

3. **Analysis**
   - Use grid view for pattern recognition
   - Compare statistics across datasets
   - Export results for further analysis

4. **Troubleshooting**
   - If too many pairs detected:
     - Increase minimum distance
     - Adjust amplitude ratio limits
     - Tighten width ratio constraints
   - If too few pairs detected:
     - Decrease minimum distance
     - Relax amplitude ratio limits
     - Widen width ratio constraints

#### Example Parameter Sets

1. **Standard Analysis**
   ```
   Minimum Distance: 0.001s
   Maximum Distance: 0.010s
   Min Amplitude Ratio: 0.1
   Max Amplitude Ratio: 5.0
   Min Width Ratio: 0.1
   Max Width Ratio: 5.0
   ```

2. **Tight Constraints**
   ```
   Minimum Distance: 0.0005s
   Maximum Distance: 0.005s
   Min Amplitude Ratio: 0.5
   Max Amplitude Ratio: 2.0
   Min Width Ratio: 0.5
   Max Width Ratio: 2.0
   ```

3. **Relaxed Constraints**
   ```
   Minimum Distance: 0.002s
   Maximum Distance: 0.020s
   Min Amplitude Ratio: 0.05
   Max Amplitude Ratio: 10.0
   Min Width Ratio: 0.05
   Max Width Ratio: 10.0
   ```

#### Export Options

1. **Data Export**
   - Save double peak pairs to CSV
   - Include timing and ratio measurements
   - Export selection criteria

2. **Visualization Export**
   - Save grid view as image
   - Export individual pair plots
   - Create summary statistics plots

3. **Report Generation**
   - Include protocol information
   - Add statistical summaries
   - Document analysis parameters

### Data Validation and Quality Control

The Peak Analysis Tool includes comprehensive data validation and quality control features to ensure reliable analysis results. This section describes the tools and procedures for validating your data and maintaining quality control throughout the analysis process.

#### Data Validation Features

1. **File Format Validation**
   - Checks file structure and format
   - Validates data types and ranges
   - Detects missing or corrupt values
   - Reports format issues

2. **Data Quality Checks**
   - Identifies outliers and anomalies
   - Detects baseline drift
   - Checks for signal saturation
   - Validates time series continuity

3. **Parameter Validation**
   - Ensures parameters are within valid ranges
   - Checks for logical consistency
   - Validates relationships between parameters
   - Prevents invalid combinations

#### Quality Control Procedures

1. **Pre-Analysis Checks**
   - Verify data completeness
   - Check for systematic errors
   - Validate calibration
   - Review acquisition settings

2. **During Analysis**
   - Monitor processing quality
   - Track parameter stability
   - Validate intermediate results
   - Check for processing artifacts

3. **Post-Analysis Validation**
   - Review peak detection quality
   - Validate statistical measures
   - Check result consistency
   - Verify export integrity

#### Quality Metrics

1. **Signal Quality Indicators**
   ```
   Signal-to-Noise Ratio: > 10
   Baseline Stability: < 5% drift
   Sampling Rate: Consistent
   Data Continuity: No gaps
   ```

2. **Processing Quality Metrics**
   ```
   Peak Detection Confidence: > 90%
   False Positive Rate: < 5%
   Processing Stability: Consistent
   Memory Usage: Within limits
   ```

3. **Result Quality Measures**
   ```
   Statistical Significance: p < 0.05
   Measurement Precision: < 1%
   Result Reproducibility: > 95%
   Export Completeness: 100%
   ```

#### Validation Tools

1. **Data Inspection**
   - Interactive data viewer
   - Statistical summaries
   - Distribution plots
   - Time series analysis

2. **Quality Reports**
   - Comprehensive quality metrics
   - Processing statistics
   - Validation results
   - Issue summaries

3. **Automated Checks**
   - Format validation
   - Range checking
   - Consistency verification
   - Completeness assessment

#### Best Practices

1. **Data Collection**
   - Follow standardized procedures
   - Document acquisition settings
   - Maintain calibration records
   - Use quality control samples

2. **Analysis Process**
   - Validate at each step
   - Document validation results
   - Track quality metrics
   - Address issues promptly

3. **Result Verification**
   - Cross-validate results
   - Compare with standards
   - Review statistical measures
   - Document verification

#### Troubleshooting Quality Issues

1. **Data Quality Problems**
   - Check acquisition settings
   - Review calibration
   - Inspect for artifacts
   - Validate file integrity

2. **Processing Issues**
   - Verify parameters
   - Check processing steps
   - Review intermediate results
   - Validate algorithms

3. **Result Quality Concerns**
   - Review detection criteria
   - Check statistical validity
   - Verify measurements
   - Validate exports

#### Quality Control Documentation

1. **Required Records**
   - Data acquisition logs
   - Calibration records
   - Processing parameters
   - Validation results

2. **Quality Reports**
   - Daily quality metrics
   - Processing statistics
   - Issue tracking
   - Resolution records

3. **Audit Trail**
   - Parameter changes
   - Processing steps
   - Validation checks
   - Result modifications

#### Quality Assurance Procedures

1. **Regular Checks**
   - Daily system validation
   - Weekly quality review
   - Monthly performance assessment
   - Quarterly comprehensive audit

2. **Documentation Requirements**
   - Standard operating procedures
   - Quality control protocols
   - Validation checklists
   - Issue resolution records

3. **Training Requirements**
   - Quality control procedures
   - Validation methods
   - Documentation standards
   - Issue resolution

### Performance Optimization and Memory Management

The Peak Analysis Tool includes several features to optimize performance and manage memory usage when working with large datasets. This section provides guidelines for efficient data handling and processing.

#### Data Size Considerations

1. **File Size Guidelines**
   - Small files (< 1MB): No special handling needed
   - Medium files (1MB - 100MB): Use standard processing
   - Large files (> 100MB): Enable optimization features
   - Very large files (> 1GB): Use batch processing

2. **Memory Usage**
   - Monitor memory usage in the status bar
   - Typical memory usage: 5x file size
   - Peak memory during processing: 10x file size
   - Warning threshold: 80% of available RAM

#### Optimization Features

1. **Data Decimation**
   - Automatically reduces data points for plotting
   - Preserves visual features while reducing memory usage
   - Configurable maximum points (default: 10,000)
   - Full resolution maintained for analysis

2. **Multi-threading**
   - Parallel processing for batch operations
   - Configurable number of worker threads
   - Automatic thread management
   - Progress tracking per thread

3. **Memory Management**
   - Automatic garbage collection
   - Temporary file usage for large datasets
   - Progressive loading of data
   - Cache management for plots

#### Performance Settings

1. **Plot Optimization**
   ```
   Maximum Plot Points: 10,000
   Decimation Method: Adaptive
   Cache Size: 100MB
   ```

2. **Processing Settings**
   ```
   Thread Count: Auto-detect
   Batch Size: 1MB
   Cache Enabled: Yes
   ```

3. **Memory Limits**
   ```
   Warning Threshold: 80% RAM
   Critical Threshold: 90% RAM
   Cache Limit: 500MB
   ```

#### Best Practices

1. **File Organization**
   - Keep files in a dedicated analysis folder
   - Use consistent naming conventions
   - Maintain backup copies
   - Archive processed files

2. **Processing Strategy**
   - Process smaller chunks for large files
   - Use batch mode for multiple files
   - Enable decimation for visualization
   - Monitor memory usage

3. **System Resources**
   - Close other memory-intensive applications
   - Monitor system resources
   - Use SSD storage when possible
   - Maintain adequate free disk space

#### Troubleshooting Performance Issues

1. **Slow Processing**
   - Enable data decimation
   - Reduce plot update frequency
   - Use batch processing
   - Check system resources

2. **High Memory Usage**
   - Close unused plot tabs
   - Clear plot cache
   - Process smaller chunks
   - Use temporary files

3. **System Crashes**
   - Check available memory
   - Reduce batch size
   - Enable auto-save
   - Monitor system logs

#### Advanced Optimization

1. **Custom Decimation**
   ```python
   # Example of custom decimation settings
   decimation_factor = calculate_decimation_factor(
       data_size=file_size,
       max_points=10000,
       sampling_rate=sampling_rate
   )
   ```

2. **Memory Profiling**
   - Monitor memory usage patterns
   - Identify memory leaks
   - Optimize data structures
   - Profile processing steps

3. **Batch Processing Optimization**
   - Configure optimal batch size
   - Set appropriate thread count
   - Manage temporary files
   - Monitor progress

#### Performance Monitoring

1. **Status Indicators**
   - Memory usage percentage
   - Processing speed
   - Thread utilization
   - Cache hit rate

2. **Logging**
   - Performance metrics
   - Memory usage patterns
   - Processing times
   - Error tracking

3. **Reporting**
   - Generate performance reports
   - Track optimization effects
   - Monitor system resources
   - Document issues

#### Recommendations for Different Scenarios

1. **Small Datasets (< 1MB)**
   - Use standard processing
   - Enable all features
   - No optimization needed

2. **Medium Datasets (1MB - 100MB)**
   - Enable basic decimation
   - Use standard batch processing
   - Monitor memory usage

3. **Large Datasets (> 100MB)**
   - Enable all optimizations
   - Use advanced batch processing
   - Implement custom decimation
   - Monitor system resources

4. **Very Large Datasets (> 1GB)**
   - Use specialized processing
   - Implement chunked processing
   - Enable temporary file usage
   - Monitor system stability

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