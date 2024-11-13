# Peak Analysis Tool

## Overview
The Peak Analysis Tool is a sophisticated GUI application designed for analyzing time-series data with a focus on peak detection and analysis. It's particularly useful for processing single-particle tracking data, fluorescence measurements, and other time-series signals.

## Features
- **Data Loading**
  - Single file or batch processing
  - Support for time-stamped data
  - Automatic file sorting and concatenation
  - Protocol information tracking

- **Signal Processing**
  - Butterworth lowpass filtering
  - Automatic cutoff frequency optimization
  - Real-time signal visualization
  - Customizable filtering parameters

- **Peak Detection**
  - Advanced peak detection algorithm
  - Adjustable detection parameters
  - Width and prominence filtering
  - Area calculation

- **Analysis Tools**
  - Time-resolved analysis
  - Peak property correlations
  - Statistical summaries
  - Throughput analysis

- **Visualization**
  - Interactive plots
  - Multiple visualization modes
  - Customizable plot layouts
  - Export capabilities

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages:
  ```bash
  pip install numpy pandas matplotlib scipy tkinter
  ```

### Setup
1. Clone the repository:
   ```bash
   git clone [repository-url]
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python Peak_analyse_Script_Lucjan_batch_v4.py
   ```

## Usage

### Basic Workflow
1. **Load Data**
   - Select single file or batch mode
   - Choose data file(s)
   - Enter protocol information

2. **Process Signal**
   - Adjust filtering parameters
   - Run signal analysis
   - Verify filtered signal

3. **Detect Peaks**
   - Set detection parameters
   - Run peak detection
   - Review detected peaks

4. **Analyze Results**
   - View time-resolved analysis
   - Check peak correlations
   - Export results

### Parameters

#### Signal Processing
- **Normalization Factor**: Scales signal amplitude (default: 1.0)
- **Big Counts**: Threshold for significant peaks (default: 100)
- **Cutoff Value**: Lowpass filter frequency in Hz (0 for auto)

#### Peak Detection
- **Height Limit**: Minimum peak height threshold
- **Distance**: Minimum separation between peaks
- **Relative Height**: Height fraction for width calculation
- **Width Range**: Expected peak width range (min,max)

### Data Format
- Input files should be tab-separated (.txt)
- Required columns:
  - Time (labeled as 'Time - Plot 0')
  - Amplitude (labeled as 'Amplitude - Plot 0')

## Output Files
- Filtered signal plots (.png)
- Peak detection results (.csv)
- Analysis summaries (.csv)
- Correlation plots (.png)

## Troubleshooting

### Common Issues
1. **File Loading Errors**
   - Check file format
   - Verify column names
   - Ensure file permissions

2. **Peak Detection Issues**
   - Adjust height threshold
   - Modify width range
   - Check filter settings

3. **Performance Problems**
   - Reduce file size
   - Close other applications
   - Check available memory

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License
MIT License

Copyright (c) 2024 Lucjan Grzegorzewski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors
- Lucjan Grzegorzewski  
- Silas

## Acknowledgments


## Version History
- v4.0: Current version with batch processing


## Contact
lgrzegor@physnet.uni-hamburg.de