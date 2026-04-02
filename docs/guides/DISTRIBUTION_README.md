# Peak Analysis Tool

## Quick Start

### Running the Application

Simply double-click **`PeakService.exe`** to start the Peak Analysis Tool.

The application will:
1. Start the backend service
2. Automatically open your web browser to the application
3. Be ready to use!

### Using the Application

#### 1. Load Data
- Click "Load" in the left sidebar
- Upload your data files (.txt, .xls, .xlsx)
- Set the time resolution if needed
- Click "Load Files"

#### 2. Preprocess (Optional)
- Choose a filter type (None, Butterworth, Savitzky-Golay)
- Configure filter parameters
- Click "Apply Filter"
- View before/after comparison

#### 3. Detect Peaks
- Set detection parameters (prominence, distance, width)
- Use "Auto" buttons for automatic parameter calculation
- Click "Detect Peaks"
- View detected peaks on the chart
- Click "View Peaks" to inspect individual peaks

#### 4. Analyze Results
- View comprehensive statistics
- Explore different chart types (time series, histograms, scatter plots)
- Export results as needed

#### 5. Double Peak Analysis (Optional)
- Set distance and ratio constraints
- Click "Analyze Double Peaks"
- View pairs that meet your criteria

#### 6. Export
- Export peak information to CSV
- Export double peak data
- Save plots as images (PNG, SVG, PDF)

---

## System Requirements

- Windows 10 or later
- Modern web browser (Chrome, Firefox, Edge, Safari)
- Minimum 4GB RAM recommended
- Display resolution 1280x720 or higher

---

## Troubleshooting

### Application Won't Start

**Issue**: Double-clicking the EXE does nothing  
**Solution**: Check if a firewall or antivirus is blocking the application. Try running as administrator.

**Issue**: "Port already in use" error  
**Solution**: Another program is using port 8765. Close other applications or restart your computer.

### Browser Doesn't Open

**Issue**: The service starts but browser doesn't open  
**Solution**: Manually open your browser and navigate to: `http://127.0.0.1:8765`

### Application is Slow

**Issue**: Processing takes a long time  
**Solution**: 
- This is normal for large datasets
- Close other applications to free up memory
- Consider filtering your data before processing

**Issue**: Charts take time to load  
**Solution**: The application automatically optimizes large datasets. First load may be slower.

---

## Data Privacy

- All data processing happens locally on your computer
- No data is sent to any external servers
- No internet connection is required (except for initial browser opening)

---

## Support

For technical support or to report issues:
- Check the full documentation at: `http://127.0.0.1:8765` (when running)
- Review the troubleshooting section above
- Contact your system administrator

---

## Version Information

**Peak Analysis Tool v2.0**  
Modern web-based interface for peak detection and analysis.

---

*For advanced usage and development information, see CUTOVER_GUIDE.md*






