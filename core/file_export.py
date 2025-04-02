"""
File export functionality for the Peak Analysis Tool.

This module contains functions for exporting various types of data, including
plots, reports, and CSV files.
"""

import os
import pandas as pd
import traceback
import logging
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
from datetime import datetime
from config.settings import Config
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def create_metadata_header(app):
    """
    Create a metadata header for the exported file.
    
    Parameters
    ----------
    app : Application
        The main application instance
        
    Returns
    -------
    str
        Formatted metadata header string
    """
    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get analysis parameters
    height_lim_factor = app.height_lim.get()
    distance = app.distance.get()
    rel_height = app.rel_height.get()
    width_values = app.width_p.get().strip().split(',')
    time_res = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution
    
    # Get protocol information
    protocol_start_time = app.protocol_start_time.get()
    protocol_particle = app.protocol_particle.get()
    protocol_concentration = app.protocol_concentration.get()
    protocol_stamp = app.protocol_stamp.get()
    protocol_laser_power = app.protocol_laser_power.get()
    protocol_setup = app.protocol_setup.get()
    protocol_notes = app.protocol_notes.get()
    protocol_files = app.protocol_files.get()
    
    # Get filter information
    filter_bandwidth = app.filter_bandwidth.get() if hasattr(app, 'filter_bandwidth') else "Not set"
    
    # Create metadata header
    metadata = [
        f"# Peak Analysis Export",
        f"# Export Date: {current_time}",
        f"#",
        f"# Protocol Information:",
        f"#   - Start Time: {protocol_start_time}",
        f"#   - Particle Type: {protocol_particle}",
        f"#   - Concentration: {protocol_concentration}",
        f"#   - Stamp: {protocol_stamp}",
        f"#   - Laser Power: {protocol_laser_power}",
        f"#   - Setup: {protocol_setup}",
        f"#   - Files: {protocol_files}",
        f"#   - Notes: {protocol_notes}",
        f"#",
        f"# Analysis Parameters:",
        f"#   - Height Limit Factor: {height_lim_factor}",
        f"#   - Distance: {distance}",
        f"#   - Relative Height: {rel_height}",
        f"#   - Width Range: {width_values}",
        f"#   - Time Resolution: {time_res} seconds",
        f"#   - Filter Bandwidth: {filter_bandwidth} Hz",
        f"#",
        f"# Data Columns:",
        f"#   1. Time (s) - Peak occurrence time in seconds",
        f"#   2. Amplitude - Peak height",
        f"#   3. Width (ms) - Peak width in milliseconds",
        f"#   4. Width (samples) - Peak width in samples",
        f"#   5. Area - Area under the peak",
        f"#   6. Start Time (s) - Peak start time",
        f"#   7. End Time (s) - Peak end time",
        f"#   8. Interval (s) - Time between consecutive peaks",
        f"#"
    ]
    
    return "\n".join(metadata)

def get_export_format(app):
    """
    Show dialog to select export format and options.
    
    Parameters
    ----------
    app : Application
        The main application instance
        
    Returns
    -------
    tuple
        (file_format, delimiter, include_metadata)
    """
    from ui.components import create_export_options_dialog
    return create_export_options_dialog(app)

def export_plot(app, figure, default_name="peak_analysis_plot"):
    """
    Export the current plot to a file.
    
    Parameters
    ----------
    app : Application
        The main application instance
    figure : matplotlib.figure.Figure
        The figure to export
    default_name : str, optional
        Default filename for the exported plot
    """
    try:
        # Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Exporting plot...")
        
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPEG files", "*.jpg")
            ],
            title="Export Plot"
        )
        
        if file_path:
            # Export plot with high resolution
            figure.savefig(
                file_path, 
                dpi=Config.Plot.EXPORT_DPI,
                bbox_inches='tight',
                facecolor=figure.get_facecolor()
            )
            
            # Update status
            app.status_indicator.set_state('success')
            app.status_indicator.set_text(f"Plot exported to {os.path.basename(file_path)}")
            
            # Update preview label
            app.preview_label.config(
                text=f"Plot exported to {file_path}", 
                foreground=app.theme_manager.get_color('success')
            )
        else:
            # Export cancelled
            app.status_indicator.set_state('idle')
            app.status_indicator.set_text("Export cancelled")
            
    except Exception as e:
        # Update status
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error exporting plot")
        
        # Log error and show to user
        logger.error(f"Error exporting plot: {str(e)}\n{traceback.format_exc()}")
        app.preview_label.config(
            text=f"Error exporting plot: {str(e)}", 
            foreground=app.theme_manager.get_color('error')
        )

def save_peak_information_to_csv(app):
    """
    Save detected peak information to a file with configurable format.
    
    Parameters
    ----------
    app : Application
        The main application instance
    """
    print("DEBUG: Starting save_peak_information_to_csv")
    print(f"DEBUG: filtered_signal exists: {app.filtered_signal is not None}")
    print(f"DEBUG: peak_detector exists: {hasattr(app, 'peak_detector')}")
    print(f"DEBUG: t_value exists: {hasattr(app, 't_value')}")
    
    if app.filtered_signal is None or not hasattr(app, 'peak_detector') or not hasattr(app, 't_value'):
        print("DEBUG: Missing required data")
        app.preview_label.config(
            text="No peak data available. Please run peak detection first.", 
            foreground=app.theme_manager.get_color('error')
        )
        app.status_indicator.set_state('warning')
        app.status_indicator.set_text("No data available")
        return None

    try:
        # Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Saving peak information...")
        
        print("DEBUG: Getting export format and options")
        # Get export format and options
        file_format, delimiter, include_metadata = get_export_format(app)
        print(f"DEBUG: Export options - format: {file_format}, delimiter: {delimiter}, metadata: {include_metadata}")
        
        # Get current parameters
        height_lim_factor = app.height_lim.get()
        distance = app.distance.get()
        rel_height = app.rel_height.get()
        width_values = app.width_p.get().strip().split(',')
        
        print("DEBUG: Checking if peaks are already detected")
        # Check if peaks are already detected
        if not hasattr(app.peak_detector, 'peaks_indices') or app.peak_detector.peaks_indices is None:
            print("DEBUG: No peaks detected, running peak detection")
            # Get time resolution - handle both Tkinter variable and float value
            time_res = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution
            
            # Detect peaks
            app.peak_detector.detect_peaks(
                app.filtered_signal,
                app.t_value,
                height_lim_factor,
                distance,
                rel_height,
                width_values,
                time_resolution=time_res
            )
        
        print("DEBUG: Checking if areas are calculated")
        # Calculate areas if not already calculated
        if not hasattr(app.peak_detector, 'peaks_properties') or 'areas' not in app.peak_detector.peaks_properties:
            print("DEBUG: Calculating peak areas")
            peak_areas, start_times, end_times = app.peak_detector.calculate_peak_areas(app.filtered_signal)
        else:
            print("DEBUG: Using existing peak areas")
            peak_areas = app.peak_detector.peaks_properties['areas']
            start_times = app.peak_detector.peaks_properties['start_indices']
            end_times = app.peak_detector.peaks_properties['end_indices']
        
        # Get peak properties
        peak_indices = app.peak_detector.peaks_indices
        print(f"DEBUG: Number of peaks found: {len(peak_indices) if peak_indices is not None else 0}")
        
        # Check if peaks were detected
        if peak_indices is None or len(peak_indices) == 0:
            print("DEBUG: No peaks to save")
            app.status_indicator.set_state('warning')
            app.status_indicator.set_text("No peaks detected to save")
            app.preview_label.config(
                text="No peaks detected to save", 
                foreground=app.theme_manager.get_color('warning')
            )
            return None
            
        # Get peak heights and widths
        peak_heights = app.peak_detector.peaks_properties.get('prominences', [])
        peak_widths = app.peak_detector.peaks_properties.get('widths', [])
        
        # Time values at peaks
        peak_times = app.t_value[peak_indices]
        
        # Use time_resolution directly instead of calculating from time differences
        rate = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution  # Time between samples in seconds
        
        # Convert widths from samples to milliseconds
        peak_widths_ms = peak_widths * rate * 1000  # Convert samples to ms
        
        print("DEBUG: Creating DataFrame")
        # Create the DataFrame
        data = {
            'Time (s)': peak_times,
            'Amplitude': peak_heights,
            'Width (ms)': peak_widths_ms,
            'Width (samples)': peak_widths  # Keep original width in samples for reference
        }
        
        # Add areas if available
        if peak_areas is not None and len(peak_areas) == len(peak_times):
            data['Area'] = peak_areas
        
        # Add start and end times if available
        if start_times is not None and end_times is not None:
            if len(start_times) == len(peak_times) and len(end_times) == len(peak_times):
                # Convert indices to time values
                start_time_values = app.t_value[start_times.astype(int)]
                end_time_values = app.t_value[end_times.astype(int)]
                data['Start Time (s)'] = start_time_values
                data['End Time (s)'] = end_time_values
        
        # Create DataFrame
        df = pd.DataFrame(data)
        print(f"DEBUG: DataFrame created with {len(df)} rows")
        
        # Calculate intervals if there are at least 2 peaks
        if len(peak_times) >= 2:
            intervals = [0]  # First peak has no interval
            intervals.extend(peak_times[1:] - peak_times[:-1])
            df['Interval (s)'] = intervals
        
        print("DEBUG: Showing file save dialog")
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{file_format}",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            title="Save Peak Information"
        )
        
        if file_path:
            print(f"DEBUG: Saving to file: {file_path}")
            # Create metadata header if requested
            if include_metadata:
                metadata = create_metadata_header(app)
                with open(file_path, 'w') as f:
                    f.write(metadata)
            
            # Export data
            df.to_csv(file_path, 
                     sep=delimiter, 
                     index=False, 
                     mode='a' if include_metadata else 'w',
                     float_format='%.6f')  # Use 6 decimal places for better precision
            
            # Update status
            app.status_indicator.set_state('success')
            app.status_indicator.set_text(f"Peak information saved to {os.path.basename(file_path)}")
            
            # Update preview label
            app.preview_label.config(
                text=f"Peak information saved to {file_path}", 
                foreground=app.theme_manager.get_color('success')
            )
            
            print("DEBUG: File saved successfully")
            return file_path
        else:
            print("DEBUG: Save cancelled by user")
            # Save cancelled
            app.status_indicator.set_state('idle')
            app.status_indicator.set_text("Save cancelled")
            return None
            
    except Exception as e:
        print(f"DEBUG: Error saving peak information: {str(e)}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        # Update status
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error saving peak information")
        
        # Log error and show to user
        logger.error(f"Error saving peak information: {str(e)}\n{traceback.format_exc()}")
        app.preview_label.config(
            text=f"Error saving peak information: {str(e)}", 
            foreground=app.theme_manager.get_color('error')
        )
        return None 