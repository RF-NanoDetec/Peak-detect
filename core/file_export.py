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
from tkinter import filedialog
from config.settings import Config
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

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
    Save detected peak information to a CSV file.
    
    Parameters
    ----------
    app : Application
        The main application instance
    """
    if app.filtered_signal is None or not hasattr(app, 'peak_detector') or not hasattr(app, 't_value'):
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
        
        # Get current parameters
        height_lim_factor = app.height_lim.get()
        distance = app.distance.get()
        rel_height = app.rel_height.get()
        width_values = app.width_p.get().strip().split(',')
        
        # Check if peaks are already detected
        if not hasattr(app.peak_detector, 'peaks_indices') or app.peak_detector.peaks_indices is None:
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
        
        # Calculate areas if not already calculated
        if not hasattr(app.peak_detector, 'peaks_properties') or 'areas' not in app.peak_detector.peaks_properties:
            peak_areas, start_times, end_times = app.peak_detector.calculate_peak_areas(app.filtered_signal)
        else:
            peak_areas = app.peak_detector.peaks_properties['areas']
            start_times = app.peak_detector.peaks_properties['start_indices']
            end_times = app.peak_detector.peaks_properties['end_indices']
        
        # Get peak properties
        peak_indices = app.peak_detector.peaks_indices
        
        # Check if peaks were detected
        if peak_indices is None or len(peak_indices) == 0:
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
        
        # Calculate intervals if there are at least 2 peaks
        if len(peak_times) >= 2:
            intervals = [0]  # First peak has no interval
            intervals.extend(peak_times[1:] - peak_times[:-1])
            df['Interval (s)'] = intervals
        
        # Ask user for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Peak Information"
        )
        
        if file_path:
            # Export to CSV
            df.to_csv(file_path, index=False)
            
            # Update status
            app.status_indicator.set_state('success')
            app.status_indicator.set_text(f"Peak information saved to {os.path.basename(file_path)}")
            
            # Update preview label
            app.preview_label.config(
                text=f"Peak information saved to {file_path}", 
                foreground=app.theme_manager.get_color('success')
            )
            
            return file_path
        else:
            # Save cancelled
            app.status_indicator.set_state('idle')
            app.status_indicator.set_text("Save cancelled")
            return None
            
    except Exception as e:
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