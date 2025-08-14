"""
Data utility functions for the Peak Analysis Tool.

This module contains utility functions for manipulating, processing, 
and transforming data for the peak analysis application.
"""

import numpy as np
import traceback
import logging
from core.performance import profile_function
from scipy.signal import find_peaks, peak_widths
import pandas as pd
# Import the optimized find_nearest function
from core.peak_analysis_utils import find_nearest as optimized_find_nearest

# Configure logging
logger = logging.getLogger(__name__)

@profile_function
def decimate_for_plot(x, y, max_points=10000):
    """
    Intelligently reduce number of points for plotting while preserving important features
    
    Args:
        x: time array
        y: signal array
        max_points: maximum number of points to plot
    
    Returns:
        x_decimated, y_decimated: decimated arrays for plotting
    """
    try:
        if len(x) <= max_points:
            return x, y
        
        # More efficient decimation algorithm for very large datasets
        n_points = len(x)
        
        # Use numpy operations instead of loops for better performance
        # Calculate decimation factor
        stride = max(1, n_points // max_points)
        
        # For extremely large datasets, use a more aggressive approach
        if stride > 50:
            # Initialize mask - avoid Python loops entirely
            mask = np.zeros(n_points, dtype=bool)
            
            # Include regularly spaced points - efficient slicing
            mask[::stride] = True
            
            # Find peaks efficiently using vectorized operations
            # Use a simplified peak finding for speed - just look for local maxima
            # This is much faster than scipy.signal.find_peaks for this purpose
            if n_points > 3:  # Need at least 3 points for this method
                # Create shifted arrays for comparison
                y_left = np.empty_like(y)
                y_left[0] = -np.inf
                y_left[1:] = y[:-1]
                
                y_right = np.empty_like(y)
                y_right[-1] = -np.inf
                y_right[:-1] = y[1:]
                
                # Find points that are larger than both neighbors (local maxima)
                peak_mask = (y > y_left) & (y > y_right)
                
                # Include all peaks in the decimated data
                mask = mask | peak_mask
                
            # Apply mask for final decimation
            return x[mask], y[mask]
        else:
            # Simple stride-based decimation for less aggressive reduction
            return x[::stride], y[::stride]
            
    except Exception as e:
        logger.error(f"Error decimating data: {str(e)}\n{traceback.format_exc()}")
        # If anything goes wrong, return original data
        return x, y

def get_width_range(width_str):
    """
    Convert width string to list of integers
    
    Parameters
    ----------
    width_str : str
        String containing comma-separated width min and max values
        
    Returns
    -------
    list
        List of [min_width, max_width]
    """
    try:
        # Split the string and convert to integers
        width_min, width_max = map(int, width_str.split(','))
        
        # Return as list
        return [width_min, width_max]
        
    except Exception as e:
        logger.error(f"Error parsing width range: {str(e)}\n{traceback.format_exc()}")
        # Return default values if there's an error
        return [1, 200]

def timestamps_to_seconds(timestamp_str):
    """
    Convert timestamps from "MM:SS" format to seconds.
    
    Parameters
    ----------
    timestamp_str : str
        Timestamp in "MM:SS" format
        
    Returns
    -------
    float
        Time in seconds
    """
    try:
        if ":" in timestamp_str:
            minutes, seconds = map(float, timestamp_str.split(":"))
            return minutes * 60 + seconds
        else:
            # If no colon, assume it's already in seconds
            return float(timestamp_str)
    except Exception as e:
        logger.error(f"Error converting timestamp to seconds: {str(e)}\n{traceback.format_exc()}")
        return 0.0

def find_nearest(array, value):
    """
    Find the index of the nearest value in an array.
    This is a wrapper around the optimized version from peak_analysis_utils.
    
    Parameters
    ----------
    array : array-like
        Array to search
    value : float
        Value to find
        
    Returns
    -------
    int
        Index of the nearest value
    """
    try:
        # Ensure array is in numpy format
        array = np.asarray(array)
        # Call the optimized version
        return optimized_find_nearest(array, value)
    except Exception as e:
        logger.error(f"Error finding nearest value: {str(e)}\n{traceback.format_exc()}")
        # Fallback to simple implementation if the optimized version fails
        idx = (np.abs(array - value)).argmin()
        return idx

def reset_application_state(app):
    """
    Reset all application variables and plots to initial state
    
    Parameters
    ----------
    app : Application
        The main application instance
    """
    try:
        # Reset data variables
        app.data = None
        app.t_value = None
        app.x_value = None
        app.filtered_signal = None
        app.segment_offset = 0
        
        # Reset peak detector if it exists
        if hasattr(app, 'peak_detector'):
            app.peak_detector.reset()
            
        # Clear loaded files list
        app.loaded_files = []

        # Reset variables to default values
        app.start_time.set("0:00")
        app.height_lim.set(20)
        app.distance.set(30)
        app.rel_height.set(0.85)
        app.width_p.set("1,200")
        app.cutoff_value.set(0)
        app.filter_enabled.set(True)  # Reset filter toggle to enabled
        
        # Clear file path
        app.file_path.set("")
        
        # Reset protocol variables
        if hasattr(app, 'protocol_start_time'):
            app.protocol_start_time.set("")
        if hasattr(app, 'protocol_particle'):
            app.protocol_particle.set("")
        if hasattr(app, 'protocol_sample_number'):
            app.protocol_sample_number.set("")
        if hasattr(app, 'protocol_concentration'):
            app.protocol_concentration.set("")
        if hasattr(app, 'protocol_stamp'):
            app.protocol_stamp.set("")
        if hasattr(app, 'protocol_laser_power'):
            app.protocol_laser_power.set("")
        if hasattr(app, 'protocol_setup'):
            app.protocol_setup.set("")
        if hasattr(app, 'protocol_notes'):
            app.protocol_notes.set("")
        if hasattr(app, 'protocol_files'):
            app.protocol_files.set("")

        # Clear results summary
        if hasattr(app, 'results_summary'):
            app.update_results_summary(preview_text="")
        else:
            # Fallback to just updating the preview label
            app.preview_label.config(text="", foreground=app.theme_manager.get_color('text'))

        # Clear all tabs except Welcome tab
        for tab in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab, "text") != "Welcome":
                app.plot_tab_control.forget(tab)

        # Reset tab figures dictionary
        app.tab_figures.clear()

        # Reset preview label
        app.preview_label.config(text="Application state reset", foreground="blue")

        # Reset progress bar
        app.update_progress_bar(0)

        # Update status
        app.status_indicator.set_state('success')
        app.status_indicator.set_text("Application reset successfully")

    except Exception as e:
        app.data = None
        app.t_value = None
        app.x_value = None
        
        logger.error(f"Error resetting application state: {str(e)}\n{traceback.format_exc()}")
        
        # Update status
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error resetting application state") 

@profile_function
def reset_application_state_with_ui(app):
    """
    Reset all application variables and plots to initial state with integrated UI handling.
    
    This function directly integrates UI updates, eliminating the need
    for a separate wrapper method in the Application class.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
        
    Returns
    -------
    bool
        True if reset was successful, False otherwise
    """
    try:
        # UI pre-processing: Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Resetting application...")
        app.update_idletasks()
        
        # Call the core function to reset the application state
        reset_application_state(app)
        
        # Update welcome screen analyze button state if it exists
        if hasattr(app, 'update_welcome_analyze_button'):
            app.update_welcome_analyze_button()
        
        # UI post-processing: Update status with success
        app.status_indicator.set_state('success')
        app.status_indicator.set_text("Application reset successfully")
        
        # Update preview label
        app.preview_label.config(
            text="Application state reset successfully",
            foreground=app.theme_manager.get_color('success')
        )
        
        return True
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error resetting application state: {str(e)}\n{traceback.format_exc()}")
        
        # Update UI with error info
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error resetting application state")
        
        # Show error dialog
        from ui.ui_utils import show_error
        show_error(app, "Error resetting application state", e)
        
        # Update preview label
        app.preview_label.config(
            text=f"Error resetting application state: {str(e)}",
            foreground=app.theme_manager.get_color('error')
        )
        
        return False 