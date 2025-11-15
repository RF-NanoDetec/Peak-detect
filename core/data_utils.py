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

def decimate_min_max(x, y, max_points=10000):
    """
    Downsample signal using min-max decimation to preserve visual peaks and valleys.
    
    OPTIMIZATION: This method ensures that both the minimum and maximum values
    in each bin are included, preserving the visual appearance of peaks.
    
    Args:
        x: time array
        y: signal array
        max_points: maximum number of points (will return ~2x this due to min/max pairs)
    
    Returns:
        x_decimated, y_decimated: decimated arrays for plotting
    """
    if len(x) <= max_points:
        return x, y
    
    # Calculate bin size
    n_bins = max_points // 2  # Each bin contributes 2 points (min and max)
    bin_size = len(x) // n_bins
    
    if bin_size <= 1:
        return x, y
    
    # Pre-allocate output arrays
    out_indices = []
    
    # Process each bin
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(y)
        
        if end > start:
            bin_y = y[start:end]
            # Find local min and max indices within bin
            local_min_idx = np.argmin(bin_y)
            local_max_idx = np.argmax(bin_y)
            
            # Add in time order
            if local_min_idx < local_max_idx:
                out_indices.extend([start + local_min_idx, start + local_max_idx])
            else:
                out_indices.extend([start + local_max_idx, start + local_min_idx])
    
    # Remove duplicates while preserving order
    out_indices = sorted(set(out_indices))
    
    return x[out_indices], y[out_indices]


@profile_function
def decimate_for_plot(x, y, max_points=10000):
    """
    Intelligently reduce number of points for plotting while preserving important features.
    
    OPTIMIZED: Uses min-max decimation for better visual preservation.
    
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
        
        # Use min-max decimation for better visual quality
        return decimate_min_max(x, y, max_points)
            
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


def convert_width_ms_to_samples(width_ms, sampling_rate):
    """
    Convert peak width range from milliseconds to samples given sampling rate.

    Parameters
    ----------
    width_ms : str | list | tuple
        Width specification in milliseconds, e.g., "1,200" or [1, 200]
    sampling_rate : float
        Samples per second (Hz)

    Returns
    -------
    list[int]
        [min_width_samples, max_width_samples]
    """
    try:
        if isinstance(width_ms, str):
            values = [v.strip() for v in width_ms.split(',')]
        else:
            values = [str(v) for v in width_ms]

        width_samples = [int(float(v) * sampling_rate / 1000) for v in values]
        return width_samples
    except Exception as e:
        logger.error(f"Error converting width from ms to samples: {e}\n{traceback.format_exc()}")
        # Reasonable default window if conversion fails
        return [1, 200]


def validate_width_ms_string(width_ms):
    """
    Validate width specification string "min,max" in milliseconds.

    Returns (ok: bool, message: str)
    """
    try:
        if isinstance(width_ms, str):
            parts = [p.strip() for p in width_ms.split(',')]
        else:
            parts = [str(v) for v in width_ms]
        if len(parts) != 2:
            return False, "Width must have two numbers: min,max (ms)"
        wmin = float(parts[0])
        wmax = float(parts[1])
        if wmin <= 0 or wmax <= 0:
            return False, "Width values must be > 0 ms"
        if wmin > wmax:
            return False, "Width min must be <= max"
        return True, ""
    except Exception:
        return False, "Invalid width format; use min,max (e.g., 1,200)"


def validate_peak_params(
    prominence_threshold,
    distance,
    rel_height,
    width_ms,
    time_resolution,
    prominence_ratio=None,
):
    """
    Validate common peak detection parameters.

    Returns (ok: bool, message: str)
    """
    # Time resolution
    try:
        if time_resolution is None or float(time_resolution) <= 0:
            return False, "Time resolution must be > 0"
    except Exception:
        return False, "Time resolution is invalid"

    # Prominence threshold
    try:
        if float(prominence_threshold) < 0:
            return False, "Prominence threshold must be >= 0"
    except Exception:
        return False, "Prominence threshold is invalid"

    # Distance (samples)
    try:
        if int(distance) < 0:
            return False, "Minimum distance must be >= 0 samples"
    except Exception:
        return False, "Minimum distance is invalid"

    # rel_height (0..1]
    try:
        rh = float(rel_height)
        if rh <= 0 or rh > 1:
            return False, "Relative height must be in (0, 1]"
    except Exception:
        return False, "Relative height is invalid"

    # prominence_ratio (0..1], optional if None
    if prominence_ratio is not None:
        try:
            pr = float(prominence_ratio)
            if pr <= 0 or pr > 1:
                return False, "Prominence ratio must be in (0, 1]"
        except Exception:
            return False, "Prominence ratio is invalid"

    # width
    ok, msg = validate_width_ms_string(width_ms)
    if not ok:
        return False, msg

    return True, ""

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
            try:
                from ui.ui_utils import update_results_summary_with_ui
                update_results_summary_with_ui(app, preview_text="")
            except Exception:
                pass
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