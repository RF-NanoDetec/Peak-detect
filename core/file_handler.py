"""
File handling utilities for Peak Analysis Tool

This module contains functions for loading and managing data files.
"""

import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import filedialog, Tcl
import logging
import traceback
import matplotlib.pyplot as plt
from tkinter import messagebox
from functools import wraps
import tkinter as tk

from core.performance import profile_function, get_memory_usage
from core.peak_analysis_utils import timestamps_array_to_seconds

# Configure logging
logger = logging.getLogger(__name__)

# Add a function to get memory usage directly from the performance module
def get_memory_usage_wrapper():
    """Wrapper around get_memory_usage to use directly in this module"""
    return get_memory_usage()

@profile_function
def load_single_file(file, timestamps=None, index=0, time_resolution=1e-4):
    """
    Helper function to load a single file
    Args:
        file (str): Path to the file to load
        timestamps (list, optional): List of timestamps for batch mode
        index (int, optional): Index of the file in the batch
        time_resolution (float, optional): Time resolution factor to convert raw time values to seconds.
                                         Default is 1e-4 (0.1 milliseconds per unit)

    Returns:
        dict: Dictionary containing time, amplitude and index data with time converted to seconds
    """
    print(f"Loading file {index+1}: {file}")
    print(f"Using time resolution factor: {time_resolution} (converts raw time values to seconds)")

    try:
        # Determine file type based on extension
        if file.lower().endswith(('.xls', '.xlsx')):
            # For Excel files, only read necessary columns to save memory
            df = pd.read_excel(file, usecols=[0, 1])
        else:
            # For CSV/TXT files, use more efficient options:
            # - Use engine='c' for faster parsing
            # - Only read the first two columns
            # - Use float32 instead of float64 to reduce memory usage
            # - Skip empty lines and comments
            df = pd.read_csv(
                file, 
                delimiter='\t',
                usecols=[0, 1],
                dtype={0: np.float32, 1: np.float32},
                engine='c', 
                skip_blank_lines=True,
                comment='#'
            )
        
        # Get column names and handle missing headers efficiently
        cols = df.columns.tolist()
        
        # Use direct dictionary access for faster column renaming
        if len(cols) >= 2:
            # Only strip whitespace if needed
            if any(c != c.strip() for c in cols):
                df.columns = [c.strip() for c in cols]
            
            # Most efficient way to get column names
            if 'Time - Plot 0' in df.columns and 'Amplitude - Plot 0' in df.columns:
                time_col = 'Time - Plot 0'
                amp_col = 'Amplitude - Plot 0'
            else:
                # Rename columns without creating a new DataFrame
                df.columns = ['Time - Plot 0', 'Amplitude - Plot 0']
                time_col = 'Time - Plot 0'
                amp_col = 'Amplitude - Plot 0'
        else:
            raise ValueError(f"File {file} doesn't have at least 2 columns")
        
        # Extract numpy arrays directly for better performance
        # and use numpy.ascontiguousarray for faster array operations later
        # Convert time values to seconds using the time_resolution factor
        time_values = np.ascontiguousarray(df[time_col].values)
        time_in_seconds = time_values * time_resolution
        
        return {
            'time': time_in_seconds,  # Now in seconds
            'time_raw': time_values,  # Keep raw values for reference if needed
            'amplitude': np.ascontiguousarray(df[amp_col].values),
            'index': index,
            'time_resolution': time_resolution  # Store the resolution used
        }
    except Exception as e:
        print(f"Error loading file {file}: {str(e)}")
        raise

@profile_function
def browse_files(app, time_resolution=1e-4):
    """
    Browse and load file(s) based on current mode
    
    Args:
        app: Application instance
        time_resolution (float, optional): Time resolution factor to convert raw time values to seconds.
                                        Default is 1e-4 (0.1 milliseconds per unit)
        
    Returns:
        tuple: (t_value, x_value, data, loaded_files) containing the loaded data with time in seconds
    """
    print(f"Memory before loading: {get_memory_usage_wrapper():.2f} MB")
    print(f"Current file mode: {app.file_mode.get()}")
    print(f"Using time resolution: {time_resolution} (converts raw time values to seconds)")
    
    try:
        # Set status indicator to loading
        app.status_indicator.set_state('loading')
        app.status_indicator.set_text("Loading files...")
        app.update_idletasks()
        
        # Reset progress bar
        app.update_progress_bar(0)
        
        files = []
        if app.file_mode.get() == "single":
            files = list(filedialog.askopenfilenames(
                title="Select Data File",
                filetypes=(
                    ("Data files", "*.txt *.xls *.xlsx"),
                    ("All files", "*.*")
                )
            ))
        else:  # batch mode
            folder = filedialog.askdirectory(title="Select Folder with Data Files")
            if folder:
                # Include both text and Excel files
                files = [os.path.join(folder, f) for f in os.listdir(folder) 
                        if f.lower().endswith(('.txt', '.xls', '.xlsx'))]
        
        if not files:
            app.status_indicator.set_state('idle')
            app.status_indicator.set_text("No files selected")
            return None, None, None, []
        
        files = list(Tcl().call('lsort', '-dict', files))
        app.preview_label.config(text="Loading files...", foreground="blue")
        app.update_idletasks()
        
        # Set maximum progress
        app.update_progress_bar(0, len(files))
        
        # Store file names
        loaded_files = [os.path.basename(f) for f in files]
        
        # Get timestamps if in batch mode
        timestamps = []
        if app.file_mode.get() == "batch":
            timestamps = [t.strip() for t in app.batch_timestamps.get().split(',') if t.strip()]
        
        # Use ThreadPoolExecutor for parallel file loading
        results = []
        with ThreadPoolExecutor(max_workers=min(len(files), os.cpu_count() * 2)) as executor:
            future_to_file = {
                executor.submit(load_single_file, file, timestamps, i, time_resolution): i 
                for i, file in enumerate(files)
            }
            
            for future in as_completed(future_to_file):
                i = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    app.update_progress_bar(len(results))
                    
                    # Update status message
                    app.status_indicator.set_text(f"Loading files... {len(results)}/{len(files)}")
                    app.update_idletasks()
                except Exception as e:
                    app.status_indicator.set_state('error')
                    app.status_indicator.set_text(f"Error loading file {i+1}")
                    print(f"Error loading file {i}: {str(e)}")
                    raise e

        # Sort results by index to maintain order
        results.sort(key=lambda x: x['index'])
        
        # Process results - More memory-efficient implementation
        # Pre-calculate the total size needed for the arrays
        total_time_points = sum(len(result['time']) for result in results)
        
        # Pre-allocate arrays with the correct size and data type
        combined_times = np.zeros(total_time_points, dtype=np.float32)
        combined_amplitudes = np.zeros(total_time_points, dtype=np.float32)
        
        # Copy data into the pre-allocated arrays
        start_idx = 0
        for i, result in enumerate(results):
            time_data = result['time']  # Already in seconds
            amplitude_data = result['amplitude']
            n_points = len(time_data)
            
            # Apply time offset directly during copy
            if app.file_mode.get() == "batch" and timestamps and i > 0:
                try:
                    # Calculate time values - timestamps are now in seconds
                    start_time = timestamps_array_to_seconds([timestamps[0]], timestamps[0])[0]
                    current_time = timestamps_array_to_seconds([timestamps[i]], timestamps[0])[0]
                    time_offset = current_time
                    combined_times[start_idx:start_idx + n_points] = time_data + time_offset
                except Exception as e:
                    print(f"Error calculating time offset: {str(e)}")
                    raise
            else:
                if i > 0:
                    # Calculate time offset from the end of the previous segment
                    time_offset = combined_times[start_idx - 1] + (time_data[1] - time_data[0])
                    combined_times[start_idx:start_idx + n_points] = time_data + time_offset
                else:
                    # First segment has no offset
                    combined_times[start_idx:start_idx + n_points] = time_data
            
            # Copy amplitude data
            combined_amplitudes[start_idx:start_idx + n_points] = amplitude_data
            
            # Update the start index for the next segment
            start_idx += n_points
        
        # Store the combined arrays
        t_value = combined_times
        x_value = combined_amplitudes
        
        print(f"Total data points after concatenation: {len(t_value)}")  # Debug print
        
        # Create combined DataFrame - more efficient version
        # Only create DataFrame with data that will actually be used
        data = pd.DataFrame({
            'Time - Plot 0': t_value,
            'Amplitude - Plot 0': x_value
        })
        
        # Update GUI
        if len(files) == 1:
            app.file_path.set(files[0])
            app.file_name_label.config(text=os.path.basename(files[0]))
        else:
            first_file = os.path.basename(files[0])
            app.file_path.set(files[0])
            app.file_name_label.config(text=f"{first_file} +{len(files)-1} more")
        
        # Create preview text
        file_order_text = "\n".join([f"{i+1}. {fname}" for i, fname in enumerate(loaded_files)])
        if timestamps:
            file_order_text += "\n\nTimestamps:"
            file_order_text += "\n".join([f"{fname}: {tstamp}" 
                                        for fname, tstamp in zip(loaded_files, timestamps)])
        
        preview_text = (
            f"Successfully loaded {len(files)} files\n"
            f"Total rows: {len(data):,}\n"
            f"Time range: {data['Time - Plot 0'].min():.2f} to {data['Time - Plot 0'].max():.2f}\n"
            f"\nFiles loaded in order:\n{file_order_text}\n"
            f"\nPreview of combined data:\n"
            f"{data.head().to_string(index=False)}"
        )
        
        # Update status
        app.status_indicator.set_state('success')
        app.status_indicator.set_text(f"Loaded {len(files)} files successfully")
        
        app.preview_label.config(text="Files loaded successfully", foreground="green")
        app.update_results_summary(preview_text=preview_text)
        
        return t_value, x_value, data, loaded_files
        
    except Exception as e:
        print(f"Detailed error: {str(e)}")
        traceback.print_exc()
        app.update_progress_bar(0)  # Reset on error
        
        # Update status
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error loading files")
        
        return None, None, None, [] 

# Create a UI-integrated version of browse_files
@profile_function
def browse_files_with_ui(app, time_resolution=1e-4):
    """
    Browse and load file(s) with integrated UI handling.
    
    This function directly integrates UI updates, eliminating the need
    for a separate wrapper method in the Application class.
    
    Parameters
    ----------
    app : Application
        Application instance with UI elements and state
    time_resolution : float, optional
        Time resolution factor to convert raw time values to seconds.
        Default is 1e-4 (0.1 milliseconds per unit)
        
    Returns
    -------
    tuple or None
        (t_value, x_value, data, loaded_files) containing the loaded data,
        or None if no files were loaded or an error occurred
    """
    try:
        result = browse_files(app, time_resolution)
        
        # Extract results
        t_value, x_value, data, loaded_files = result
        
        # Store data in app
        app.t_value = t_value
        app.x_value = x_value
        app.data = data  # This is critical for plot_raw_data to work
        app.loaded_files = loaded_files
        
        # Debugging: check if data was properly set
        if not hasattr(app, 'data') or app.data is None:
            print("WARNING: app.data is not properly set!")
        else:
            print(f"Data successfully set with {len(app.data)} rows")
        
        # Store the resolution used - but don't overwrite the Tkinter variable
        if hasattr(app.time_resolution, 'set'):
            app.time_resolution.set(time_resolution)
        else:
            app.time_resolution = tk.DoubleVar(value=time_resolution)
            
        print(f"Using time resolution: {time_resolution}")
        
        # Set data_loaded flag
        app.data_loaded = bool(data is not None)
        
        # Consolidate memory (reduce memory usage after loading)
        if hasattr(app, 'consolidate_memory'):
            app.consolidate_memory()
        
        # Check if double peak analysis is enabled
        if hasattr(app, 'double_peak_analysis') and app.double_peak_analysis.get() == "1":
            app.status_indicator.set_state('info')
            app.status_indicator.set_text("Double Peak Analysis Mode Active")
        
        return result
        
    except Exception as e:
        app.show_error("Error loading files", e)
        return None, None, None, [] 