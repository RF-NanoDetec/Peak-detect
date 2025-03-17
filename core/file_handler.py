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

from core.performance import profile_function, get_memory_usage
from core.peak_analysis_utils import timestamps_array_to_seconds

# Configure logging
logger = logging.getLogger(__name__)

# Add a function to get memory usage directly from the performance module
def get_memory_usage_wrapper():
    """Wrapper around get_memory_usage to use directly in this module"""
    return get_memory_usage()

@profile_function
def load_single_file(file, timestamps=None, index=0):
    """
    Helper function to load a single file
    Args:
        file (str): Path to the file to load
        timestamps (list, optional): List of timestamps for batch mode
        index (int, optional): Index of the file in the batch

    Returns:
        dict: Dictionary containing time, amplitude and index data
    """
    print(f"Loading file {index+1}: {file}")

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
        return {
            'time': np.ascontiguousarray(df[time_col].values),
            'amplitude': np.ascontiguousarray(df[amp_col].values),
            'index': index
        }
    except Exception as e:
        print(f"Error loading file {file}: {str(e)}")
        raise

@profile_function
def browse_files(app):
    """
    Browse and load file(s) based on current mode
    
    Args:
        app: Application instance
        
    Returns:
        tuple: (t_value, x_value, data, loaded_files) containing the loaded data
    """
    print(f"Memory before loading: {get_memory_usage_wrapper():.2f} MB")
    print(f"Current file mode: {app.file_mode.get()}")
    
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
                executor.submit(load_single_file, file, timestamps, i): i 
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
            time_data = result['time']
            amplitude_data = result['amplitude']
            n_points = len(time_data)
            
            # Apply time offset directly during copy
            if app.file_mode.get() == "batch" and timestamps and i > 0:
                try:
                    # Calculate time values
                    start_time = timestamps_array_to_seconds([timestamps[0]], timestamps[0])[0]*1e4
                    current_time = timestamps_array_to_seconds([timestamps[i]], timestamps[0])[0]*1e4
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
def browse_files_with_ui(app):
    """
    Browse and load file(s) with integrated UI handling.
    
    This function directly integrates UI updates, eliminating the need
    for a separate wrapper method in the Application class.
    
    Parameters
    ----------
    app : Application
        Application instance with UI elements and state
        
    Returns
    -------
    tuple or None
        (t_value, x_value, data, loaded_files) containing the loaded data,
        or None if no files were loaded or an error occurred
    """
    try:
        # Reset application state before loading new file
        from core.data_utils import reset_application_state
        reset_application_state(app)
        
        # UI pre-processing: Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Loading files...")
        app.update_idletasks()
        
        # Reset progress bar
        app.update_progress_bar(0)
        
        # Use the core function to load files
        result = browse_files(app)
        
        if not result:
            app.status_indicator.set_state('idle')
            app.status_indicator.set_text("No files selected")
            return None
            
        # Unpack results
        t_value, x_value, data, loaded_files = result
        
        # Return early if no data was loaded
        if t_value is None:
            return None
            
        # Store data in application
        app.t_value = t_value
        app.x_value = x_value
        app.data = data
        app.loaded_files = loaded_files
        
        # Show success message with number of files loaded
        num_files = len(loaded_files) if loaded_files else 0
        file_text = "file" if num_files == 1 else "files"
        
        # Update UI with success
        app.status_indicator.set_state('success')
        app.status_indicator.set_text(f"Loaded {num_files} {file_text} successfully")
        
        app.preview_label.config(
            text=f"Successfully loaded {num_files} {file_text}",
            foreground=app.theme_manager.get_color('success')
        )
        
        return result
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error loading files: {str(e)}\n{traceback.format_exc()}")
        
        # Update UI with error info
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error loading files")
        
        # Show error dialog
        from ui.ui_utils import show_error
        show_error(app, "Error loading files", e)
        
        # Update preview label
        app.preview_label.config(
            text=f"Error loading files: {str(e)}",
            foreground=app.theme_manager.get_color('error')
        )
        
        return None 