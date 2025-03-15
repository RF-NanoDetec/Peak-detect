"""
File handling utilities for the Peak Analysis Tool.

This module provides functions for loading, saving, and managing files.
"""

import os
import numpy as np
import pandas as pd
from tkinter import filedialog, Tcl
from utils.performance import profile_function

@profile_function
def load_file(file_path, timestamps=None, index=0):
    """
    Load a single data file.
    
    Parameters
    ----------
    file_path : str
        Path to the file to load
    timestamps : list, optional
        List of timestamps for batch mode
    index : int, optional
        Index of the file in a batch
        
    Returns
    -------
    dict
        Dictionary containing time, amplitude and index data
    """
    print(f"Loading file {index+1}: {file_path}")

    try:
        # Determine file type based on extension
        if file_path.lower().endswith(('.xls', '.xlsx')):
            # For Excel files, only read necessary columns to save memory
            df = pd.read_excel(file_path, usecols=[0, 1])
        else:
            # For CSV/TXT files, use more efficient options:
            # - Use engine='c' for faster parsing
            # - Only read the first two columns
            # - Use float32 instead of float64 to reduce memory usage
            # - Skip empty lines and comments
            df = pd.read_csv(
                file_path, 
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
            raise ValueError(f"File {file_path} doesn't have at least 2 columns")
        
        # Extract numpy arrays directly for better performance
        # and use numpy.ascontiguousarray for faster array operations later
        return {
            'time': np.ascontiguousarray(df[time_col].values),
            'amplitude': np.ascontiguousarray(df[amp_col].values),
            'index': index
        }
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        raise

def select_files(batch_mode=False):
    """
    Display file selection dialog based on mode.
    
    Parameters
    ----------
    batch_mode : bool
        If True, select a folder, otherwise select file(s)
        
    Returns
    -------
    list
        List of file paths
    """
    files = []
    if not batch_mode:
        files = list(filedialog.askopenfilenames(
            title="Select Data File",
            filetypes=(
                ("Data files", "*.txt *.xls *.xlsx"),
                ("All files", "*.*")
            )
        ))
    else:
        folder = filedialog.askdirectory(title="Select Folder with Data Files")
        if folder:
            # Include both text and Excel files
            files = [os.path.join(folder, f) for f in os.listdir(folder) 
                    if f.lower().endswith(('.txt', '.xls', '.xlsx'))]
    
    # Sort files naturally
    if files:
        files = list(Tcl().call('lsort', '-dict', files))
    
    return files

def save_results_to_csv(dataframe, title="Save Results"):
    """
    Save a DataFrame to a CSV file.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame to save
    title : str, optional
        Title for the file dialog
        
    Returns
    -------
    str or None
        Path where file was saved, or None if cancelled
    """
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        title=title
    )
    
    if file_path:
        dataframe.to_csv(file_path, index=False)
        return file_path
    
    return None

def export_figure(figure, title="Export Figure"):
    """
    Save a matplotlib figure to a file.
    
    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save
    title : str, optional
        Title for the file dialog
        
    Returns
    -------
    str or None
        Path where file was saved, or None if cancelled
    """
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("PDF files", "*.pdf"),
            ("SVG files", "*.svg"),
            ("TIFF files", "*.tiff")
        ],
        title=title
    )
    
    if file_path:
        # Remember original facecolor
        original_facecolor = figure.get_facecolor()
        figure.set_facecolor('white')
        
        # Set axes backgrounds to white
        for ax in figure.get_axes():
            original_ax_facecolor = ax.get_facecolor()
            ax.set_facecolor('white')
        
        # Save with tight layout
        figure.savefig(
            file_path,
            dpi=300,  # High resolution
            bbox_inches='tight',
            pad_inches=0.1,
            facecolor='white',
            edgecolor='none'
        )
        
        # Restore original colors
        figure.set_facecolor(original_facecolor)
        for ax in figure.get_axes():
            ax.set_facecolor(original_ax_facecolor)
            
        return file_path
    
    return None 