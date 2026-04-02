"""
File handling utilities for Peak Analysis Tool

This module contains functions for loading and managing data files.
"""

import logging

import numpy as np
import pandas as pd

from core.performance import profile_function

# Configure logging
logger = logging.getLogger(__name__)


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
    logger.debug(f"Loading file {index + 1}: {file}")
    logger.debug(
        f"Using time resolution factor: {time_resolution} (converts raw time values to seconds)"
    )

    try:
        # Determine file type based on extension
        if file.lower().endswith((".xls", ".xlsx")):
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
                delimiter="\t",
                usecols=[0, 1],
                dtype={0: np.float32, 1: np.float32},
                engine="c",
                skip_blank_lines=True,
                comment="#",
            )

        # Get column names and handle missing headers efficiently
        cols = df.columns.tolist()

        # Use direct dictionary access for faster column renaming
        if len(cols) >= 2:
            # Only strip whitespace if needed
            if any(c != c.strip() for c in cols):
                df.columns = [c.strip() for c in cols]

            # Most efficient way to get column names
            if "Time - Plot 0" in df.columns and "Amplitude - Plot 0" in df.columns:
                time_col = "Time - Plot 0"
                amp_col = "Amplitude - Plot 0"
            else:
                # Rename columns without creating a new DataFrame
                df.columns = ["Time - Plot 0", "Amplitude - Plot 0"]
                time_col = "Time - Plot 0"
                amp_col = "Amplitude - Plot 0"
        else:
            raise ValueError(f"File {file} doesn't have at least 2 columns")

        # Extract numpy arrays directly for better performance
        # Use float64 for time to preserve sub-millisecond precision on long spans
        time_values = np.ascontiguousarray(df[time_col].values, dtype=np.float64)
        time_in_seconds = time_values * time_resolution

        return {
            "time": time_in_seconds,  # Now in seconds
            "time_raw": time_values,  # Keep raw values for reference if needed
            "amplitude": np.ascontiguousarray(df[amp_col].values),
            "index": index,
            "time_resolution": time_resolution,  # Store the resolution used
        }
    except Exception as e:
        logger.error(f"Error loading file {file}: {str(e)}")
        raise
