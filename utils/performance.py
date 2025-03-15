"""
Performance monitoring utilities for the Peak Analysis Tool.

This module provides functions for profiling, memory monitoring, and
performance optimization.
"""

import os
import time
import psutil
from functools import wraps

def profile_function(func):
    """
    Decorator to profile function execution time and memory usage.
    
    Parameters
    ----------
    func : function
        Function to profile
    
    Returns
    -------
    function
        Wrapped function with profiling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Get start time
        start_time = time.time()
        
        # Run the function
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"ERROR in {func.__name__}: {str(e)} (took {elapsed_time:.2f} seconds)")
            raise
        
        # Get end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
        memory_diff = memory_after - memory_before
        
        # Print profiling information
        print(f"DEBUG: {func.__name__} took {elapsed_time:.2f} seconds and used {memory_diff:.1f} MB of memory")
        
        return result
    
    return wrapper

def get_memory_usage():
    """
    Get current memory usage of the process.
    
    Returns
    -------
    float
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB 