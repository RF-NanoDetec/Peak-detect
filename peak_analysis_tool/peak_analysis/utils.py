"""
Utility functions and decorators for the Peak Analysis Tool.
Provides common functionality used across the application.
"""

import os
import time
import logging
import traceback
import psutil
from functools import wraps
from typing import Callable, Any, Dict, Optional
from datetime import datetime
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_logging(
    log_file: str = 'peak_analysis.log',
    level: int = logging.INFO
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def profile_function(func: Callable) -> Callable:
    """
    Decorator to profile function execution time and memory usage.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get start time and memory
        start_time = time.time()
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Calculate execution time and memory usage
            execution_time = time.time() - start_time
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_diff = memory_after - memory_before
            
            # Log performance metrics
            logger.info(f"Function '{func.__name__}' performance:")
            logger.info(f"  Execution time: {execution_time:.2f} seconds")
            logger.info(f"  Memory change: {memory_diff:+.1f} MB")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
    return wrapper

def cache_result(cache_dir: str = '.cache') -> Callable:
    """
    Decorator to cache function results to disk.
    
    Args:
        cache_dir: Directory for cache files
        
    Returns:
        Wrapped function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory
            os.makedirs(cache_dir, exist_ok=True)
            
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
            
            # Check if cached result exists
            if os.path.exists(cache_file):
                logger.info(f"Loading cached result for {func.__name__}")
                with np.load(cache_file) as data:
                    return data['result']
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            np.savez_compressed(cache_file, result=result)
            
            return result
        return wrapper
    return decorator

def validate_input(validation_func: Callable) -> Callable:
    """
    Decorator to validate function inputs.
    
    Args:
        validation_func: Function to validate inputs
        
    Returns:
        Wrapped function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not validation_func(*args, **kwargs):
                raise ValueError(f"Invalid input for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        execution_time = time.time() - self.start_time
        logger.info(f"{self.name} completed in {execution_time:.2f} seconds")

class DataSaver:
    """Utility class for saving and loading data."""
    
    def __init__(self, base_dir: str = 'data'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(
        self,
        data: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Save analysis results to file.
        
        Args:
            data: Dictionary of results to save
            filename: Optional filename, defaults to timestamp
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"results_{timestamp}.json"
        
        filepath = self.base_dir / filename
        
        # Convert numpy arrays to lists
        processed_data = self._process_data_for_saving(data)
        
        with open(filepath, 'w') as f:
            json.dump(processed_data, f, indent=4)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """
        Load saved results from file.
        
        Args:
            filename: Name of file to load
            
        Returns:
            Dictionary of loaded results
        """
        filepath = self.base_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays where appropriate
        processed_data = self._process_data_for_loading(data)
        
        logger.info(f"Results loaded from {filepath}")
        return processed_data
    
    def _process_data_for_saving(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for JSON serialization."""
        processed = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                processed[key] = {
                    'type': 'ndarray',
                    'data': value.tolist(),
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, dict):
                processed[key] = self._process_data_for_saving(value)
            else:
                processed[key] = value
        return processed
    
    def _process_data_for_loading(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process loaded data to restore numpy arrays."""
        processed = {}
        for key, value in data.items():
            if isinstance(value, dict) and 'type' in value:
                if value['type'] == 'ndarray':
                    processed[key] = np.array(
                        value['data'],
                        dtype=value['dtype']
                    )
            elif isinstance(value, dict):
                processed[key] = self._process_data_for_loading(value)
            else:
                processed[key] = value
        return processed

def create_timestamp() -> str:
    """Create formatted timestamp string."""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def ensure_directory(path: str) -> str:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Absolute path to directory
    """
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    return path

def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level=logging.DEBUG)
    
    # Example function with profiling
    @profile_function
    def example_function(size: int):
        return np.random.random(size)
    
    # Example function with caching
    @cache_result()
    def cached_function(x: int):
        return x ** 2
    
    # Example with timer
    with Timer("Array operation"):
        result = example_function(1000000)
    
    # Example with data saver
    saver = DataSaver()
    data = {
        'array': np.array([1, 2, 3]),
        'value': 42
    }
    filepath = saver.save_results(data)
    loaded_data = saver.load_results(os.path.basename(filepath))
    
    print("Loaded data:", loaded_data)
