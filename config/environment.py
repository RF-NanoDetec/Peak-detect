"""
Environment configuration for the Peak Analysis Tool.

This module contains environment-specific settings and functions that help
with managing the application's runtime environment.
"""

import os
import sys
import logging
from pathlib import Path
import platform

# Application version
APP_VERSION = "1.0.0"

# Platform identification
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Base directories
BASE_DIR = Path(__file__).parent.parent.absolute()
RESOURCE_DIR = BASE_DIR / "resources"
LOG_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Logging configuration
LOG_FILE = LOG_DIR / "app.log"
PERFORMANCE_LOG_FILE = LOG_DIR / "performance.log"

# Debug mode - set to False for production
DEBUG_MODE = os.environ.get("PEAK_ANALYSIS_DEBUG", "False").lower() in ("true", "1", "t")

def resource_path(relative_path):
    """
    Get absolute path to resource, works for development and for PyInstaller
    
    Parameters:
    -----------
    relative_path : str
        Relative path to the resource
        
    Returns:
    --------
    str
        Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = getattr(sys, '_MEIPASS', str(BASE_DIR))
        return os.path.join(base_path, relative_path)
    except Exception:
        return os.path.join(os.path.abspath("."), relative_path)

def setup_logging(level=None):
    """
    Set up logging for the application
    
    Parameters:
    -----------
    level : int, optional
        Logging level (default is DEBUG in debug mode, INFO in production)
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    if level is None:
        level = logging.DEBUG if DEBUG_MODE else logging.INFO
        
    # Create logger
    logger = logging.getLogger("peak_analysis")
    logger.setLevel(level)
    
    # Create file handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# User preferences file
USER_PREFS_FILE = os.path.join(
    os.path.expanduser("~"), 
    ".peak_analysis", 
    "preferences.json"
)

# Ensure user preferences directory exists
os.makedirs(os.path.dirname(USER_PREFS_FILE), exist_ok=True)

def is_frozen():
    """
    Determine if the application is running as a frozen executable
    
    Returns:
    --------
    bool
        True if the application is frozen, False otherwise
    """
    return hasattr(sys, 'frozen') and hasattr(sys, '_MEIPASS')

def get_app_info():
    """
    Get information about the application environment
    
    Returns:
    --------
    dict
        Dictionary containing application information
    """
    return {
        "version": APP_VERSION,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "frozen": is_frozen(),
        "debug_mode": DEBUG_MODE,
        "base_dir": str(BASE_DIR),
    }

# Print environment info in debug mode
if DEBUG_MODE:
    logger.debug(f"App Info: {get_app_info()}")
    logger.debug(f"Resource Directory: {RESOURCE_DIR}")
    logger.debug(f"Log Directory: {LOG_DIR}") 