"""
Environment Configuration for the Peak Analysis Tool
===================================================

This module contains environment-specific settings and functions that help
with managing the application's runtime environment. It handles:

1. Platform detection and adaptation
2. Directory structure initialization
3. Resource path resolution (for bundled executables)
4. Logging configuration
5. System information reporting

Constants:
    APP_VERSION: Current version of the application
    DEBUG_MODE: Flag to enable debug features
    IS_WINDOWS, IS_MAC, IS_LINUX: Platform flags
    BASE_DIR, RESOURCE_DIR, LOG_DIR, TEMP_DIR: Directory paths

Functions:
    resource_path: Resolve resource paths for both development and bundled execution
    setup_logging: Configure application logging
    is_frozen: Check if application is running as a bundled executable
    get_app_info: Get formatted system and application information
"""

import os
import sys
import logging
from pathlib import Path
import platform

# Application version
APP_VERSION = "1.0.0"

# Debug mode - controls additional debugging features
DEBUG_MODE = os.environ.get("PEAK_ANALYSIS_DEBUG", "0") == "1"

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

def resource_path(relative_path):
    """
    Get the absolute path to a resource file.
    
    This function resolves resource paths correctly in both development mode
    and when running as a bundled application (PyInstaller).
    
    Parameters:
        relative_path (str): Path relative to the application's base directory
            
    Returns:
        str: Absolute path to the resource
        
    Example:
        >>> image_path = resource_path("resources/images/logo.png")
        >>> config_path = resource_path("config/settings.ini")
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = getattr(sys, '_MEIPASS', str(BASE_DIR))
        return os.path.join(base_path, relative_path)
    except Exception:
        return os.path.join(os.path.abspath("."), relative_path)

def setup_logging(level=None):
    """
    Configure the application's logging system.
    
    This function sets up handlers for both console and file logging,
    establishes formatting, and sets appropriate log levels based on
    the application mode.
    
    Parameters:
        level (int, optional): Logging level to use (e.g., logging.DEBUG).
            If None, uses logging.DEBUG in debug mode and logging.INFO otherwise.
            
    Returns:
        logging.Logger: Configured root logger instance
        
    Example:
        >>> logger = setup_logging(logging.DEBUG)
        >>> logger.info("Application started")
        >>> logger.error("An error occurred", exc_info=True)
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
    Determine if the application is running as a bundled executable.
    
    This function checks if the application is running from a bundled executable
    created with tools like PyInstaller.
    
    Returns:
        bool: True if running as a bundled executable, False if running in development mode
        
    Example:
        >>> if is_frozen():
        >>>     print("Running from bundled executable")
        >>> else:
        >>>     print("Running in development mode")
    """
    return hasattr(sys, 'frozen') and hasattr(sys, '_MEIPASS')

def get_app_info():
    """
    Get formatted information about the application and system environment.
    
    This function collects and formats various pieces of information about
    the runtime environment, including Python version, OS details, and
    application mode.
    
    Returns:
        str: Multi-line string with formatted application and system information
        
    Example:
        >>> info = get_app_info()
        >>> print(info)
        Python: 3.9.1 (v3.9.1:1e5d33e, Dec 7 2020, 17:08:44)
        Platform: Windows-10-10.0.19041-SP0
        Mode: Development
    """
    # First collect the info in a dictionary
    info_dict = {
        "version": APP_VERSION,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "frozen": is_frozen(),
        "debug_mode": DEBUG_MODE,
        "base_dir": str(BASE_DIR),
    }
    
    # Format into a multi-line string
    mode = "Production" if not DEBUG_MODE else "Development"
    exe_mode = "Bundled Executable" if is_frozen() else "Development Mode"
    
    info_str = (
        f"Peak Analysis Tool v{APP_VERSION}\n"
        f"Python: {platform.python_version()} ({sys.version})\n"
        f"Platform: {platform.platform()}\n"
        f"Mode: {mode}\n"
        f"Execution: {exe_mode}\n"
        f"Base Directory: {BASE_DIR}"
    )
    
    return info_str

# Print environment info in debug mode
if DEBUG_MODE:
    logger.debug(f"App Info:\n{get_app_info()}")
    logger.debug(f"Resource Directory: {RESOURCE_DIR}")
    logger.debug(f"Log Directory: {LOG_DIR}") 