"""
Configuration package for the Peak Analysis Tool.

This package contains modules for configuration settings
and environment variables used throughout the application.
"""

# Import configuration modules
from . import environment

# Re-export commonly used elements
from .environment import (
    APP_VERSION,
    DEBUG_MODE,
    resource_path,
    setup_logging,
    is_frozen,
    get_app_info,
    logger
)
