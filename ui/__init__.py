"""
UI Module for the Peak Analysis Tool
===================================

This module provides UI components, styling, and interaction utilities for the application.
It implements a consistent look and feel across the application while supporting
both light and dark themes.

Components:
    ThemeManager: Manages application-wide theming with support for light and dark modes
    EnhancedTooltip: Improved tooltips with better styling and positioning
    StatusIndicator: Visual indicator for operation status (success, error, warning, info)

Functions:
    create_tooltip: Helper function to easily create tooltips for any widget

Usage:
    >>> from ui import ThemeManager, create_tooltip, StatusIndicator
    >>> theme_manager = ThemeManager(theme_name='dark')
    >>> create_tooltip(button, "Click to analyze data")
    >>> status = StatusIndicator(parent_frame)
    >>> status.set_success("Operation completed successfully")
"""

from ui.theme import ThemeManager
from ui.tooltips import EnhancedTooltip, create_tooltip
from ui.status_indicator import StatusIndicator

__all__ = ['ThemeManager', 'EnhancedTooltip', 'create_tooltip', 'StatusIndicator'] 