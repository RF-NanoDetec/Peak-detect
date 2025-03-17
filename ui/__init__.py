"""
UI components and utilities for the Peak Analysis Tool.

These modules provide the user interface elements, themes,
and interaction utilities for the application.
"""

# Import main components
from ui.theme import ThemeManager
from ui.status_indicator import StatusIndicator
from ui.components import create_control_panel, create_menu_bar, create_preview_frame
from ui.tooltips import setup_tooltips

# Import new UI utilities
from ui.ui_utils import (
    update_results_summary,
    validate_float,
    update_progress_bar,
    take_screenshot,
    show_error,
    add_tooltip,
    show_documentation,
    show_about_dialog,
    on_file_mode_change,
    with_ui_error_handling,
    ui_action
)

__all__ = [
    'ThemeManager', 
    'StatusIndicator', 
    'create_control_panel', 
    'create_menu_bar', 
    'create_preview_frame', 
    'setup_tooltips',
    'update_results_summary',
    'validate_float',
    'update_progress_bar',
    'take_screenshot',
    'show_error',
    'add_tooltip',
    'show_documentation',
    'show_about_dialog',
    'on_file_mode_change',
    'with_ui_error_handling',
    'ui_action'
] 