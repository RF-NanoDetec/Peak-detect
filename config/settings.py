"""
Application Settings Module
==========================

This module defines all application-wide settings and configuration constants.
It provides a centralized location for adjusting application behavior without
modifying the core code.

Classes:
    Config: Main configuration class with application-wide settings
    Config.Colors: Color constants for UI elements
    Config.Plot: Plotting-related configuration
"""

import os

class Config:
    """
    Application-wide configuration settings.
    
    This class serves as a centralized repository for all configuration constants
    used throughout the application. It includes settings for performance,
    appearance, and behavior.
    
    Attributes:
        MAX_WORKERS (int): Maximum number of worker threads for parallel processing,
            defaulting to twice the CPU count.
        MAX_PLOT_POINTS (int): Maximum number of points to display in interactive plots
            to maintain performance.
        PROGRESS_RESET_DELAY (int): Delay in milliseconds before resetting progress bars.
        DEFAULT_WINDOW_SIZE (tuple): Default application window size in pixels (width, height).
        
    Example:
        >>> from config.settings import Config
        >>> num_workers = min(data_size // 1000, Config.MAX_WORKERS)
        >>> window = tk.Tk()
        >>> window.geometry(f"{Config.DEFAULT_WINDOW_SIZE[0]}x{Config.DEFAULT_WINDOW_SIZE[1]}")
    """
    MAX_WORKERS = os.cpu_count() * 2
    MAX_PLOT_POINTS = 10000
    PROGRESS_RESET_DELAY = 500  # milliseconds
    DEFAULT_WINDOW_SIZE = (1920, 1080)
    
    class Colors:
        """
        Color constants for UI elements.
        
        This nested class defines standard colors used throughout the application
        for consistent visual feedback based on message type or status.
        
        Attributes:
            ERROR (str): Color for error messages and indicators (red).
            SUCCESS (str): Color for success messages and indicators (green).
            INFO (str): Color for informational messages and indicators (blue).
            WARNING (str): Color for warning messages and indicators (orange).
            
        Example:
            >>> status_label.config(foreground=Config.Colors.SUCCESS)
            >>> error_message.config(foreground=Config.Colors.ERROR)
        """
        ERROR = "red"
        SUCCESS = "green"
        INFO = "blue"
        WARNING = "orange"
    
    class Plot:
        """
        Plot configuration settings.
        
        This nested class defines all settings related to plot appearance and
        behavior, ensuring consistent visualization throughout the application.
        
        Attributes:
            DPI (int): Dots per inch for on-screen display of plots.
            FIGURE_SIZE (tuple): Default figure size in inches (width, height).
            EXPORT_DPI (int): Higher resolution DPI value for exported plots.
            LINE_WIDTH (float): Default line width for plot elements.
            FONT_SIZE (int): Standard font size for plot text elements.
            TITLE_SIZE (int): Font size for plot titles.
            LABEL_SIZE (int): Font size for axis labels.
            
        Example:
            >>> fig = plt.figure(figsize=Config.Plot.FIGURE_SIZE, dpi=Config.Plot.DPI)
            >>> plt.title("Peak Analysis", fontsize=Config.Plot.TITLE_SIZE)
            >>> plt.xlabel("Time (s)", fontsize=Config.Plot.LABEL_SIZE)
        """
        DPI = 100  # Increased from default 100
        FIGURE_SIZE = (12, 8)  # Inches
        EXPORT_DPI = 300  # Higher DPI for exports
        LINE_WIDTH = 0.5
        FONT_SIZE = 10
        TITLE_SIZE = 12
        LABEL_SIZE = 8