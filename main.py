"""
Created Nov 17 21:518:48 2024

@author: Lucjan & Silas
"""

# Standard library
import os
import time
import logging
import traceback
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt, peak_widths
from scipy import stats
import seaborn as sns
import psutil
from numba import njit  # Add this import at the top of your file
# Tkinter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Tcl
from tkinter.scrolledtext import ScrolledText
import sys
from PIL import Image, ImageTk

# Local imports
from config.settings import Config
from config import resource_path, APP_VERSION
# Import performance utilities
from core.performance import profile_function, get_memory_usage
# Import UI utilities
from ui import ThemeManager, StatusIndicator, ui_action
from ui.ui_utils import (
    update_results_summary_with_ui,
    validate_float_with_ui,
    update_progress_bar_with_ui,
    take_screenshot_with_ui,
    show_error_with_ui,
    add_tooltip,
    show_documentation_with_ui,
    show_about_dialog_with_ui,
    on_file_mode_change_with_ui
)

# Import core modules
from core.peak_detection import PeakDetector
from core.file_handler import browse_files_with_ui
from core.data_analysis import (
    calculate_peak_areas as calculate_peak_areas_function,
    calculate_peak_intervals,
    calculate_auto_threshold
)
from core.data_utils import (
    decimate_for_plot as decimate_for_plot_function,
    get_width_range as get_width_range_function,
    reset_application_state_with_ui,
    find_nearest,
    timestamps_to_seconds  # For single timestamp conversion
)
from core.peak_analysis_utils import timestamps_array_to_seconds, adjust_lowpass_cutoff, calculate_lowpass_cutoff  # For array of timestamps
from core.file_export import (
    export_plot as export_plot_function,
    save_peak_information_to_csv as save_peak_information_to_csv_function
)

# Import UI modules
from ui.components import create_control_panel, create_menu_bar, create_preview_frame
from ui.theme import ThemeManager
from ui.status_indicator import StatusIndicator

# Import all plotting functions directly
from plotting.raw_data import plot_raw_data as plot_raw_data_function
from plotting.data_processing import start_analysis as start_analysis_function
from plotting.peak_visualization import run_peak_detection as run_peak_detection_function
from plotting.peak_visualization import plot_filtered_peaks as plot_filtered_peaks_function
from plotting.peak_visualization import show_next_peaks as show_next_peaks_function
from plotting.analysis_visualization import plot_data as plot_data_function
from plotting.analysis_visualization import plot_scatter as plot_scatter_function

# Set default seaborn style
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.2)
sns.set_context("notebook", rc={"lines.linewidth": 1.0})


class Application(tk.Tk):
    """
    Main application class for the Peak Analysis Tool.
    
    This class is the central component of the application, responsible for:
    1. Creating and managing the user interface
    2. Coordinating data loading and processing
    3. Managing analysis workflows
    4. Visualizing results with plots and reports
    
    The Application class inherits from tkinter.Tk to create the main window
    and implements a comprehensive set of methods for peak analysis operations.
    
    Attributes:
        theme_manager (ThemeManager): Manages application theming (light/dark)
        detector (PeakDetector): Core peak detection engine
        data_frame (pandas.DataFrame): Loaded and processed data
        raw_data (numpy.ndarray): Original unprocessed signal data
        current_protocol (dict): Information about the current analysis protocol
        
    Example:
        >>> app = Application()
        >>> app.mainloop()  # Start the application
    """
    
    def __init__(self):
        """
        Initialize the application instance and set up the main window.
        
        This method:
        1. Sets up performance logging
        2. Configures the application window
        3. Initializes the theme manager
        4. Creates the user interface
        5. Sets up internal state variables
        """
        super().__init__()
        # Initialize logger before other components
        self.setup_performance_logging()
        
        # Application title and window setup
        self.title("Peak Analysis Tool")
        self.geometry("1920x1080")
        
        # Initialize theme manager with light theme as default
        self.theme_manager = ThemeManager(theme_name='light')
        self.style = self.theme_manager.apply_theme(self)
        
        # Store the StatusIndicator class for use in create_control_panel
        self.status_indicator_class = StatusIndicator
        
        # Initialize high-resolution figure
        self.figure = Figure(
            figsize=Config.Plot.FIGURE_SIZE, 
            dpi=Config.Plot.DPI,
            facecolor=self.theme_manager.get_color('canvas_bg')
        )
        
        # Configure default plot styling
        plt.rcParams['figure.dpi'] = Config.Plot.DPI
        plt.rcParams['savefig.dpi'] = Config.Plot.EXPORT_DPI
        plt.rcParams['lines.linewidth'] = Config.Plot.LINE_WIDTH
        plt.rcParams['font.size'] = Config.Plot.FONT_SIZE
        plt.rcParams['axes.titlesize'] = Config.Plot.TITLE_SIZE
        plt.rcParams['axes.labelsize'] = Config.Plot.LABEL_SIZE

        # Initialize figure and canvas as None
        self.canvas = None

        self.file_path = tk.StringVar()
        self.start_time = tk.StringVar(value="0:00")
        self.height_lim = tk.DoubleVar(value=20)
        self.distance = tk.IntVar(value=30)  # Renamed to Min. Distance Peaks
        self.rel_height = tk.DoubleVar(value=0.85)
        self.width_p = tk.StringVar(value="1,200")
        self.time_resolution = tk.DoubleVar(value=1e-4)  # Time resolution factor (default: 0.1ms)
        self.cutoff_value = tk.DoubleVar(value=0)  # Default 0 means auto-detect
        self.filter_enabled = tk.BooleanVar(value=True)  # Toggle for filtering (True=enabled)
        self.sigma_multiplier = tk.DoubleVar(value=5.0)  # Sigma multiplier for auto threshold detection (1-10)
        self.filtered_signal = None
        self.rect_selector = None

        # Protocol Variables
        self.protocol_start_time = tk.StringVar()
        self.protocol_particle = tk.StringVar()
        self.protocol_concentration = tk.StringVar()
        self.protocol_stamp = tk.StringVar()
        self.protocol_laser_power = tk.StringVar()
        self.protocol_setup = tk.StringVar()
        self.protocol_notes = tk.StringVar()

        # Add file mode selection
        self.file_mode = tk.StringVar(value="single")
        self.batch_timestamps = tk.StringVar()

        # Add loaded files tracking
        self.loaded_files = []
        
        # Add a specific entry for file order in protocol
        self.protocol_files = tk.StringVar()
        
        # Create the menu bar
        self.menu_bar = create_menu_bar(self)

        self.create_widgets()
        self.blank_tab_exists = True  # Track if the blank tab exists

        # Initialize data plot attributes
        self.data_figure = None
        self.data_canvas = None
        self.data_original_xlims = None
        self.data_original_ylims = None

        self.tab_figures = {}  # Dictionary to store figures for each tab
        
        # Initialize the PeakDetector
        self.peak_detector = PeakDetector(logger=self.logger)
        
        # Set icon and window title
        self.iconbitmap(self.get_icon_path())
        self.setup_window_title()
        
    def get_icon_path(self):
        """Get path to icon file, or return empty string if not found"""
        icon_path = os.path.join(os.path.dirname(__file__), "resources", "images", "icon.ico")
        if os.path.exists(icon_path):
            return icon_path
        return ""
        
    def setup_window_title(self):
        """Set up a more descriptive window title"""
        version = "1.0"  # Define your version number
        self.title(f"Peak Analysis Tool v{version} - Signal Processing and Analysis")

    def setup_performance_logging(self):
        """
        Set up performance logging for the application.
        
        This method configures the logging system to record performance metrics,
        execution times, and debugging information during application runtime.
        It creates a dedicated logger that writes to a performance.log file.
        
        The logged information is valuable for:
        - Identifying performance bottlenecks
        - Debugging complex operations
        - Monitoring resource usage
        
        The log format includes timestamps, component names, log levels, and messages,
        making it easier to trace the application's execution flow.
        
        Returns:
            None - The logger is stored as self.logger for use throughout the application
        """
        logging.basicConfig(
            filename='performance.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PeakAnalysis')

    @ui_action(
        processing_message="Calculating threshold...",
        success_message="Threshold calculated successfully",
        error_message="Error calculating threshold"
    )
    def calculate_auto_threshold(self):
        """
        Wrapper around the core.data_analysis.calculate_auto_threshold function.
        This method handles the UI updates and error handling for the threshold calculation.
        """
        if self.filtered_signal is None:
            self.preview_label.config(
                text="No filtered signal available. Please run analysis first.", 
                foreground=self.theme_manager.get_color('error')
            )
            self.status_indicator.set_state('warning')
            self.status_indicator.set_text("No data available")
            return None
            
        # Get the current sigma multiplier value from the slider
        sigma_multiplier = self.sigma_multiplier.get()
        
        # Use the core module function with current sigma value
        suggested_threshold = calculate_auto_threshold(self.filtered_signal, sigma_multiplier=sigma_multiplier)
        
        # Update the UI component
        self.height_lim.set(suggested_threshold)
        
        # Update status with the sigma value used
        self.preview_label.config(
            text=f"Threshold calculated using {sigma_multiplier:.1f}σ = {suggested_threshold:.2f}",
            foreground=self.theme_manager.get_color('success')
        )
        
        # Return the result
        return suggested_threshold

    @ui_action(
        processing_message="Calculating cutoff frequency...",
        success_message="Cutoff frequency calculated successfully",
        error_message="Error calculating cutoff frequency"
    )
    def calculate_auto_cutoff_frequency(self):
        """
        Calculate an appropriate cutoff frequency based on signal characteristics.
        
        This method finds the highest signal value and uses 70% of that value as a 
        threshold for determining the appropriate cutoff frequency. This approach
        ensures that the filter preserves the most important peaks while removing noise.
        """
        if self.x_value is None:
            self.preview_label.config(
                text="No data available. Please load a file first.", 
                foreground=self.theme_manager.get_color('error')
            )
            self.status_indicator.set_state('warning')
            self.status_indicator.set_text("No data available")
            return None
        
        try:
            print("DEBUG: Starting calculate_auto_cutoff_frequency")
            
            # Get time resolution
            try:
                time_res = self.time_resolution.get() if hasattr(self.time_resolution, 'get') else self.time_resolution
                print(f"DEBUG: Successfully retrieved time_resolution: {time_res}")
            except Exception as e:
                print(f"DEBUG: Error getting time_resolution: {str(e)}")
                print(f"DEBUG: Falling back to default value 0.0001")
                time_res = 0.0001  # Default fallback value
            
            # Calculate sampling rate
            fs = 1 / time_res
            print(f"DEBUG: Calculated sampling rate (fs): {fs} Hz from time_res: {time_res}")
            
            # Find the highest signal value and calculate 70% threshold
            signal_max = np.max(self.x_value)
            threshold = signal_max * 0.7  # 70% of max value (30% below max)
            print(f"DEBUG: Maximum signal value: {signal_max}")
            print(f"DEBUG: Using 70% threshold: {threshold}")
            
            # Detect peaks above the 70% threshold to measure their widths
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(self.x_value, height=threshold)
            
            if len(peaks) == 0:
                print("DEBUG: No peaks found above 70% threshold, using default cutoff")
                suggested_cutoff = 10.0  # Default cutoff if no peaks found
            else:
                print(f"DEBUG: Found {len(peaks)} peaks above 70% threshold")
                
                # Use the existing core functions but with our calculated threshold
                # instead of the big_counts and normalization_factor parameters
                from core.peak_analysis_utils import calculate_lowpass_cutoff
                suggested_cutoff = calculate_lowpass_cutoff(
                    self.x_value, fs, threshold, 1.0, time_resolution=time_res
                )
            
            print(f"DEBUG: Calculated cutoff frequency: {suggested_cutoff} Hz")
            
            # Update the cutoff value in GUI
            self.cutoff_value.set(suggested_cutoff)
            
            # Custom success message with the calculated value
            self.preview_label.config(
                text=f"Cutoff frequency set to {suggested_cutoff:.1f} Hz (using 70% of max signal)",
                foreground=self.theme_manager.get_color('success')
            )
            
            return suggested_cutoff
            
        except Exception as e:
            print(f"DEBUG: Exception in calculate_auto_cutoff_frequency: {str(e)}")
            import traceback
            traceback.print_exc()
            self.show_error("Error calculating cutoff frequency", str(e))
            return None

    # Create all the GUI widgets
    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

        # Create left control panel
        create_control_panel(self, main_frame)
        
        # Create preview frame with plot tabs on the right
        create_preview_frame(self, main_frame)

    @ui_action(
        processing_message="Loading documentation...",
        success_message="Documentation displayed",
        error_message="Error showing documentation"
    )
    def show_documentation(self):
        """
        Show application documentation in a separate window.
        This method now uses the integrated UI function from ui.ui_utils.
        """
        return show_documentation_with_ui(self)
    
    @ui_action(
        processing_message="Loading about dialog...",
        success_message="About dialog displayed",
        error_message="Error showing about dialog"
    )
    def show_about_dialog(self):
        """
        Show about dialog with application information.
        This method now uses the integrated UI function from ui.ui_utils.
        """
        return show_about_dialog_with_ui(self)

    @ui_action(
        processing_message="Adding tooltip...",
        success_message="Tooltip added",
        error_message="Error adding tooltip"
    )
    def add_tooltip(self, widget, text):
        """
        Add a tooltip to a widget.
        This is a UI wrapper around the ui.ui_utils.add_tooltip function.
        
        Parameters
        ----------
        widget : tkinter.Widget
            The widget to add the tooltip to
        text : str
            The tooltip text to display
            
        Returns
        -------
        object
            The tooltip object
        """
        return add_tooltip(widget, text)

    @profile_function
    @ui_action(
        processing_message="Loading files...",
        success_message="Files loaded successfully",
        error_message="Error loading files"
    )
    def browse_file(self):
        """
        Browse and load file(s) based on current mode.
        This method now uses the integrated UI function from core.file_handler.
        """
        time_res = self.time_resolution.get()
        print(f"Using time resolution: {time_res}")
        return browse_files_with_ui(self, time_resolution=time_res)

    @ui_action(
        processing_message="Resetting application...",
        success_message="Application state reset",
        error_message="Error resetting application state"
    )
    def reset_application_state(self):
        """
        Reset all application variables and plots to initial state.
        This method now uses the integrated UI function from core.data_utils.
        """
        return reset_application_state_with_ui(self)

    @profile_function
    @ui_action(
        processing_message="Plotting raw data...",
        success_message="Raw data plotted successfully",
        error_message="Error plotting raw data"
    )
    def plot_raw_data(self):
        """
        Plot raw data with optimized rendering.
        This is a UI wrapper around the plotting.raw_data.plot_raw_data function.
        """
        return plot_raw_data_function(self, profiler=profile_function)
        
    @ui_action(
        processing_message="Starting analysis...",
        success_message="Analysis completed successfully",
        error_message="Error during analysis"
    )
    def start_analysis(self):
        """
        Start the analysis pipeline with the current configuration.
        This is a UI wrapper around the analysis pipeline function.
        """
        return start_analysis_function(self, profile_function=profile_function)
        
    @ui_action(
        processing_message="Detecting peaks...",
        success_message="Peak detection completed",
        error_message="Error during peak detection"
    )
    def run_peak_detection(self):
        """
        Execute peak detection algorithm with current parameters.
        This is a UI wrapper around the peak detection pipeline function.
        """
        return run_peak_detection_function(self, profile_function=profile_function)
        
    @ui_action(
        processing_message="Plotting filtered peaks...",
        success_message="Filtered peaks plotted successfully",
        error_message="Error plotting filtered peaks"
    )
    def plot_filtered_peaks(self):
        """
        Plot the peaks detected in the filtered signal.
        This is a UI wrapper around the plotting.peak_visualization function.
        """
        return plot_filtered_peaks_function(self, profile_function=profile_function)
        
    @ui_action(
        processing_message="Navigating to next peaks...",
        success_message="Showing next set of peaks",
        error_message="Error navigating to next peaks"
    )
    def show_next_peaks(self):
        """
        Navigate to the next set of peaks in the visualization.
        This is a UI wrapper around the visualization navigation function.
        """
        return show_next_peaks_function(self, profile_function=profile_function)
        
    @ui_action(
        processing_message="Generating plots...",
        success_message="Data plots generated successfully",
        error_message="Error generating data plots"
    )
    def plot_data(self):
        """
        Generate data plots for analysis results.
        This is a UI wrapper around the plotting.data_processing.plot_data function.
        """
        return plot_data_function(self, profile_function=profile_function)
        
    @ui_action(
        processing_message="Creating scatter plots...",
        success_message="Scatter plots created successfully",
        error_message="Error creating scatter plots"
    )
    def plot_scatter(self):
        """
        Create scatter plots for peak analysis visualization.
        This is a UI wrapper around the plotting.analysis_visualization.plot_scatter function.
        """
        return plot_scatter_function(self, profile_function=profile_function)

    # Function to calculate the areas of detected peaks
    @profile_function
    @ui_action(
        processing_message="Calculating peak areas...",
        success_message="Peak area calculation completed",
        error_message="Error calculating peak areas"
    )
    def calculate_peak_areas(self):
        """
        Calculate the areas of detected peaks in the signal.
        This is a UI wrapper around the core.data_analysis.calculate_peak_areas function.
        """
        if self.filtered_signal is None:
            self.preview_label.config(
                text="Filtered signal not available. Please start the analysis first.", 
                foreground=self.theme_manager.get_color('error')
            )
            self.status_indicator.set_state('warning')
            self.status_indicator.set_text("No data available")
            return None

        # Get current parameters
        height_lim_factor = self.height_lim.get()
        distance = self.distance.get()
        rel_height = self.rel_height.get()
        width_values = self.width_p.get().strip().split(',')
        time_res = self.time_resolution.get()
        
        # Use the core module function
        result = calculate_peak_areas_function(
            self.peak_detector,
            self.filtered_signal,
            self.t_value,
            height_lim_factor,
            distance,
            rel_height,
            width_values,
            time_resolution=time_res
        )
        
        if result:
            peak_area, start, end = result
            
            # Update results summary
            self.update_results_summary(peak_areas=peak_area)
            return peak_area, start, end
        else:
            return None

    # Function to save peak information to CSV
    @ui_action(
        processing_message="Saving peak information...",
        success_message="Peak information saved successfully",
        error_message="Error saving peak information"
    )
    def save_peak_information_to_csv(self):
        """
        Export detected peak information to a CSV file.
        This is a UI wrapper around the core.file_export.save_peak_information_to_csv function.
        
        Returns
        -------
        str or None
            Path where the CSV file was saved, or None if canceled or an error occurred
        """
        return save_peak_information_to_csv_function(self)
            
    @ui_action(
        processing_message="Taking screenshot...",
        success_message="Screenshot taken successfully",
        error_message="Error taking screenshot"
    )
    def take_screenshot(self):
        """
        Capture a screenshot of the current application view.
        This method now uses the integrated UI function from ui.ui_utils.
        
        Returns
        -------
        str or None
            Path where the screenshot was saved, or None if canceled or an error occurred
        """
        return take_screenshot_with_ui(self)

    @ui_action(
        processing_message="Updating file mode...",
        success_message="File mode updated",
        error_message="Error updating file mode"
    )
    def on_file_mode_change(self):
        """
        Handle UI updates when file mode changes (single/batch).
        This method now uses the integrated function from ui.ui_utils.
        """
        return on_file_mode_change_with_ui(self)

    # Add this helper function at the class level
    def show_error(self, title, error):
        """
        Display an error dialog with detailed error information.
        This method now uses the integrated function from ui.ui_utils.
        """
        return show_error_with_ui(self, title, error)

    # Add this method to the Application class
    def get_width_range(self):
        """
        Parse width parameter string into a usable range.
        This is a UI wrapper around the core.data_utils.get_width_range function.
        """
        return get_width_range_function(self.width_p.get())

    # Function to update the results summary text box
    def update_results_summary(self, events=None, max_amp=None, peak_areas=None, peak_intervals=None, preview_text=None):
        """
        Update the results summary text widget with analysis results.
        This method now uses the integrated UI function from ui.ui_utils.
        """
        return update_results_summary_with_ui(self, events, max_amp, peak_areas, peak_intervals, preview_text)

    # Add this validation method to your class
    def validate_float(self, value):
        """
        Validate that an input string is a valid float.
        This method now uses the integrated function from ui.ui_utils.
        """
        return validate_float_with_ui(self, value)

    def update_progress_bar(self, value=0, maximum=None):
        """
        Update the application progress bar with current progress.
        This method now uses the integrated function from ui.ui_utils.
        """
        return update_progress_bar_with_ui(self, value, maximum)

    def decimate_for_plot(self, x, y, max_points=10000):
        """
        Reduce data points for efficient plotting while preserving visual features.
        This is a UI wrapper around the core.data_utils.decimate_for_plot function.
        """
        return decimate_for_plot_function(x, y, max_points)

    @ui_action(
        processing_message="Exporting plot...",
        success_message="Plot exported successfully",
        error_message="Error exporting plot"
    )
    def export_plot(self, figure, default_name="peak_analysis_plot"):
        """
        Export the current plot to an image file.
        This is a UI wrapper around the core.file_export.export_plot function.
        
        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The figure to export
        default_name : str, optional
            Default filename for the exported plot
        
        Returns
        -------
        str or None
            Path where the plot was saved, or None if canceled or an error occurred
        """
        return export_plot_function(self, figure, default_name)

    @ui_action(
        processing_message="Switching theme...",
        success_message="Theme switched successfully",
        error_message="Error switching theme"
    )
    def toggle_theme(self):
        """
        Toggle between light and dark theme and update UI elements accordingly.
        """
        new_theme = self.theme_manager.toggle_theme()
        
        # Apply the theme to the application
        self.style = self.theme_manager.apply_theme(self)
        
        # Update ScrolledText colors
        if self.results_summary:
            self.results_summary.config(
                bg=self.theme_manager.get_color('card_bg'),
                fg=self.theme_manager.get_color('text'),
                insertbackground=self.theme_manager.get_color('text')
            )
        
        # Update welcome label if it exists
        for child in self.blank_tab.winfo_children():
            if isinstance(child, ttk.Label):
                child.config(
                    foreground=self.theme_manager.get_color('text'),
                    background=self.theme_manager.get_color('background')
                )
        
        # Update visual elements on all tabs
        for tab in self.tab_control.winfo_children():
            if isinstance(tab, ttk.Frame):
                # Search for visual elements that need theme updates
                for frame in tab.winfo_children():
                    if isinstance(frame, ttk.LabelFrame):
                        # Look for canvas elements (diagrams)
                        self._update_frame_theme_elements(frame)
                        
                        # Also look in child frames
                        for child in frame.winfo_children():
                            if isinstance(child, ttk.Frame):
                                self._update_frame_theme_elements(child)
        
        # Update all existing figures to match the new theme
        for tab_name, figure in self.tab_figures.items():
            if figure:
                # Keep white background for plots regardless of theme
                figure.patch.set_facecolor('white')
                for ax in figure.get_axes():
                    ax.set_facecolor('white')
                    ax.tick_params(colors='#333333')  # Dark tick labels
                    ax.xaxis.label.set_color('#333333')  # Dark axis labels
                    ax.yaxis.label.set_color('#333333')
                    if ax.get_title():
                        ax.title.set_color('#333333')
                    # Dark spines
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#666666')
                
                # Find and redraw any canvas associated with this figure
                for widget in self.plot_tab_control.winfo_children():
                    if isinstance(widget, ttk.Frame):
                        for child in widget.winfo_children():
                            if hasattr(child, 'figure') and child.figure == figure:
                                child.draw()
        
        # Recreate the menu bar to update the theme toggle label
        from ui.components import create_menu_bar
        self.menu_bar = create_menu_bar(self)
        
        # Custom success message with theme name
        theme_name = "Dark" if new_theme == "dark" else "Light"
        self.preview_label.config(
            text=f"Switched to {theme_name} Theme", 
            foreground=self.theme_manager.get_color('success')
        )
        
        return new_theme
        
    def _update_frame_theme_elements(self, frame):
        """Helper method to update theme-aware elements within a frame"""
        # Update canvases
        for widget in frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.config(bg=self.theme_manager.get_color('card_bg'))
                
                # Try to find and update texts with primary color
                for item_id in widget.find_all():
                    if widget.type(item_id) == "text":
                        # Check if this is a label for filtered data
                        text = widget.itemcget(item_id, "text")
                        if "Filtered:" in text:
                            widget.itemconfig(item_id, fill=self.theme_manager.get_color('primary'))
                        
                    # Update line colors for filtered data
                    if widget.type(item_id) == "line":
                        # Get the current color
                        fill = widget.itemcget(item_id, "fill")
                        # If it's the primary color, update it
                        if fill == "#4285f4" or fill == "#3949ab" or fill.lower() == "#3f51b5":  # Common primary colors
                            widget.itemconfig(item_id, fill=self.theme_manager.get_color('primary'))
            
            # Update Scale widgets
            elif isinstance(widget, tk.Scale):
                widget.config(
                    bg=self.theme_manager.get_color('card_bg'),
                    fg=self.theme_manager.get_color('text'),
                    troughcolor=self.theme_manager.get_color('background')
                )
                
            # Update color indicators
            elif isinstance(widget, ttk.Label) and widget.cget("text").strip() == "":
                if widget.cget("background") == self.theme_manager.colors['light']['primary'] or \
                   widget.cget("background") == self.theme_manager.colors['dark']['primary']:
                    widget.config(background=self.theme_manager.get_color('primary'))

# Your main program code goes here

if __name__ == "__main__":
    app = Application()
    app.mainloop()