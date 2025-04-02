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
from plotting.double_peak_analysis import (
    analyze_double_peaks as analyze_double_peaks_function,
    show_next_double_peaks_page as show_next_double_peaks_page_function,
    show_prev_double_peaks_page as show_prev_double_peaks_page_function
)

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
        self.distance = tk.IntVar(value=5)  # Default value for minimum distance between peaks
        self.rel_height = tk.DoubleVar(value=0.8)  # Default value for relative height
        self.width_p = tk.StringVar(value="0.1,50")  # Default value for width range
        self.time_resolution = tk.DoubleVar(value=1e-4)  # Time resolution factor (default: 0.1ms)
        self.cutoff_value = tk.DoubleVar(value=0)  # Default 0 means auto-detect
        self.filter_enabled = tk.BooleanVar(value=True)  # Toggle for filtering (True=enabled)
        self.sigma_multiplier = tk.DoubleVar(value=5.0)  # Sigma multiplier for auto threshold detection (1-10)
        self.filter_bandwidth = tk.DoubleVar(value=0)  # Store the current filter bandwidth
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
        
        # Add double peak analysis toggle
        self.double_peak_analysis = tk.StringVar(value="0")  # "0" for normal, "1" for double peak

        # Add double peak analysis parameters
        self.double_peak_min_distance = tk.DoubleVar(value=0.001)  # 1 ms
        self.double_peak_max_distance = tk.DoubleVar(value=0.010)  # 10 ms
        self.double_peak_min_amp_ratio = tk.DoubleVar(value=0.1)   # 10%
        self.double_peak_max_amp_ratio = tk.DoubleVar(value=5.0)   # 500%
        self.double_peak_min_width_ratio = tk.DoubleVar(value=0.1) # 10%
        self.double_peak_max_width_ratio = tk.DoubleVar(value=5.0) # 500%
        
        # Add scale mode tracking
        self.log_scale_enabled = tk.BooleanVar(value=True)  # True for logarithmic, False for linear
        
        # Variables for double peak analysis results
        self.double_peaks = None
        self.current_double_peak_page = 0

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
            peaks, _ = find_peaks(self.x_value, height=threshold)
            
            if len(peaks) == 0:
                print("DEBUG: No peaks found above 70% threshold, using default cutoff")
                suggested_cutoff = 10.0  # Default cutoff if no peaks found
            else:
                print(f"DEBUG: Found {len(peaks)} peaks above 70% threshold")
                
                # Use the existing core functions but with our calculated threshold
                # instead of the big_counts and normalization_factor parameters
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
        
        # Update status if double peak analysis is enabled
        if self.double_peak_analysis.get() == "1":
            self.status_indicator.set_text("Double Peak Analysis Mode Enabled")
            
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
        processing_message="Generating scatter plot...",
        success_message="Scatter plot created successfully",
        error_message="Error creating scatter plot"
    )
    def plot_scatter(self):
        """
        Generate scatter plots of peak properties.
        This is a UI wrapper around the plotting.analysis_visualization.plot_scatter function.
        """
        return plot_scatter_function(self, profile_function=profile_function)
        
    @ui_action(
        processing_message="Analyzing double peaks...",
        success_message="Double peak analysis completed",
        error_message="Error analyzing double peaks"
    )
    def analyze_double_peaks(self):
        """
        Analyze double peaks and create visualizations.
        This is a UI wrapper around the plotting.double_peak_analysis.analyze_double_peaks function.
        """
        double_peaks, figures = analyze_double_peaks_function(self, profile_function=profile_function)
        
        if double_peaks and figures:
            # Unpack figures
            selection_figure, grid_figure = figures
            
            # Create or update the tab for double peak selection
            selection_tab_name = "Double Peak Selection"
            selection_tab_exists = False
            
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") == selection_tab_name:
                    self.plot_tab_control.forget(tab)
                    selection_tab_exists = True
                    break
            
            selection_tab = ttk.Frame(self.plot_tab_control)
            self.plot_tab_control.add(selection_tab, text=selection_tab_name)
            
            # Create canvas in the tab
            selection_canvas = FigureCanvasTkAgg(selection_figure, selection_tab)
            selection_canvas.draw()
            selection_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store the figure
            self.tab_figures[selection_tab_name] = selection_figure
            
            # Create or update the tab for double peak grid
            grid_tab_name = "Double Peak Grid"
            grid_tab_exists = False
            
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") == grid_tab_name:
                    self.plot_tab_control.forget(tab)
                    grid_tab_exists = True
                    break
            
            grid_tab = ttk.Frame(self.plot_tab_control)
            self.plot_tab_control.add(grid_tab, text=grid_tab_name)
            
            # Create canvas in the tab
            grid_canvas = FigureCanvasTkAgg(grid_figure, grid_tab)
            grid_canvas.draw()
            grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Store the figure
            self.tab_figures[grid_tab_name] = grid_figure
            
            # Select the selection tab if it's new, otherwise grid tab
            if not selection_tab_exists:
                self.plot_tab_control.select(selection_tab)
            elif not grid_tab_exists:
                self.plot_tab_control.select(grid_tab)
            
            # Update results summary
            summary_text = (
                f"Double Peak Analysis Results:\n"
                f"Found {len(double_peaks)} double peak pairs matching the criteria.\n"
                f"Parameters used:\n"
                f"- Distance range: {self.double_peak_min_distance.get()*1000:.1f} - {self.double_peak_max_distance.get()*1000:.1f} ms\n"
                f"- Amplitude ratio range: {self.double_peak_min_amp_ratio.get():.2f} - {self.double_peak_max_amp_ratio.get():.2f}\n"
                f"- Width ratio range: {self.double_peak_min_width_ratio.get():.2f} - {self.double_peak_max_width_ratio.get():.2f}\n"
            )
            self.update_results_summary(preview_text=summary_text)
            
            return double_peaks
        else:
            # If there was an error or no results, show a message
            self.preview_label.config(text="No double peaks found with current parameters", foreground="orange")
            return None

    def show_double_peaks_grid(self):
        """
        Show the grid view of detected double peaks.
        This selects the Double Peak Grid tab if it exists.
        """
        grid_tab_name = "Double Peak Grid"
        
        for tab in self.plot_tab_control.tabs():
            if self.plot_tab_control.tab(tab, "text") == grid_tab_name:
                self.plot_tab_control.select(tab)
                return True
        
        # If tab doesn't exist, run the analysis first
        self.preview_label.config(text="No double peak grid view exists. Running analysis first...", foreground="blue")
        return self.analyze_double_peaks()
        
    @ui_action(
        processing_message="Navigating to next double peaks...",
        success_message="Showing next set of double peaks",
        error_message="Error navigating to next double peaks"
    )
    def show_next_double_peaks_page(self):
        """
        Show the next page of double peaks.
        This is a UI wrapper around the plotting.double_peak_analysis.show_next_double_peaks_page function.
        """
        # Get the updated figure
        grid_figure = show_next_double_peaks_page_function(self)
        
        if grid_figure:
            # Update the tab
            grid_tab_name = "Double Peak Grid"
            
            # Find the tab
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") == grid_tab_name:
                    # Get the frame
                    grid_tab = self.plot_tab_control.nametowidget(tab)
                    
                    # Remove old canvas
                    for widget in grid_tab.winfo_children():
                        widget.destroy()
                    
                    # Create new canvas
                    grid_canvas = FigureCanvasTkAgg(grid_figure, grid_tab)
                    grid_canvas.draw()
                    grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    # Select the tab
                    self.plot_tab_control.select(tab)
                    
                    # Store the updated figure
                    self.tab_figures[grid_tab_name] = grid_figure
                    
                    # Update status
                    page_num = self.current_double_peak_page + 1
                    total_pages = (len(self.double_peaks) - 1) // 25 + 1
                    self.preview_label.config(
                        text=f"Showing double peaks page {page_num} of {total_pages}",
                        foreground="green"
                    )
                    
                    return True
            
            # If tab doesn't exist, create it
            self.preview_label.config(text="Double peak grid tab not found. Creating it...", foreground="blue")
            return self.analyze_double_peaks()
        else:
            self.preview_label.config(text="No double peaks to show", foreground="orange")
            return False
            
    @ui_action(
        processing_message="Navigating to previous double peaks...",
        success_message="Showing previous set of double peaks",
        error_message="Error navigating to previous double peaks"
    )
    def show_prev_double_peaks_page(self):
        """
        Show the previous page of double peaks.
        This is a UI wrapper around the plotting.double_peak_analysis.show_prev_double_peaks_page function.
        """
        # Get the updated figure
        grid_figure = show_prev_double_peaks_page_function(self)
        
        if grid_figure:
            # Update the tab
            grid_tab_name = "Double Peak Grid"
            
            # Find the tab
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") == grid_tab_name:
                    # Get the frame
                    grid_tab = self.plot_tab_control.nametowidget(tab)
                    
                    # Remove old canvas
                    for widget in grid_tab.winfo_children():
                        widget.destroy()
                    
                    # Create new canvas
                    grid_canvas = FigureCanvasTkAgg(grid_figure, grid_tab)
                    grid_canvas.draw()
                    grid_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    
                    # Select the tab
                    self.plot_tab_control.select(tab)
                    
                    # Store the updated figure
                    self.tab_figures[grid_tab_name] = grid_figure
                    
                    # Update status
                    page_num = self.current_double_peak_page + 1
                    total_pages = (len(self.double_peaks) - 1) // 25 + 1
                    self.preview_label.config(
                        text=f"Showing double peaks page {page_num} of {total_pages}",
                        foreground="green"
                    )
                    
                    return True
            
            # If tab doesn't exist, create it
            self.preview_label.config(text="Double peak grid tab not found. Creating it...", foreground="blue")
            return self.analyze_double_peaks()
        else:
            self.preview_label.config(text="No double peaks to show", foreground="orange")
            return False

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

    @ui_action(
        processing_message="Saving peak information...",
        success_message="Peak information saved successfully",
        error_message="Error saving peak information"
    )
    def save_peak_information_to_csv(self):
        """
        Save detected peak information to a file with configurable format.
        This is a UI wrapper around the core.file_export.save_peak_information_to_csv function.
        """
        from core.file_export import save_peak_information_to_csv as save_peaks
        return save_peaks(self)
            
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
        processing_message="Switching file mode...",
        success_message="File mode switched successfully",
        error_message="Error switching file mode"
    )
    def on_file_mode_change(self):
        """
        Handle changes to the file mode (single or batch).
        This updates UI elements based on the selected mode.
        """
        return on_file_mode_change_with_ui(self)
        
    @ui_action(
        processing_message="Updating double peak mode...",
        success_message="Double peak mode updated",
        error_message="Error updating double peak mode"
    )
    def on_double_peak_mode_change(self):
        """
        Handle changes to the double peak analysis mode.
        This method adds or removes the double peak analysis tab based on the mode.
        """
        try:
            # Check if double peak mode is enabled
            double_peak_enabled = self.double_peak_analysis.get() == "1"
            
            # Update status message
            if double_peak_enabled:
                self.status_indicator.set_text("Double Peak Analysis Mode Enabled")
            else:
                self.status_indicator.set_text("Normal Analysis Mode")
            
            # Look for existing double peak tab
            found_tab = False
            for tab in self.tab_control.tabs():
                tab_text = self.tab_control.tab(tab, "text")
                if tab_text == "Double Peak Analysis":
                    found_tab = True
                    if not double_peak_enabled:
                        # Remove the tab if double peak mode is disabled
                        self.tab_control.forget(tab)
                    break
            
            # Add the tab if it doesn't exist and double peak mode is enabled
            if double_peak_enabled and not found_tab:
                from ui.components import create_double_peak_analysis_tab
                create_double_peak_analysis_tab(self, self.tab_control)
            
            # Also update plot tabs - remove double peak tabs if mode is disabled
            if not double_peak_enabled:
                for tab in list(self.plot_tab_control.tabs()):
                    tab_text = self.plot_tab_control.tab(tab, "text")
                    if tab_text in ["Double Peak Selection", "Double Peak Grid"]:
                        self.plot_tab_control.forget(tab)
                        # Also remove from tab_figures
                        if tab_text in self.tab_figures:
                            del self.tab_figures[tab_text]
            
            return True
            
        except Exception as e:
            self.show_error("Error updating double peak mode", e)
            return False

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
    def export_plot(self, figure=None, default_name="peak_analysis_plot"):
        """
        Export the current plot to an image file.
        This is a UI wrapper around the core.file_export.export_plot function.
        
        Parameters
        ----------
        figure : matplotlib.figure.Figure, optional
            The figure to export. If None, will try to get the current figure from the active tab.
        default_name : str, optional
            Default filename for the exported plot
        
        Returns
        -------
        str or None
            Path where the plot was saved, or None if canceled or an error occurred
        """
        # If no figure is provided, try to get the current figure from the active tab
        if figure is None:
            # Get the current tab
            current_tab = self.plot_tab_control.select()
            tab_text = self.plot_tab_control.tab(current_tab, "text")
            
            # Try to get the figure from the tab_figures dictionary
            if tab_text in self.tab_figures:
                figure = self.tab_figures[tab_text]
            else:
                # If no figure found, show error message
                self.preview_label.config(
                    text="No plot available to export. Please generate a plot first.",
                    foreground=self.theme_manager.get_color('error')
                )
                return None
        
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

    @ui_action(
        processing_message="Toggling scale mode...",
        success_message="Scale mode toggled successfully",
        error_message="Error toggling scale mode"
    )
    def toggle_scale_mode(self):
        """
        Toggle between linear and logarithmic scales for peak analysis plots.
        """
        try:
            # Toggle the scale mode
            self.log_scale_enabled.set(not self.log_scale_enabled.get())
            
            # Get the current scale mode
            scale_mode = 'log' if self.log_scale_enabled.get() else 'linear'
            
            # Update the plots if they exist
            if hasattr(self, 'data_figure') and self.data_figure is not None:
                # Get all axes
                axes = self.data_figure.get_axes()
                
                # Update scale for each subplot (except throughput)
                for ax in axes[:-1]:  # Skip the last axis (throughput)
                    ax.set_yscale(scale_mode)
                
                # Redraw the canvas
                if hasattr(self, 'data_canvas') and self.data_canvas is not None:
                    self.data_canvas.draw()
                
                # Update status message
                self.preview_label.config(
                    text=f"Switched to {scale_mode.capitalize()} Scale",
                    foreground=self.theme_manager.get_color('success')
                )
            
            return True
            
        except Exception as e:
            self.show_error("Error toggling scale mode", str(e))
            return False

# Your main program code goes here

if __name__ == "__main__":
    app = Application()
    app.mainloop()