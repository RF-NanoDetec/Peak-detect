"""                                                                                             
Created 2 Apr 2025 21:518:48 2024

@author: Lucjan Grzegorzewski
"""

# Standard library
import os
import time
import logging
import traceback
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party libraries
import numpy as np  # numpy.trapz will be used instead of auc
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
from numba import njit
# Tkinter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Tcl
from tkinter.scrolledtext import ScrolledText
import sys
from PIL import Image, ImageTk
import datetime
import platform
import shutil
import json
import csv
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

# Local imports
from config.settings import Config
from config import resource_path, APP_VERSION
from config.environment import load_user_preferences, save_user_preferences
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
    on_file_mode_change_with_ui,
    ui_operation
)

# Import core modules
from core.peak_detection import PeakDetector
from core.file_handler import browse_files_with_ui
from core.data_analysis import (
    calculate_peak_areas as calculate_peak_areas_function,
    calculate_peak_intervals,
    calculate_auto_threshold,
    analyze_time_resolved as analyze_time_resolved_function,
    analyze_time_resolved_pure as analyze_time_resolved_pure_function,
)
from core.data_utils import (
    decimate_for_plot as decimate_for_plot_function,
    get_width_range as get_width_range_function,
    reset_application_state_with_ui,
    find_nearest,
    timestamps_to_seconds,  # For single timestamp conversion
    convert_width_ms_to_samples,
)
from core.peak_analysis_utils import timestamps_array_to_seconds, adjust_lowpass_cutoff, calculate_lowpass_cutoff, find_peaks_with_window
from core.file_export import (
    export_plot as export_plot_function,
    save_peak_information_to_csv as save_peak_information_to_csv_function,
    save_double_peak_information_to_csv as save_double_peak_information_to_csv_function
)

# Import UI modules
from ui.components import create_control_panel, create_menu_bar, create_preview_frame

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

# Seaborn global styling is handled via ThemeManager's rcParams; avoid global overrides here


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
        
        # Add keyboard shortcuts - bind to all widgets and use add=True
        self.bind_all('<Control-o>', lambda e: self.browse_file(), add=True)  # Ctrl+O to open file
        self.bind_all('<Control-e>', lambda e: self.export_plot(), add=True)  # Ctrl+E to export plot
        self.bind_all('<Control-q>', lambda e: self.quit(), add=True)  # Ctrl+Q to quit
        self.bind_all('<Control-/>', lambda e: self.show_shortcuts_dialog(), add=True)  # Ctrl+/ to show shortcuts
        self.bind_all('<F5>', lambda e: self.run_all_pipeline(), add=True)  # F5 Run All
        
        # Load preferences
        self.prefs = load_user_preferences()

        # Initialize theme manager using saved preference if available
        theme_name = self.prefs.get('theme', 'dark')
        self.theme_manager = ThemeManager(theme_name=theme_name)
        self.style = self.theme_manager.apply_theme(self)
        # Initialize recent files list in prefs
        self.prefs.setdefault('recent_files', [])

        # Ensure matplotlib theme is synchronized
        try:
            self.theme_manager.apply_matplotlib_theme()
        except Exception:
            pass
        
        # Store the StatusIndicator class for use in create_control_panel
        self.status_indicator_class = StatusIndicator
        
        # Top toolbar
        try:
            from ui.components import create_toolbar
            self.toolbar = create_toolbar(self, self)
        except Exception:
            pass

        # Initialize high-resolution figure
        self.figure = Figure(
            figsize=Config.Plot.FIGURE_SIZE, 
            dpi=Config.Plot.DPI,
            facecolor=self.theme_manager.get_color('canvas_bg')
        )
        
        # Configure default plot styling via ThemeManager and Config
        try:
            self.theme_manager.apply_matplotlib_theme()
        except Exception:
            pass
        # Apply numeric rcParams that come from Config (non-color aesthetic tokens)
        try:
            plt.rcParams['figure.dpi'] = Config.Plot.DPI
            plt.rcParams['savefig.dpi'] = Config.Plot.EXPORT_DPI
            plt.rcParams['lines.linewidth'] = Config.Plot.LINE_WIDTH
            plt.rcParams['font.size'] = Config.Plot.FONT_SIZE
            plt.rcParams['axes.titlesize'] = Config.Plot.TITLE_SIZE
            plt.rcParams['axes.labelsize'] = Config.Plot.LABEL_SIZE
        except Exception:
            pass

        # Initialize figure and canvas as None
        self.canvas = None

        self.file_path = tk.StringVar()
        self.start_time = tk.StringVar(value="0:00")
        self.height_lim = tk.DoubleVar(value=float(self.prefs.get('prominence_threshold', 20)))
        self.distance = tk.IntVar(value=int(self.prefs.get('distance', 5)))  # minimum distance between peaks
        self.rel_height = tk.DoubleVar(value=float(self.prefs.get('rel_height', 0.8)))  # relative height
        self.width_p = tk.StringVar(value=str(self.prefs.get('width_ms', "0.1,50")))  # width range in ms
        self.time_resolution = tk.DoubleVar(value=1e-4)  # Time resolution factor (default: 0.1ms)
        self.cutoff_value = tk.DoubleVar(value=float(self.prefs.get('cutoff_value', 0)))  # 0 means auto-detect
        self.filter_enabled = tk.BooleanVar(value=True)  # Toggle for filtering (True=enabled)
        self.sigma_multiplier = tk.DoubleVar(value=float(self.prefs.get('sigma_multiplier', 5.0)))  # Sigma multiplier for auto threshold detection (1-10)
        self.filter_bandwidth = tk.DoubleVar(value=0)  # Store the current filter bandwidth
        self.filtered_signal = None
        self.rect_selector = None

        # Protocol Variables
        self.protocol_start_time = tk.StringVar()
        self.protocol_id_filter = tk.StringVar()  # New: ID filter
        self.protocol_buffer = tk.StringVar()     # New: Buffer
        self.protocol_buffer_concentration = tk.StringVar()  # New: Buffer concentration
        self.protocol_measurement_date = tk.StringVar()  # New: Measurement date
        self.protocol_sample_number = tk.StringVar()  # New: Sample number
        self.protocol_particle = tk.StringVar()
        self.protocol_concentration = tk.StringVar()
        self.protocol_stamp = tk.StringVar()
        self.protocol_laser_power = tk.StringVar()
        self.protocol_setup = tk.StringVar()
        self.protocol_notes = tk.StringVar()

        # Add file mode selection
        self.file_mode = tk.StringVar(value=self.prefs.get('file_mode', "single"))  # "single" or "batch"
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
        self.log_scale_enabled = tk.BooleanVar(value=bool(self.prefs.get('log_scale', True)))
        # Guided mode tracking (default on)
        self.guided_mode = tk.IntVar(value=1 if bool(self.prefs.get('guided_mode', True)) else 0)
        
        # Variables for double peak analysis results
        self.double_peaks = None
        self.current_double_peak_page = 0

        # Add loaded files tracking
        self.loaded_files = []
        
        # Add a specific entry for file order in protocol
        self.protocol_files = tk.StringVar()
        
        # Add throughput interval parameter for time-resolved analysis
        self.throughput_interval = tk.DoubleVar(value=float(self.prefs.get('throughput_interval', 10.0)))
        
        # Create the menu bar
        self.menu_bar = create_menu_bar(self)

        self.create_widgets()
        # Restore splitter sizes if saved
        try:
            sizes = self.prefs.get('paned_sizes')
            if sizes and hasattr(self, 'paned_window'):
                self.paned_window.sashpos(0, sizes[0])
                # Right sash move: need width; apply after idle to ensure geometry
                self.after(100, lambda: self.paned_window.sashpos(1, sizes[1]))
        except Exception:
            pass
        self.blank_tab_exists = True  # Track if the blank tab exists

        # Initialize data plot attributes
        self.data_figure = None
        self.data_canvas = None
        self.data_original_xlims = None
        self.data_original_ylims = None

        self.tab_figures = {}  # Dictionary to store figures for each tab
        
        # Initialize the PeakDetector
        self.peak_detector = PeakDetector(logger=self.logger)
        
        # Set icon and window title (guard against missing icon)
        icon_path = self.get_icon_path()
        if icon_path:
            try:
                self.iconbitmap(icon_path)
            except Exception:
                print(f"Warning: Unable to set window icon from {icon_path}")
        self.setup_window_title()

    def toggle_guided_mode(self):
        try:
            # Persist preference
            self.prefs['guided_mode'] = bool(self.guided_mode.get())
            save_user_preferences(self.prefs)
        except Exception:
            pass
        try:
            # Update visibility on relevant tabs/controls
            from ui.components import update_guided_mode_visibility
            update_guided_mode_visibility(self)
        except Exception:
            pass
        
    def get_icon_path(self):
        """Get path to icon file, handling packaged app scenario."""
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # Running in a PyInstaller bundle
            base_path = sys._MEIPASS
        else:
            # Running in a normal Python environment
            base_path = os.path.dirname(os.path.abspath(__file__))

        # Construct the path relative to the base path
        # This assumes your 'resources' folder is copied into the root of the bundle
        icon_path = os.path.join(base_path, "resources", "images", "icon.ico")

        # Check if the file exists at the determined path
        if os.path.exists(icon_path):
             return icon_path
        else:
             # Add a print statement for debugging if the icon isn't found
             print(f"Warning: Icon file not found at expected location: {icon_path}")
             return "" # Return empty string if not found

    def setup_window_title(self):
        """Set up a more descriptive window title"""
        version = "1.0"  # Define your version number
        self.title(f"Peak Analysis Tool v{version} - Signal Processing and Analysis")

    def reset_layout(self):
        try:
            # Remove saved pane sizes and restore defaults
            if 'paned_sizes' in self.prefs:
                del self.prefs['paned_sizes']
                save_user_preferences(self.prefs)
            # Recreate layout by destroying and rebuilding widgets
            for child in self.winfo_children():
                child.destroy()
            self.create_widgets()
        except Exception:
            pass

    def quit(self):
        try:
            # Save preferences on exit
            try:
                if hasattr(self, 'paned_window'):
                    sizes = []
                    try:
                        sizes.append(self.paned_window.sashpos(0))
                        sizes.append(self.paned_window.sashpos(1))
                    except Exception:
                        pass
                    self.prefs['paned_sizes'] = sizes
                self.prefs['file_mode'] = self.file_mode.get()
                self.prefs['log_scale'] = bool(self.log_scale_enabled.get())
                self.prefs['throughput_interval'] = float(self.throughput_interval.get())
                # Persist filter settings if present
                try:
                    self.prefs['filter_type'] = self.filter_type_var.get()
                    self.prefs['savgol_window'] = int(self.savgol_window_var.get()) if self.savgol_window_var.get() else ''
                    self.prefs['savgol_polyorder'] = int(self.savgol_polyorder_var.get()) if self.savgol_polyorder_var.get() else ''
                    self.prefs['butter_order'] = int(self.butter_order_var.get()) if self.butter_order_var.get() else Config.DEFAULT_BUTTER_FILTER_ORDER
                except Exception:
                    pass
                save_user_preferences(self.prefs)
            except Exception:
                pass
            super().quit()
        except Exception:
            try:
                self.destroy()
            except Exception:
                pass

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
            
            # Update the cutoff value in GUI - use the new variable
            if hasattr(self, 'filter_cutoff_freq'):
                self.filter_cutoff_freq.set(f"{suggested_cutoff:.2f}")
            else:
                # Fallback or log if the new variable doesn't exist for some reason
                print("Warning: app.filter_cutoff_freq not found, falling back to app.cutoff_value")
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
        
        # Top toolbar
        toolbar = ttk.Frame(main_frame, style='Toolbar.TFrame')
        toolbar.pack(fill=tk.X, side=tk.TOP, padx=6, pady=6)

        self.btn_load = ttk.Button(toolbar, text="Load", command=self.browse_file, style='Tool.TButton')
        self.btn_load.pack(side=tk.LEFT, padx=3)
        from ui.ui_utils import add_tooltip as _add_tt
        _add_tt(self.btn_load, "Load data file(s)")

        self.btn_analyze = ttk.Button(toolbar, text="Analyze", command=self.start_analysis, style='Tool.TButton')
        self.btn_analyze.pack(side=tk.LEFT, padx=3)
        _add_tt(self.btn_analyze, "Filter and process the loaded data")

        self.btn_detect = ttk.Button(toolbar, text="Detect Peaks", command=self.run_peak_detection, style='Tool.TButton')
        self.btn_detect.pack(side=tk.LEFT, padx=3)
        _add_tt(self.btn_detect, "Detect peaks with current parameters")

        self.btn_plot = ttk.Button(toolbar, text="Plot Analysis", command=self.plot_data, style='Tool.TButton')
        self.btn_plot.pack(side=tk.LEFT, padx=3)
        _add_tt(self.btn_plot, "Show peak statistics and throughput plots")

        self.btn_export = ttk.Button(toolbar, text="Export Plot", command=self.export_plot, style='Tool.TButton')
        self.btn_export.pack(side=tk.LEFT, padx=3)
        _add_tt(self.btn_export, "Export current plot as image")

        # Run All button
        self.btn_run_all = ttk.Button(toolbar, text="Run All", command=self.run_all_pipeline, style='Tool.TButton')
        self.btn_run_all.pack(side=tk.LEFT, padx=3)
        _add_tt(self.btn_run_all, "Run Load → Preprocess → Detect → Analyze")

        # Theme toggle button on toolbar (right side)
        self.btn_theme_toggle = ttk.Button(toolbar, text="Theme", command=self.toggle_theme, style='Tool.TButton')
        self.btn_theme_toggle.pack(side=tk.RIGHT, padx=3)
        _add_tt(self.btn_theme_toggle, "Toggle between Light and Dark themes")

        # Paned layout: left (controls), center (plots), right (summary)
        self.paned_window = ttk.Panedwindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left pane: Control panel
        control_container = ttk.Frame(self.paned_window)
        control_container.pack(fill=tk.BOTH, expand=False)
        control_frame = create_control_panel(self, control_container)
        control_frame.pack(fill=tk.BOTH, expand=True)
        self.paned_window.add(control_container, weight=0)
        
        # Center pane: Preview frame with plot tabs
        preview_container = ttk.Frame(self.paned_window)
        preview_container.pack(fill=tk.BOTH, expand=True)
        preview_frame = create_preview_frame(self, preview_container)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        self.paned_window.add(preview_container, weight=3)
        
        # Right pane: Results summary
        summary_container = ttk.Frame(self.paned_window)
        summary_container.pack(fill=tk.BOTH, expand=False)
        
        summary_frame = ttk.Frame(summary_container)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure the summary frame's grid
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        
        # Set a fixed width for the results summary panel
        summary_frame.grid_propagate(False)  # Prevent the frame from resizing to its children
        summary_frame.configure(width=300)   # Set a fixed width
        
        # Create the results summary label frame
        results_label_frame = ttk.LabelFrame(summary_frame, text="Results Summary")
        results_label_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Add a ScrolledText widget for results summary
        self.results_summary = ScrolledText(results_label_frame, wrap=tk.WORD, height=30, width=30)
        self.results_summary.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_summary.config(state=tk.DISABLED)
        
        # Apply theme to the ScrolledText widget
        self.theme_manager.apply_scrolledtext_theme(self.results_summary)
        
        self.paned_window.add(summary_container, weight=0)

        # Initialize action states
        try:
            self.update_action_states()
        except Exception:
            pass

    def update_action_states(self):
        """Enable/disable primary actions based on current state."""
        try:
            data_loaded = hasattr(self, 'data') and self.data is not None
            filtered_ready = hasattr(self, 'filtered_signal') and self.filtered_signal is not None
            peaks_ready = hasattr(self, 'peaks') and getattr(self, 'peaks', None) is not None and len(self.peaks) > 0

            # Toolbar buttons
            try:
                self.btn_analyze.config(state=(tk.NORMAL if data_loaded else tk.DISABLED))
            except Exception:
                pass
            try:
                self.btn_detect.config(state=(tk.NORMAL if filtered_ready else tk.DISABLED))
            except Exception:
                pass
            try:
                self.btn_plot.config(state=(tk.NORMAL if peaks_ready else tk.DISABLED))
            except Exception:
                pass

            # Tab-local buttons if present
            for attr, enabled in [
                ('view_raw_btn', data_loaded),
                ('process_btn', data_loaded),
                ('btn_detect_tab', filtered_ready),
                ('btn_view_peaks_tab', peaks_ready),
                ('btn_next_peaks_tab', peaks_ready),
                ('btn_save_results_tab', peaks_ready),
            ]:
                try:
                    btn = getattr(self, attr)
                    btn.config(state=(tk.NORMAL if enabled else tk.DISABLED))
                except Exception:
                    pass
        except Exception:
            pass

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

    def show_shortcuts_dialog(self):
        try:
            top = tk.Toplevel(self)
            top.title("Keyboard Shortcuts")
            top.transient(self)
            top.grab_set()
            top.configure(background=self.theme_manager.get_color('background'))
            frm = ttk.Frame(top, padding=10)
            frm.pack(fill=tk.BOTH, expand=True)
            sections = [
                ("General", [
                    ("Ctrl+O", "Open File/Folder"),
                    ("Ctrl+E", "Export Current Plot"),
                    ("Ctrl+Q", "Quit"),
                    ("Ctrl+/", "Show Shortcuts"),
                ]),
                ("Workflow", [
                    ("Analyze", "Run preprocessing/analysis (toolbar)"),
                    ("Detect Peaks", "Run peak detection (toolbar)"),
                    ("Plot Analysis", "Show analysis plots (toolbar)"),
                ]),
            ]
            for section, items in sections:
                ttk.Label(frm, text=section, style='Heading.TLabel').pack(anchor='w', pady=(8, 2))
                for keys, desc in items:
                    row = ttk.Frame(frm)
                    row.pack(fill=tk.X, pady=1)
                    ttk.Label(row, text=keys, width=16).pack(side=tk.LEFT)
                    ttk.Label(row, text=desc).pack(side=tk.LEFT)
            ttk.Button(frm, text='Close', command=top.destroy).pack(pady=10, anchor='e')
        except Exception:
            pass

    @ui_action(
        processing_message="Running full pipeline...",
        success_message="Pipeline complete",
        error_message="Pipeline failed"
    )
    def run_all_pipeline(self):
        """Run the main workflow in sequence with current settings."""
        try:
            # 1) If no data, prompt load
            if not (hasattr(self, 'data') and self.data is not None):
                if not self.browse_file():
                    return False
            self.update_progress_bar_with_ui(10)

            # 2) Preprocess / Analyze
            self.start_analysis()
            self.update_progress_bar_with_ui(40)

            # 3) Detect peaks (validate handled inside)
            ok = self.run_peak_detection()
            self.update_progress_bar_with_ui(70)

            # 4) Show analysis plots
            self.plot_data()
            self.update_progress_bar_with_ui(100)

            # Bring the Analyze tab forward in the right notebook if exists
            try:
                for tab in self.plot_tab_control.tabs():
                    if self.plot_tab_control.tab(tab, 'text') in ('Analysis', 'Analyze'):
                        self.plot_tab_control.select(tab)
                        break
            except Exception:
                pass
            return ok
        finally:
            try:
                self.update_action_states()
            except Exception:
                pass

    @ui_action(
        processing_message="Saving preset...",
        success_message="Preset saved",
        error_message="Error saving preset"
    )
    def save_preset(self):
        try:
            preset = {
                'file_mode': self.file_mode.get(),
                'time_resolution_seconds': float(self.time_resolution.get()) if hasattr(self.time_resolution, 'get') else float(self.time_resolution),
                'filter_enabled': bool(self.filter_enabled.get()),
                'filter_type': self.filter_type_var.get() if hasattr(self, 'filter_type_var') else None,
                'filter_cutoff_freq': float(self.filter_cutoff_freq.get()) if hasattr(self, 'filter_cutoff_freq') and self.filter_cutoff_freq.get() != '' else None,
                'butter_order': int(self.butter_order_var.get()) if hasattr(self, 'butter_order_var') and self.butter_order_var.get() != '' else None,
                'savgol_window': int(self.savgol_window_var.get()) if hasattr(self, 'savgol_window_var') and self.savgol_window_var.get() != '' else None,
                'savgol_polyorder': int(self.savgol_polyorder_var.get()) if hasattr(self, 'savgol_polyorder_var') and self.savgol_polyorder_var.get() != '' else None,
                'prominence_threshold': float(self.height_lim.get()),
                'distance': int(self.distance.get()),
                'rel_height': float(self.rel_height.get()),
                'width_ms': self.width_p.get(),
                'prominence_ratio': float(self.prominence_ratio.get()) if hasattr(self, 'prominence_ratio') else None,
                'throughput_interval': float(self.throughput_interval.get()),
                'log_scale': bool(self.log_scale_enabled.get()),
                'guided_mode': bool(self.guided_mode.get()),
                'double_peak_analysis': self.double_peak_analysis.get(),
                'double_peak_min_distance': float(self.double_peak_min_distance.get()),
                'double_peak_max_distance': float(self.double_peak_max_distance.get()),
                'double_peak_min_amp_ratio': float(self.double_peak_min_amp_ratio.get()),
                'double_peak_max_amp_ratio': float(self.double_peak_max_amp_ratio.get()),
                'double_peak_min_width_ratio': float(self.double_peak_min_width_ratio.get()),
                'double_peak_max_width_ratio': float(self.double_peak_max_width_ratio.get()),
            }
            path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[["JSON files", "*.json"], ["All files", "*.*"]],
                title="Save Preset"
            )
            if not path:
                return None
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(preset, f, indent=2)
            return path
        except Exception as e:
            raise e

    @ui_action(
        processing_message="Loading preset...",
        success_message="Preset loaded",
        error_message="Error loading preset"
    )
    def load_preset(self):
        try:
            path = filedialog.askopenfilename(
                defaultextension=".json",
                filetypes=[["JSON files", "*.json"], ["All files", "*.*"]],
                title="Load Preset"
            )
            if not path:
                return None
            with open(path, 'r', encoding='utf-8') as f:
                preset = json.load(f)

            if 'file_mode' in preset:
                self.file_mode.set(preset['file_mode'])
            if 'time_resolution_seconds' in preset:
                self.time_resolution.set(float(preset['time_resolution_seconds']))
                try:
                    if hasattr(self, 'time_resolution_ms'):
                        self.time_resolution_ms.set(f"{float(preset['time_resolution_seconds'])*1000:.1f}")
                except Exception:
                    pass
            if 'filter_enabled' in preset:
                self.filter_enabled.set(bool(preset['filter_enabled']))
            if 'filter_type' in preset and hasattr(self, 'filter_type_var'):
                self.filter_type_var.set(preset['filter_type'])
            if 'filter_cutoff_freq' in preset and hasattr(self, 'filter_cutoff_freq') and preset['filter_cutoff_freq'] is not None:
                self.filter_cutoff_freq.set(str(preset['filter_cutoff_freq']))
            if 'butter_order' in preset and hasattr(self, 'butter_order_var') and preset['butter_order'] is not None:
                self.butter_order_var.set(str(preset['butter_order']))
            if 'savgol_window' in preset and hasattr(self, 'savgol_window_var') and preset['savgol_window'] is not None:
                self.savgol_window_var.set(str(preset['savgol_window']))
            if 'savgol_polyorder' in preset and hasattr(self, 'savgol_polyorder_var') and preset['savgol_polyorder'] is not None:
                self.savgol_polyorder_var.set(str(preset['savgol_polyorder']))

            if 'prominence_threshold' in preset:
                self.height_lim.set(float(preset['prominence_threshold']))
            if 'distance' in preset:
                self.distance.set(int(preset['distance']))
            if 'rel_height' in preset:
                self.rel_height.set(float(preset['rel_height']))
            if 'width_ms' in preset:
                self.width_p.set(preset['width_ms'])
                try:
                    if hasattr(self, 'width_min_ms_var') and hasattr(self, 'width_max_ms_var'):
                        parts = [p.strip() for p in preset['width_ms'].split(',')]
                        if len(parts) >= 2:
                            self.width_min_ms_var.set(parts[0])
                            self.width_max_ms_var.set(parts[1])
                except Exception:
                    pass
            if 'prominence_ratio' in preset and hasattr(self, 'prominence_ratio') and preset['prominence_ratio'] is not None:
                self.prominence_ratio.set(float(preset['prominence_ratio']))
            if 'throughput_interval' in preset:
                self.throughput_interval.set(float(preset['throughput_interval']))
            if 'log_scale' in preset:
                self.log_scale_enabled.set(bool(preset['log_scale']))
            if 'guided_mode' in preset:
                self.guided_mode.set(1 if preset['guided_mode'] else 0)
                try:
                    self.toggle_guided_mode()
                except Exception:
                    pass
            if 'double_peak_analysis' in preset:
                self.double_peak_analysis.set(str(preset['double_peak_analysis']))
            for key, var in [
                ('double_peak_min_distance', self.double_peak_min_distance),
                ('double_peak_max_distance', self.double_peak_max_distance),
                ('double_peak_min_amp_ratio', self.double_peak_min_amp_ratio),
                ('double_peak_max_amp_ratio', self.double_peak_max_amp_ratio),
                ('double_peak_min_width_ratio', self.double_peak_min_width_ratio),
                ('double_peak_max_width_ratio', self.double_peak_max_width_ratio),
            ]:
                if key in preset:
                    try:
                        var.set(float(preset[key]))
                    except Exception:
                        pass

            try:
                self.update_action_states()
            except Exception:
                pass
            try:
                from ui.components import update_guided_mode_visibility
                update_guided_mode_visibility(self)
            except Exception:
                pass

            return True
        except Exception as e:
            raise e

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
        ui.tooltips.Tooltip
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
        import logging
        logging.getLogger(__name__).debug(f"Using time resolution: {time_res}")
        
        # Update status if double peak analysis is enabled
        if self.double_peak_analysis.get() == "1":
            self.status_indicator.set_text("Double Peak Analysis Mode Enabled")
            
        result = browse_files_with_ui(self, time_resolution=time_res)
        # Track recent files if a path or list of paths is returned
        try:
            paths = []
            if isinstance(result, str):
                paths = [result]
            elif isinstance(result, (list, tuple)):
                paths = [p for p in result if isinstance(p, str)]
            if paths:
                recent = self.prefs.get('recent_files', [])
                for p in paths:
                    if p in recent:
                        recent.remove(p)
                    recent.insert(0, p)
                self.prefs['recent_files'] = recent[:10]
                save_user_preferences(self.prefs)
        except Exception:
            pass
        try:
            self.update_action_states()
        except Exception:
            pass
        return result

    @ui_action(
        processing_message="Opening file...",
        success_message="File opened",
        error_message="Error opening file"
    )
    def open_path(self, path):
        """Open a specific file path and update recents."""
        try:
            # Reuse browse handler path if available
            # Set internal state and call core handler directly
            # Delegating to core.file_handler if it supports direct path load
            from core.file_handler import load_file_direct_with_ui as load_direct
        except Exception:
            load_direct = None
        try:
            if load_direct:
                loaded = load_direct(self, path)
            else:
                # Fallback: set selection and call browse handler
                loaded = browse_files_with_ui(self, preset_paths=[path], time_resolution=self.time_resolution.get())
            # Update recent
            recent = self.prefs.get('recent_files', [])
            if path in recent:
                recent.remove(path)
            recent.insert(0, path)
            self.prefs['recent_files'] = recent[:10]
            save_user_preferences(self.prefs)
            return loaded
        except Exception as e:
            raise e

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
        result = reset_application_state_with_ui(self)
        try:
            self.update_action_states()
        except Exception:
            pass
        return result

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
        result = start_analysis_function(self, profile_function=profile_function)
        try:
            self.update_action_states()
        except Exception:
            pass
        return result
        
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
        try:
            # Validate parameters before running
            from core.data_utils import validate_peak_params
            ok, msg = validate_peak_params(
                prominence_threshold=self.height_lim.get(),
                distance=self.distance.get(),
                rel_height=self.rel_height.get(),
                width_ms=self.width_p.get(),
                time_resolution=self.time_resolution.get() if hasattr(self.time_resolution, 'get') else self.time_resolution,
                prominence_ratio=self.prominence_ratio.get() if hasattr(self, 'prominence_ratio') else None,
            )
            if not ok:
                # Inline validation feedback
                try:
                    if hasattr(self, 'validation_label'):
                        self.validation_label.config(text=msg, foreground=self.theme_manager.get_color('error'))
                    else:
                        # Create on-demand near toolbar
                        self.validation_label = ttk.Label(self, text=msg, foreground=self.theme_manager.get_color('error'))
                        self.validation_label.pack(side=tk.TOP, anchor='w', padx=8)
                except Exception:
                    pass
                self.preview_label.config(text=f"Parameter error: {msg}", foreground=self.theme_manager.get_color('error'))
                if hasattr(self, 'status_indicator'):
                    self.status_indicator.set_state('warning')
                    self.status_indicator.set_text(msg)
                return False
            # Debug: Print width values used for peak detection
            width_values = self.width_p.get().strip().split(',')
            time_res = self.time_resolution.get() if hasattr(self.time_resolution, 'get') else self.time_resolution
            sampling_rate = 1 / time_res
            width_samples = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
            import logging
            logging.getLogger(__name__).debug(f"[run_peak_detection] width_p (ms): {width_values}, width_p (samples): {width_samples}, sampling_rate: {sampling_rate}")

            # Run peak detection
            peaks, properties, peak_areas = run_peak_detection_function(self, profile_function=profile_function)
            
            if peaks is not None and properties is not None:
                # Store peak properties in app instance
                self.peaks = peaks
                self.peak_heights = properties['prominences']
                self.peak_widths = properties['widths']
                self.peak_left_ips = properties['left_ips']
                self.peak_right_ips = properties['right_ips']
                self.peak_width_heights = properties['width_heights']
                self.peak_areas = peak_areas  # Store peak areas
                
                # Update status
                self.preview_label.config(
                    text=f"Detected {len(peaks)} peaks",
                    foreground=self.theme_manager.get_color('success')
                )
                return True
            else:
                self.preview_label.config(
                    text="No peaks detected",
                    foreground=self.theme_manager.get_color('warning')
                )
                return False
                
        except Exception as e:
            self.show_error("Error during peak detection", str(e))
            return False
        finally:
            try:
                self.update_action_states()
            except Exception:
                pass
        
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
        Analyze double peaks using current parameters and update the visualization.
        """
        try:
            # Run double peak analysis
            double_peaks, figures = analyze_double_peaks_function(self)
            
            if double_peaks and figures:
                # Unpack figures
                selection_figure, grid_figure = figures
                
                # Create or update the tab for double peak selection
                selection_tab_name = "Double Peak Selection"
                selection_tab_exists = False
                
                # Remove existing selection tab if it exists
                tabs_to_remove = []
                for tab in self.plot_tab_control.tabs():
                    if self.plot_tab_control.tab(tab, "text") == selection_tab_name:
                        tabs_to_remove.append(tab)
                        selection_tab_exists = True
                
                for tab in tabs_to_remove:
                    self.plot_tab_control.forget(tab)
                    # Also remove from tab_figures
                    if selection_tab_name in self.tab_figures:
                        del self.tab_figures[selection_tab_name]
                
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
                
                # Remove existing grid tab if it exists
                tabs_to_remove = []
                for tab in self.plot_tab_control.tabs():
                    if self.plot_tab_control.tab(tab, "text") == grid_tab_name:
                        tabs_to_remove.append(tab)
                        grid_tab_exists = True
                
                for tab in tabs_to_remove:
                    self.plot_tab_control.forget(tab)
                    # Also remove from tab_figures
                    if grid_tab_name in self.tab_figures:
                        del self.tab_figures[grid_tab_name]
                
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
                
                # Update the right panel results summary
                if hasattr(self, 'results_summary'):
                    # Update only preview label and status - no need to change main summary for export
                    if hasattr(self, 'preview_label'):
                        self.preview_label.config(text=summary_text, foreground=self.theme_manager.get_color('success'))
                    if hasattr(self, 'status_indicator'):
                        self.status_indicator.set_text(summary_text)
                return True
            else:
                # If there was an error or no results, show a message
                self.preview_label.config(text="No double peaks found with current parameters", foreground="orange")
                return False
            
        except Exception as e:
            show_error_with_ui(self, "Error during double peak analysis", str(e))
            return False

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
        prominence_ratio = self.prominence_ratio.get()
        
        # Use the core module function
        result = calculate_peak_areas_function(
            self.peak_detector,
            self.filtered_signal,
            self.t_value,
            height_lim_factor,
            distance,
            rel_height,
            width_values,
            time_resolution=time_res,
            prominence_ratio=prominence_ratio  # Pass the prominence ratio parameter
        )
        
        if result:
            peak_area, start, end = result
            
            # Update results summary
            # Just update preview label for single area calculation
            self.preview_label.config(
                text=f"Peak Area: {peak_area:.2f}",
                foreground=self.theme_manager.get_color('success'))
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
        processing_message="Saving double peak information...",
        success_message="Double peak information saved successfully",
        error_message="Error saving double peak information"
    )
    def save_double_peak_information_to_csv(self):
        """
        Save detected double peak information to a file with configurable format.
        This is a UI wrapper around the core.file_export.save_double_peak_information_to_csv function.
        """
        from core.file_export import save_double_peak_information_to_csv as save_double_peaks
        return save_double_peaks(self)
            
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
            tabs_to_remove = []
            for tab in self.tab_control.tabs():
                tab_text = self.tab_control.tab(tab, "text")
                if tab_text in ["Double Peak Analysis", "Double Peak"]:
                    found_tab = True
                    if not double_peak_enabled:
                        # Mark tab for removal if double peak mode is disabled
                        tabs_to_remove.append(tab)
            
            # Remove tabs after iteration to avoid modifying collection during iteration
            for tab in tabs_to_remove:
                self.tab_control.forget(tab)
            
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

    # Removed deprecated update_results_summary; call the UI utility directly where needed

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
        Toggle between light and dark theme and update UI and plots accordingly.
        """
        new_theme = self.theme_manager.toggle_theme()

        # 1. Apply theme to UI elements and set matplotlib rcParams
        self.style = self.theme_manager.apply_theme(self)
        
        # Call our new method to update all sliders with the new theme colors
        self.theme_manager.update_sliders(self)

        # 2. Update specific UI Elements (Existing Code - Keep this)
        # 2. Update specific UI Elements (Existing Code - Keep this)
        # Update ScrolledText colors if it exists
        if hasattr(self, 'results_summary') and self.results_summary:
            self.results_summary.config(
                bg=self.theme_manager.get_color('card_bg'),
                fg=self.theme_manager.get_color('text'),
                insertbackground=self.theme_manager.get_color('text'),
                selectbackground=self.theme_manager.get_color('primary'),
                selectforeground='white',
                highlightcolor=self.theme_manager.get_color('border'),
                highlightbackground=self.theme_manager.get_color('border')
            )
            # Also update the scrollbar if it exists
            if hasattr(self.results_summary, 'vbar'):
                self.results_summary.vbar.configure(
                    bg=self.theme_manager.get_color('panel_bg'),
                    troughcolor=self.theme_manager.get_color('background'),
                    activebackground=self.theme_manager.get_color('primary')
                )

        # Update welcome label if it exists
        if hasattr(self, 'blank_tab'):
            for child in self.blank_tab.winfo_children():
                if isinstance(child, ttk.Label):
                    child.config(
                        foreground=self.theme_manager.get_color('text'),
                        background=self.theme_manager.get_color('background')
                    )
                elif isinstance(child, ttk.Frame):
                    # This is likely our welcome_container frame
                    # Use a recursive function to find all descendants since winfo_descendants() may not be available
                    def update_widget_theme(parent):
                        """Recursively update theme for all child widgets"""
                        for widget in parent.winfo_children():
                            # For labels, update text and background colors
                            if isinstance(widget, ttk.Label):
                                if 'text' in widget.configure():
                                    # Check for special labels to give them appropriate colors
                                    if 'Peak Analysis Tool' in widget.cget('text'):
                                        # Title label - use primary color
                                        widget.config(
                                            foreground=self.theme_manager.get_color('primary'),
                                            background=self.theme_manager.get_color('background')
                                        )
                                    elif widget.cget('text').startswith('v'):
                                        # Version label - use secondary color
                                        widget.config(
                                            foreground=self.theme_manager.get_color('secondary'),
                                            background=self.theme_manager.get_color('background')
                                        )
                                    elif widget.cget('text') == '💡':
                                        # Tip icon - use card background
                                        widget.config(
                                            background=self.theme_manager.get_color('card_bg')
                                        )
                                    elif 'Tip:' in widget.cget('text'):
                                        # Tip text - use card background
                                        widget.config(
                                            foreground=self.theme_manager.get_color('text'),
                                            background=self.theme_manager.get_color('card_bg')
                                        )
                                    else:
                                        # Regular labels - use text color and background
                                        widget.config(
                                            foreground=self.theme_manager.get_color('text'),
                                            background=self.theme_manager.get_color('background')
                                        )
                            # Update any image labels that have background
                            if isinstance(widget, ttk.Label) and 'image' in widget.configure():
                                if 'example_photo' in str(widget.cget('image')):
                                    # Example image - use card background
                                    widget.config(
                                        background=self.theme_manager.get_color('card_bg')
                                    )
                                elif 'icon_photo' in str(widget.cget('image')):
                                    # Icon - use regular background
                                    widget.config(
                                        background=self.theme_manager.get_color('background')
                                    )
                                    
                            # LabelFrames also need to be updated
                            if isinstance(widget, ttk.LabelFrame):
                                widget.config(style='TLabelframe')  # Apply theme style
                                
                            # Recursively update any child containers
                            if hasattr(widget, 'winfo_children'):
                                update_widget_theme(widget)
                    
                    # Call the recursive function on the welcome container
                    update_widget_theme(child)

        # Update Data Loading Tab - Dwell Time section explicitly
        if hasattr(self, 'explanation_frame'):
            self.explanation_frame.config(style='TFrame')
        if hasattr(self, 'explanation_label'):
            self.explanation_label.config(
                background=self.theme_manager.get_color('background'),
                foreground=self.theme_manager.get_color('text')
            )
        if hasattr(self, 'highlight_frame'):
            accent_bg = self.theme_manager.get_color('panel_bg') if new_theme == 'dark' else '#e6f2ff'
            self.highlight_frame.config(style='Accent.TFrame')
            try:
                self.highlight_frame.configure(background=accent_bg)
            except tk.TclError:
                 pass
            for child in self.highlight_frame.winfo_children():
                if isinstance(child, ttk.Label):
                    child.config(
                        background=accent_bg,
                        foreground=self.theme_manager.get_color('text')
                    )
        if hasattr(self, 'time_res_entry'):
             pass # Style handles ttk.Entry

        # === Update Peak Detection Tab ===
        if hasattr(self, 'peak_detection_main_canvas'):
            self.peak_detection_main_canvas.config(bg=self.theme_manager.get_color('background'))
        if hasattr(self, 'sigma_container'):
            self.sigma_container.config(style='TFrame')
        if hasattr(self, 'slider_frame'):
            self.slider_frame.config(style='TFrame')
            for child in self.slider_frame.winfo_children():
                if isinstance(child, ttk.Label):
                     child.config(
                         background=self.theme_manager.get_color('background'),
                         foreground=self.theme_manager.get_color('text')
                     )
        if hasattr(self, 'threshold_diagram_canvas'):
            canvas_bg_color = self.theme_manager.get_color('background')
            self.threshold_diagram_canvas.config(bg=canvas_bg_color)
            self._redraw_threshold_diagram()
        if hasattr(self, 'sigma_slider'):
            self.sigma_slider.config(
                bg=self.theme_manager.get_color('card_bg'),
                fg=self.theme_manager.get_color('text'),
                troughcolor=self.theme_manager.get_color('background')
            )
        if hasattr(self, 'manual_diagram_canvas'):
            canvas_bg_color = self.theme_manager.get_color('background')
            self.manual_diagram_canvas.config(bg=canvas_bg_color)
            self._redraw_manual_diagram()
        if hasattr(self, 'distance_slider'):
            self.distance_slider.config(
                bg=self.theme_manager.get_color('card_bg'),
                fg=self.theme_manager.get_color('text'),
                troughcolor=self.theme_manager.get_color('background')
            )
        if hasattr(self, 'rel_height_slider'):
            self.rel_height_slider.config(
                bg=self.theme_manager.get_color('card_bg'),
                fg=self.theme_manager.get_color('text'),
                troughcolor=self.theme_manager.get_color('background')
            )
            
        # Update double peak tab sliders
        for slider_name in ['min_slider', 'max_slider', 'min_amp_slider', 'max_amp_slider', 
                           'min_width_slider', 'max_width_slider']:
            if hasattr(self, slider_name):
                getattr(self, slider_name).config(
                    bg=self.theme_manager.get_color('card_bg'),
                    fg=self.theme_manager.get_color('text'),
                    troughcolor=self.theme_manager.get_color('background')
                )

        # === Update Double Peak Analysis Tab Histograms ===
        if hasattr(self, 'amp_hist_canvas') and hasattr(self, 'amp_hist_ax'):
            self._update_histogram_theme(self.amp_hist_canvas, self.amp_hist_ax)
            self.amp_hist_canvas.draw()
        if hasattr(self, 'width_hist_canvas') and hasattr(self, 'width_hist_ax'):
            self._update_histogram_theme(self.width_hist_canvas, self.width_hist_ax)
            self.width_hist_canvas.draw()
            
        # Update prominence diagram if it exists
        if hasattr(self, 'prominence_diagram_canvas'):
            self.prominence_diagram_canvas.config(bg=self.theme_manager.get_color('background'))
            self._draw_prominence_diagram()

        # === Update Preprocessing Tab ===
        if hasattr(self, 'raw_color_indicator'):
            raw_indicator_bg = '#bbbbbb' if new_theme == 'dark' else '#333333'
            self.raw_color_indicator.config(background=raw_indicator_bg)
        if hasattr(self, 'filter_color_indicator'):
            filter_indicator_bg = '#81D4FA' if new_theme == 'dark' else '#0078D7'
            self.filter_color_indicator.config(background=filter_indicator_bg)
        if hasattr(self, 'preprocessing_comparison_canvas'):
            self._redraw_preprocessing_comparison()

        # Persist theme selection
        try:
            self.prefs['theme'] = new_theme
            save_user_preferences(self.prefs)
        except Exception:
            pass

        # === Update General UI Elements within Tabs ===
        for tab in self.tab_control.winfo_children():
            if isinstance(tab, ttk.Frame):
                for frame in tab.winfo_children():
                    if isinstance(frame, ttk.LabelFrame):
                        self._update_frame_theme_elements(frame)
                        for child in frame.winfo_children():
                            if isinstance(child, ttk.Frame):
                                self._update_frame_theme_elements(child)
        # === End Update UI Elements ===

        # 3. Regenerate Existing Plots to Apply Full Theme
        # Get a copy of figure keys (tab names) *before* potentially modifying the dict
        existing_plot_tabs = list(self.tab_figures.keys())
        print(f"Theme toggled to '{new_theme}'. Regenerating plots for tabs: {existing_plot_tabs}")

        # Store the currently selected plot tab to re-select it later
        selected_tab_id = None
        if hasattr(self, 'plot_tab_control') and self.plot_tab_control.winfo_exists():
             try:
                  selected_tab_id = self.plot_tab_control.select()
                  # Get the text of the selected tab if needed for comparison later
                  # selected_tab_text = self.plot_tab_control.tab(selected_tab_id, "text")
             except tk.TclError: # Handle case where no tab is selected or control destroyed
                  selected_tab_id = None

        regenerated_tabs = set() # Keep track to avoid double regeneration (e.g., double peak)

        for tab_name in existing_plot_tabs:
             if tab_name in regenerated_tabs:
                 continue

             print(f"Attempting regeneration for plot tab: {tab_name}")
             # Ensure figure exists for this tab before trying to clear/replot
             if tab_name not in self.tab_figures or not isinstance(self.tab_figures[tab_name], Figure):
                  print(f"  Skipping regeneration for '{tab_name}', no valid figure found.")
                  continue

             try:
                # Use if/elif based on the tab name to call the correct replotting function
                # Add checks for necessary data before calling plot functions
                if tab_name == "Raw Data":
                    if hasattr(self, 'data') and self.data is not None:
                        self.plot_raw_data() # Replots raw data
                    else:
                        self._clear_plot_tab(tab_name) # Clear if no data

                elif tab_name == "Processed Data":
                    if hasattr(self, 'data') and self.data is not None:
                         # start_analysis creates the 'Processed Data' plot
                        self.start_analysis()
                    else:
                        self._clear_plot_tab(tab_name)

                elif tab_name == "Filtered Peaks":
                     if hasattr(self, 'filtered_signal') and self.filtered_signal is not None and \
                        hasattr(self, 'peaks') and self.peaks is not None:
                          # plot_filtered_peaks creates the grid
                          self.plot_filtered_peaks()
                     else:
                         self._clear_plot_tab(tab_name)

                elif tab_name == "Peak Analysis":
                     if hasattr(self, 'filtered_signal') and self.filtered_signal is not None and \
                        hasattr(self, 't_value') and self.t_value is not None:
                          self.plot_data() # Replots peak properties over time
                     else:
                          self._clear_plot_tab(tab_name)

                elif tab_name == "Peak Properties":
                     if hasattr(self, 'filtered_signal') and self.filtered_signal is not None:
                          self.plot_scatter() # Replots scatter correlations
                     else:
                          self._clear_plot_tab(tab_name)

                elif tab_name in ["Double Peak Selection", "Double Peak Grid"]:
                     # Check prerequisites: filtered signal exists AND double peak mode is on
                     if hasattr(self, 'filtered_signal') and self.filtered_signal is not None and \
                        hasattr(self, 'double_peak_analysis') and self.double_peak_analysis.get() == "1":
                          # This call regenerates both Selection and Grid plots/tabs
                          # It handles checks for enough peaks internally
                          self.analyze_double_peaks()
                          regenerated_tabs.add("Double Peak Selection")
                          regenerated_tabs.add("Double Peak Grid")
                          print(f"  Regenerated Double Peak plots.")
                     else:
                          # Clear both potential tabs if prerequisites aren't met
                          print(f"  Clearing Double Peak plots as prerequisites not met.")
                          self._clear_plot_tab("Double Peak Selection")
                          self._clear_plot_tab("Double Peak Grid")
                          regenerated_tabs.add("Double Peak Selection") # Mark as handled
                          regenerated_tabs.add("Double Peak Grid")

                # Add elif clauses here for any other plot tabs you might have

                else:
                     print(f"  No regeneration logic defined for tab: {tab_name}")
                     # Optionally, try a generic redraw for unknown tabs
                     # self._redraw_canvas_for_tab(tab_name)


             except Exception as e:
                 print(f"ERROR regenerating plot for tab '{tab_name}' during theme switch:")
                 traceback.print_exc() # Print detailed error

        # 4. Re-select the previously selected tab if possible
        if selected_tab_id and hasattr(self, 'plot_tab_control') and self.plot_tab_control.winfo_exists():
             try:
                  # Check if the tab still exists before selecting
                  if selected_tab_id in self.plot_tab_control.tabs():
                       self.plot_tab_control.select(selected_tab_id)
                       print(f"Re-selected tab ID: {selected_tab_id}")
                  else:
                       print(f"Tab ID {selected_tab_id} no longer exists after regeneration.")
             except tk.TclError as e:
                  print(f"Could not re-select tab ID {selected_tab_id} after theme change: {e}")


        # 5. Recreate the menu bar to update the theme toggle label
        from ui.components import create_menu_bar
        self.menu_bar = create_menu_bar(self)

        # 6. Update status message
        theme_name_str = "Dark" if new_theme == "dark" else "Light"
        self.preview_label.config(
            text=f"Switched to {theme_name_str} Theme",
            foreground=self.theme_manager.get_color('success')
        )

        return new_theme

    def _clear_plot_tab(self, tab_name):
        """Clears the figure associated with a tab and redraws its canvas."""
        if tab_name in self.tab_figures and isinstance(self.tab_figures[tab_name], Figure):
            print(f"Clearing figure for tab: {tab_name}")
            self.tab_figures[tab_name].clear() # Clear the figure object
            self._redraw_canvas_for_tab(tab_name) # Redraw the canvas to show blank state
        else:
             print(f"No figure found to clear for tab: {tab_name}")


    def _redraw_canvas_for_tab(self, tab_name):
        """Finds the canvas associated with a tab name and calls draw() on it."""
        if not hasattr(self, 'plot_tab_control') or not self.plot_tab_control.winfo_exists():
            return # Tab control doesn't exist

        try:
            for tab_id in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab_id, "text") == tab_name:
                    tab_frame = self.plot_tab_control.nametowidget(tab_id)
                    for widget in tab_frame.winfo_children():
                        if isinstance(widget, FigureCanvasTkAgg):
                            print(f"Redrawing canvas for tab: {tab_name}")
                            widget.draw()
                            return # Found and drew the canvas
                    print(f"No canvas found in tab frame for: {tab_name}")
                    return # No canvas found in this tab
            # print(f"Tab '{tab_name}' not found in plot_tab_control.")
        except tk.TclError as e:
            print(f"TclError finding/redrawing canvas for tab '{tab_name}': {e}")
        except Exception as e:
             print(f"Unexpected error redrawing canvas for tab '{tab_name}': {e}")


    def _update_frame_theme_elements(self, frame):
        """Helper method to update theme-aware elements within a frame"""
        # Update canvases
        for widget in frame.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.config(bg=self.theme_manager.get_color('background'))
                
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
                    bg=self.theme_manager.get_color('background'),
                    fg=self.theme_manager.get_color('text'),
                    troughcolor=self.theme_manager.get_color('panel_bg')
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

    # Method to redraw the threshold diagram with theme-appropriate colors
    def _redraw_threshold_diagram(self):
        """Redraws the threshold explanation diagram with current theme colors."""
        if not hasattr(self, 'threshold_diagram_canvas'):
            return # Canvas doesn't exist yet
            
        canvas = self.threshold_diagram_canvas
        canvas_bg = self.theme_manager.get_color('background') # Use main background
        text_color = self.theme_manager.get_color('text')
        signal_color = self.theme_manager.get_color('primary') # Use theme primary for signal
        
        # Define colors for threshold lines (using theme colors where appropriate)
        # For dark theme, use brighter colors
        if self.theme_manager.current_theme == 'dark':
            low_thresh_color = "#66BB6A"  # Brighter Green
            med_thresh_color = "#FFA726"  # Brighter Orange
            high_thresh_color = "#EF5350" # Brighter Red
            low_thresh_text = "σ=2"
            med_thresh_text = "σ=5"
            high_thresh_text = "σ=8"
        else:
            low_thresh_color = "#4CAF50"  # Standard Green
            med_thresh_color = "#FF9800"  # Standard Orange
            high_thresh_color = "#F44336" # Standard Red
            low_thresh_text = "σ=2"
            med_thresh_text = "σ=5"
            high_thresh_text = "σ=8"

        canvas.delete("all") # Clear previous drawing
        canvas.config(bg=canvas_bg) # Ensure background is set
        
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 50 or canvas_height < 20: # Check if canvas size is valid
             canvas_width = 380 # Default if not rendered yet
             canvas_height = 80
             
        baseline_y = canvas_height // 2 + 15

        # First draw background elements
        # --- Draw threshold lines ---
        low_thresh_y = baseline_y - 15
        med_thresh_y = baseline_y - 25
        high_thresh_y = baseline_y - 40

        # Low sigma
        canvas.create_line(10, low_thresh_y, canvas_width-10, low_thresh_y, 
                           fill=low_thresh_color, width=1, dash=(2, 2), tags="background")

        # Medium sigma
        canvas.create_line(10, med_thresh_y, canvas_width-10, med_thresh_y, 
                           fill=med_thresh_color, width=1, dash=(2, 2), tags="background")

        # High sigma
        canvas.create_line(10, high_thresh_y, canvas_width-10, high_thresh_y, 
                           fill=high_thresh_color, width=1, dash=(2, 2), tags="background")

        # --- Draw signal ---
        data_points = []
        np.random.seed(42)
        for x in range(10, canvas_width-10, 4):
            y = baseline_y
            if 70 <= x <= 90:
                peak_height = 35
                y = baseline_y - peak_height * np.exp(-0.02 * (x - 80) ** 2)
            elif 180 <= x <= 200:
                peak_height = 45
                y = baseline_y - peak_height * np.exp(-0.02 * (x - 190) ** 2)
            elif 270 <= x <= 290:
                peak_height = 25
                y = baseline_y - peak_height * np.exp(-0.02 * (x - 280) ** 2)
            y += np.random.normal(0, 3)
            data_points.append(x)
            data_points.append(int(y))
        canvas.create_line(data_points, fill=signal_color, width=2, smooth=True, tags="signal")

        # --- Draw markers ---
        for x_pos in [80, 190, 280]:
            canvas.create_oval(x_pos-3, low_thresh_y-3, x_pos+3, low_thresh_y+3, 
                               fill=low_thresh_color, outline="", tags="markers")
        for x_pos in [190, 280]:
            canvas.create_oval(x_pos-3, med_thresh_y-3, x_pos+3, med_thresh_y+3, 
                               fill=med_thresh_color, outline="", tags="markers")
        canvas.create_oval(190-3, high_thresh_y-3, 190+3, high_thresh_y+3, 
                           fill=high_thresh_color, outline="", tags="markers")

        # Draw tooltips last (on top)
        canvas.create_text(canvas_width-15, low_thresh_y-8, text=low_thresh_text, 
                           fill=low_thresh_color, anchor=tk.E, font=("TkDefaultFont", 8), tags="tooltip")
        canvas.create_text(canvas_width-15, med_thresh_y-8, text=med_thresh_text, 
                           fill=med_thresh_color, anchor=tk.E, font=("TkDefaultFont", 8), tags="tooltip")
        canvas.create_text(canvas_width-15, high_thresh_y-8, text=high_thresh_text, 
                           fill=high_thresh_color, anchor=tk.E, font=("TkDefaultFont", 8), tags="tooltip")

    # Method to redraw the manual parameters diagram with theme-appropriate colors
    def _redraw_manual_diagram(self):
        """Redraws the manual parameters explanation diagram with current theme colors."""
        if not hasattr(self, 'manual_diagram_canvas'):
            return # Canvas doesn't exist yet

        canvas = self.manual_diagram_canvas
        canvas_bg = self.theme_manager.get_color('background')
        text_color = self.theme_manager.get_color('text')
        signal_color = self.theme_manager.get_color('primary')

        # Define colors for indicators (brighter for dark theme)
        if self.theme_manager.current_theme == 'dark':
            dist_color = "#FF8A80"  # Brighter Red
            height_color = "#80CBC4" # Brighter Teal
            width_color = "#81D4FA" # Brighter Blue
        else:
            dist_color = "#FF6B6B"  # Standard Red
            height_color = "#4ECDC4" # Standard Teal
            width_color = "#45B7D1" # Standard Blue

        canvas.delete("all") # Clear previous drawing
        canvas.config(bg=canvas_bg) # Ensure background is set

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 50 or canvas_height < 20: # Check if canvas size is valid
            canvas_width = 380 # Default if not rendered yet
            canvas_height = 120
            
        baseline_y = 60

        # First draw the signal
        data_points = []
        np.random.seed(42)
        peaks = [
            {'x': 100, 'height': 50, 'width': 20},
            {'x': 250, 'height': 35, 'width': 25}
        ]
        for x in range(10, canvas_width-10, 2):
            y = baseline_y
            for peak in peaks:
                if abs(x - peak['x']) < peak['width'] * 2:
                    y -= peak['height'] * np.exp(-0.5 * ((x - peak['x']) / (peak['width']/2))**2)
            y += np.random.normal(0, 0.5)
            data_points.append(x)
            data_points.append(int(y))
        canvas.create_line(data_points, fill=signal_color, width=2, smooth=True, tags="signal")

        # Draw indicator lines
        # Distance
        distance_y = baseline_y + 15
        canvas.create_line(peaks[0]['x'], distance_y, peaks[1]['x'], distance_y,
                           fill=dist_color, width=1, dash=(2, 2), tags="indicator_lines")

        # Height
        rel_height_y = baseline_y - peaks[0]['height'] * 0.2
        canvas.create_line(peaks[0]['x'], baseline_y - peaks[0]['height'], peaks[0]['x'], rel_height_y,
                           fill=height_color, width=1, dash=(2, 2), tags="indicator_lines")

        # Width
        width_y = baseline_y + 5
        width_val = peaks[0]['width']
        canvas.create_line(peaks[0]['x'] - width_val, width_y, peaks[0]['x'] + width_val, width_y,
                           fill=width_color, width=1, dash=(2, 2), tags="indicator_lines")
        canvas.create_line(peaks[0]['x'] - width_val, width_y, peaks[0]['x'] - width_val, rel_height_y,
                           fill=width_color, width=1, dash=(2, 2), tags="indicator_lines")
        canvas.create_line(peaks[0]['x'] + width_val, width_y, peaks[0]['x'] + width_val, rel_height_y,
                           fill=width_color, width=1, dash=(2, 2), tags="indicator_lines")

        # Draw tooltips last (on top)
        canvas.create_text((peaks[0]['x'] + peaks[1]['x'])/2, distance_y + 10,
                           text="Distance between peaks", fill=text_color, 
                           font=("TkDefaultFont", 8), tags="tooltip")
        canvas.create_text(peaks[0]['x'], rel_height_y - 10,
                           text="Relative Height (0.8 = 80% from top)", fill=text_color, 
                           font=("TkDefaultFont", 8), tags="tooltip")
        canvas.create_text(peaks[0]['x'], width_y + 10,
                           text="Width Range", fill=text_color, 
                           font=("TkDefaultFont", 8), tags="tooltip")

    # Method to update the theme styling of a Matplotlib histogram
    def _update_histogram_theme(self, canvas, ax):
        """Applies the current theme colors to a histogram's figure and axes."""
        if canvas is None or ax is None:
            return
            
        fig = ax.get_figure()
        is_dark = self.theme_manager.current_theme == 'dark'
        
        # Colors
        bg_color = self.theme_manager.get_color('background')
        text_color = self.theme_manager.get_color('text')
        grid_color = self.theme_manager.get_color('border')
        spine_color = self.theme_manager.get_color('secondary')
        
        # Update figure and axes background
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        
        # Update spines
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
            
        # Update ticks and labels
        ax.tick_params(axis='x', colors=text_color, labelsize=6) # Keep labelsize small
        ax.tick_params(axis='y', colors=text_color, labelsize=6)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        
        # Update grid
        ax.grid(True, color=grid_color, alpha=0.3)
        
        # Update title (if exists)
        if ax.get_title():
            ax.title.set_color(text_color)
            
        # Redraw the canvas
        canvas.draw()

    # Method to redraw the preprocessing comparison diagram
    def _redraw_preprocessing_comparison(self):
        """Redraws the preprocessing comparison diagram with current theme colors."""
        if not hasattr(self, 'preprocessing_comparison_canvas'):
            return

        canvas = self.preprocessing_comparison_canvas
        canvas_bg = self.theme_manager.get_color('background') # Use background for this canvas
        text_color = self.theme_manager.get_color('text')
        is_dark = self.theme_manager.current_theme == 'dark'
        
        # Define colors based on theme
        raw_color = '#bbbbbb' if is_dark else '#333333'
        filtered_color = '#81D4FA' if is_dark else '#0078D7' # Lighter blue for dark
        axis_color = self.theme_manager.get_color('secondary')

        canvas.delete("all")
        canvas.config(bg=canvas_bg)

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 50 or canvas_height < 20:
            canvas_width = 380
            canvas_height = 80
            
        baseline_y = canvas_height // 2

        # First draw the axis (background)
        canvas.create_line(10, baseline_y, canvas_width - 10, baseline_y,
                           fill=axis_color, dash=(4, 4), width=1, tags="background")

        # Draw raw data
        raw_points = []
        np.random.seed(42)
        for x in range(10, canvas_width-10, 3):
            noise = np.random.normal(0, 6) if x % 9 != 0 else np.random.normal(0, 2)
            y = baseline_y - 15 * np.sin((x-10) / 30) + noise
            raw_points.append(x)
            raw_points.append(int(y))
        canvas.create_line(raw_points, fill=raw_color, width=1.5, smooth=False, tags="raw_data")

        # Draw filtered data
        filtered_points = []
        for x in range(10, canvas_width-10, 3):
            y = baseline_y - 15 * np.sin((x-10) / 30)
            filtered_points.append(x)
            filtered_points.append(int(y))
        canvas.create_line(filtered_points, fill=filtered_color, width=2, smooth=True, tags="filtered_data")

        # Draw tooltips last (on top)
        # Add tooltips for raw and filtered data
        canvas.create_text(30, baseline_y - 30, text="Raw Data", 
                          fill=raw_color, anchor=tk.W, font=("TkDefaultFont", 8), tags="tooltip")
        canvas.create_text(30, baseline_y + 20, text="Filtered Data", 
                          fill=filtered_color, anchor=tk.W, font=("TkDefaultFont", 8), tags="tooltip")

    @ui_action(
        processing_message="Toggling filtered peaks visibility...",
        success_message="Filtered peaks visibility updated",
        error_message="Error toggling filtered peaks visibility"
    )
    def toggle_filtered_peaks_visibility(self):
        """
        Toggle the visibility of peaks filtered by the prominence ratio threshold.
        
        When enabled, peaks that would be filtered out are shown in light red color
        in both the time-resolved and scatter plots, making it easier to visualize
        which peaks are being excluded by the current threshold setting.
        """
        try:
            # Get the new visibility state from the toggle button
            show_filtered = self.show_filtered_peaks.get()
            
            # Update status indicator
            if show_filtered:
                self.status_indicator.set_text("Showing filtered peaks in light red")
            else:
                self.status_indicator.set_text("Filtered peaks hidden")
            
            # Check if we have any plots to update
            tab_to_update = None
            
            # Identify which tab is currently visible
            for tab_name in ["Peak Analysis", "Peak Properties"]:
                if tab_name in self.tab_figures:
                    tab_to_update = tab_name
                    break
            
            if tab_to_update:
                # Re-plot the data with the new visibility setting
                if tab_to_update == "Peak Analysis":
                    self.plot_data()
                elif tab_to_update == "Peak Properties":
                    self.plot_scatter()
                
                # Show a message in the preview label
                self.preview_label.config(
                    text=f"Filtered peaks {'shown' if show_filtered else 'hidden'} in plots",
                    foreground=self.theme_manager.get_color('success')
                )
            else:
                # No plots to update yet
                self.preview_label.config(
                    text="No plots to update. Generate plots first using the analysis buttons.",
                    foreground=self.theme_manager.get_color('warning')
                )
            
            return True
            
        except Exception as e:
            self.show_error("Error toggling filtered peaks visibility", str(e))
            return False

    @ui_action(
        processing_message="Showing tooltip popup...",
        success_message="Tooltip popup shown",
        error_message="Error showing tooltip popup"
    )
    def show_tooltip_popup(self, title, text):
        """
        Show a popup window with information text.
        
        Parameters
        ----------
        title : str
            The title of the popup window
        text : str
            The text to display in the popup
            
        Returns
        -------
        None
        """
        popup = tk.Toplevel(self)  # Use self instead of self.root since Application inherits from tk.Tk
        popup.title(title)
        popup.geometry("400x300")
        popup.resizable(True, True)
        popup.transient(self)  # Use self instead of self.root
        popup.grab_set()
        
        # Apply theme
        popup.configure(bg=self.theme_manager.get_color('background'))
        
        # Add text
        frame = ttk.Frame(popup)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(frame, wrap=tk.WORD, bg=self.theme_manager.get_color('background'),
                             fg=self.theme_manager.get_color('text'), relief=tk.FLAT,
                             highlightthickness=0, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)
        
        # Add close button
        button_frame = ttk.Frame(popup)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        close_button = ttk.Button(button_frame, text="Close", command=popup.destroy)
        close_button.pack(side=tk.RIGHT, padx=10)
        
        # Position the popup window relative to the root window
        root_x = self.winfo_rootx()  # Use self instead of self.root
        root_y = self.winfo_rooty()  # Use self instead of self.root
        root_width = self.winfo_width()  # Use self instead of self.root
        root_height = self.winfo_height()  # Use self instead of self.root
        
        popup_width = 400
        popup_height = 300
        
        x = root_x + (root_width - popup_width) // 2
        y = root_y + (root_height - popup_height) // 2
        
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
        
        return popup

    def get_peak_filter_stats(self):
        """
        Returns (total_peaks, filtered_out, filtered_kept) using the current parameters and prominence ratio.
        Uses the same logic as plot_data for consistency.
        """
        import numpy as np
        from core.peak_analysis_utils import find_peaks_with_window
        if self.filtered_signal is None:
            return 0, 0, 0
        width_values = self.width_p.get().strip().split(',')
        rate = self.time_resolution.get() if hasattr(self.time_resolution, 'get') else self.time_resolution
        if rate <= 0:
            rate = 0.0001
        sampling_rate = 1 / rate
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        print(f"[DEBUG][get_peak_filter_stats] width_p (ms): {width_values}, width_p (samples): {width_p}, sampling_rate: {sampling_rate}")
        # Only use the current prominence ratio value, don't fall back to a default value
        prominence_ratio = self.prominence_ratio.get()
        all_peaks, all_properties = find_peaks_with_window(
            self.filtered_signal,
            width=width_p,
            prominence=self.height_lim.get(),
            distance=self.distance.get(),
            rel_height=self.rel_height.get(),
            prominence_ratio=prominence_ratio
        )
        # The peaks returned by find_peaks_with_window already have the filter applied
        # But we need to calculate how many were filtered out, so we need to get all peaks first
        # without the filter
        from scipy.signal import find_peaks, peak_widths
        # Find all peaks without prominence_ratio filtering
        all_unfiltered_peaks, all_unfiltered_properties = find_peaks(
            self.filtered_signal,
            width=width_p,
            prominence=self.height_lim.get(),
            distance=self.distance.get(),
            rel_height=self.rel_height.get()
        )
        
        if len(all_unfiltered_peaks) > 0:
            # Calculate widths and add to properties
            width_results = peak_widths(self.filtered_signal, all_unfiltered_peaks, rel_height=self.rel_height.get())
            all_unfiltered_properties['widths'] = width_results[0]
            all_unfiltered_properties['width_heights'] = width_results[1]
            all_unfiltered_properties['left_ips'] = width_results[2]
            all_unfiltered_properties['right_ips'] = width_results[3]
        
        # Calculate how many peaks were filtered by prominence ratio
        total_peaks = len(all_unfiltered_peaks)
        filtered_kept = len(all_peaks)  # already filtered by find_peaks_with_window
        filtered_out = total_peaks - filtered_kept
        
        return total_peaks, filtered_out, filtered_kept

    def on_apply_prominence_ratio(self):
        """
        Handler for the 'Apply' button in the peak analysis tab.
        Updates the plots, filtered peaks feedback, and results summary based on the current prominence ratio.
        """
        try:
            # Update the main analysis plot
            self.plot_data()

            # Use unified logic for feedback and summary
            total_peaks, filtered_out, filtered_kept = self.get_peak_filter_stats()
            if total_peaks > 0:
                filtered_percentage = (filtered_out / total_peaks) * 100
                msg = f"Filtered out {filtered_out} of {total_peaks} peaks ({filtered_percentage:.1f}%)"
                self.filtered_peaks_feedback.config(text=msg, foreground="blue")
                
                # Update peak_detector.all_peaks_count to ensure results summary uses the same total count
                if hasattr(self, 'peak_detector'):
                    self.peak_detector.all_peaks_count = total_peaks
                else:
                    print("Warning: peak_detector not found when setting all_peaks_count in on_apply_prominence_ratio")
            else:
                self.filtered_peaks_feedback.config(text="No peaks detected.", foreground="red")
                if hasattr(self, 'peak_detector'):
                    self.peak_detector.all_peaks_count = 0

            # Update the results summary (will include filtering info)
            update_results_summary_with_ui(self, events=filtered_kept, context='peak_analysis')
        except Exception as e:
            self.filtered_peaks_feedback.config(text=f"Error: {e}", foreground="red")
            update_results_summary_with_ui(self, context='peak_analysis')

    @ui_action(
        processing_message="Analyzing time-resolved data...",
        success_message="Time-resolved data analyzed successfully",
        error_message="Error analyzing time-resolved data"
    )
    def analyze_time_resolved(self):
        """
        Analyze time-resolved data using current parameters and update the visualization.
        """
        try:
            # Validate parameters first
            from core.data_utils import validate_peak_params
            ok, msg = validate_peak_params(
                prominence_threshold=self.height_lim.get(),
                distance=self.distance.get(),
                rel_height=self.rel_height.get(),
                width_ms=self.width_p.get(),
                time_resolution=self.time_resolution.get() if hasattr(self.time_resolution, 'get') else self.time_resolution,
                prominence_ratio=self.prominence_ratio.get() if hasattr(self, 'prominence_ratio') else None,
            )
            if not ok:
                try:
                    if hasattr(self, 'validation_label'):
                        self.validation_label.config(text=msg, foreground=self.theme_manager.get_color('error'))
                    else:
                        self.validation_label = ttk.Label(self, text=msg, foreground=self.theme_manager.get_color('error'))
                        self.validation_label.pack(side=tk.TOP, anchor='w', padx=8)
                except Exception:
                    pass
                self.preview_label.config(text=f"Parameter error: {msg}", foreground=self.theme_manager.get_color('error'))
                if hasattr(self, 'status_indicator'):
                    self.status_indicator.set_state('warning')
                    self.status_indicator.set_text(msg)
                return False

            # Prefer pure analysis function with explicit parameters
            time_res = self.time_resolution.get() if hasattr(self.time_resolution, 'get') else self.time_resolution
            results = analyze_time_resolved_pure_function(
                signal=self.filtered_signal,
                time_values=self.t_value,
                width_ms=self.width_p.get(),
                time_resolution=time_res,
                prominence_threshold=self.height_lim.get(),
                distance=self.distance.get(),
                rel_height=self.rel_height.get(),
                prominence_ratio=self.prominence_ratio.get(),
            )
            
            if results:
                peaks, areas, intervals = results
                
                # Update results summary
                summary_text = (
                    f"Time-Resolved Analysis Results:\n"
                    f"Found {len(peaks)} peaks\n"
                    f"Average peak area: {np.mean(areas):.2f} ± {np.std(areas):.2f}\n"
                    f"Average interval: {np.mean(intervals)*1000:.1f} ± {np.std(intervals)*1000:.1f} ms\n"
                    f"Parameters used:\n"
                    f"- Prominence ratio: {self.prominence_ratio.get():.2f}\n"
                    f"- Min peak distance: {self.min_peak_distance.get()*1000:.1f} ms\n"
                )
                
                # Update the right panel results summary
                if hasattr(self, 'results_summary'):
                    update_results_summary_with_ui(
                        self,
                        events=len(peaks),
                        peak_areas=areas,
                        peak_intervals=intervals,
                        preview_text=summary_text
                    )
                
                return True
            else:
                # If there was an error or no results, show a message
                self.preview_label.config(text="No peaks found with current parameters", foreground="orange")
                return False
                
        except Exception as e:
            show_error_with_ui(self, "Error during time-resolved analysis", str(e))
            return False

    @ui_action(
        processing_message="Exporting peak properties...",
        success_message="Peak properties exported successfully",
        error_message="Error exporting peak properties"
    )
    def export_peak_properties(self):
        """Export peak properties to a CSV file."""
        try:
            # Get file path from user
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Peak Properties"
            )
            
            if not file_path:  # User cancelled
                self.status_indicator.set_state('idle')
                self.status_indicator.set_text("Export cancelled")
                return False

            # Call the core export function (assuming it exists)
            success = save_peak_information_to_csv_function(self, file_path)
            
            if success:
                # Update status indicators
                summary_text = f"Exported peak properties to {os.path.basename(file_path)}"
                if hasattr(self, 'preview_label'):
                    self.preview_label.config(
                        text=summary_text,
                        foreground=self.theme_manager.get_color('success')
                    )
                if hasattr(self, 'status_indicator'):
                    self.status_indicator.set_text(summary_text)
                return True
            else:
                self.show_error("Export Error", "Failed to export peak properties")
                return False
                
        except Exception as e:
            self.show_error("Export Error", str(e))
            return False

    def _draw_prominence_diagram(self):
        """Internal method to draw the prominence diagram with current theme colors"""
        if not hasattr(self, 'prominence_diagram_canvas'):
            return
            
        canvas = self.prominence_diagram_canvas
        canvas.delete("all")
        
        # Colors - adapt to current theme
        text_color = self.theme_manager.get_color('text')
        signal_color = self.theme_manager.get_color('primary')
        
        # Choose colors based on theme
        if self.theme_manager.current_theme == 'dark':
            prominence_color = "#66BB6A"  # Brighter Green for dark theme
            height_color = "#FFA726"      # Brighter Orange for dark theme
            subpeak_color = "#EF5350"     # Brighter Red for dark theme
        else:
            prominence_color = "#4CAF50"  # Standard Green for light theme
            height_color = "#FF9800"      # Standard Orange for light theme
            subpeak_color = "#F44336"     # Standard Red for light theme
        
        # Draw axis
        canvas.create_line(10, 90, 370, 90, fill=text_color, dash=(2,2))
        canvas.create_text(15, 95, text="0", fill=text_color, anchor="nw")
        
        # Draw a main peak
        peak_x = 180
        peak_height = 60
        peak_y = 90 - peak_height  # 90 is baseline
        
        # Draw main peak
        points = []
        for x in range(50, 320, 5):
            y = 90 - peak_height * np.exp(-0.0015 * (x - peak_x) ** 2) 
            points.append(x)
            points.append(int(y))
        
        canvas.create_line(points, fill=signal_color, width=2, smooth=True)
        
        # Draw a subpeak
        subpeak_x = 180
        subpeak_height = 20
        sub_points = []
        for x in range(150, 210, 2):
            y = peak_y - subpeak_height * np.exp(-0.005 * (x - subpeak_x) ** 2)
            sub_points.append(x)
            sub_points.append(int(y))
        
        canvas.create_line(sub_points, fill=subpeak_color, width=2, smooth=True)
        canvas.create_text(150, 20, text="Subpeak (filtered out)", fill=subpeak_color)
        
        # Draw prominence and height measurements for subpeak
        left_ref_x = 130
        right_ref_x = 230
        
        # Main peak reference lines
        peak_base_y = 90  # Baseline
        
        # Subpeak reference lines
        subpeak_base_y = peak_y  # Subpeak's base is on the main peak
        subpeak_top_y = peak_y - subpeak_height
        
        # Draw measurement arrows
        arrow_x = 350
        
        # Peak height (from baseline to peak)
        canvas.create_line([arrow_x, peak_base_y, arrow_x, peak_y], 
                          fill=height_color, arrow="last", width=1.5)
        canvas.create_text(arrow_x+5, (peak_base_y+peak_y)/2, 
                          text="Peak\nHeight", fill=height_color, anchor="w")
        
        # Subpeak prominence (from subpeak base to top)
        canvas.create_line([arrow_x-20, subpeak_base_y, arrow_x-20, subpeak_top_y], 
                          fill=prominence_color, arrow="last", width=1.5)
        canvas.create_text(arrow_x-15, (subpeak_base_y+subpeak_top_y)/2, 
                          text="Prominence", fill=prominence_color, anchor="w")
        
        # Ratio explanation
        canvas.create_text(
            20, 
            110, 
            text="Ratio = Prominence / Peak Height  (Keep peaks with ratio ≥ threshold)", 
            fill=text_color, 
            anchor="nw",
            font=("TkDefaultFont", 8)
        )


# Your main program code goes here

if __name__ == "__main__":
    app = Application()
    app.mainloop()