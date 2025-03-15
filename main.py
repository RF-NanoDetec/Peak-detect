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
import os
from PIL import Image, ImageTk

# Local imports
from config.settings import Config
# Import performance utilities
from utils.performance import profile_function, get_memory_usage
# Import core analysis utilities
from core.peak_analysis_utils import find_peaks_with_window, find_nearest, timestamps_to_seconds, adjust_lowpass_cutoff 
# Import all plotting functions directly
from plotting.raw_data import plot_raw_data as plot_raw_data_function
from plotting.data_processing import start_analysis as start_analysis_function
from plotting.peak_visualization import run_peak_detection as run_peak_detection_function
from plotting.peak_visualization import plot_filtered_peaks as plot_filtered_peaks_function
from plotting.peak_visualization import show_next_peaks as show_next_peaks_function
from plotting.analysis_visualization import plot_data as plot_data_function
from plotting.analysis_visualization import plot_scatter as plot_scatter_function
from core.peak_detection import PeakDetector, calculate_auto_threshold
from ui import ThemeManager, create_tooltip, StatusIndicator


# Set default seaborn style
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.2)
sns.set_context("notebook", rc={"lines.linewidth": 1.0})


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        # Initialize logger before other components
        self.setup_performance_logging()
        
        # Application title and window setup
        self.title("Peak Analysis Tool")
        self.geometry("1920x1080")
        
        # Initialize theme manager with light theme as default
        self.theme_manager = ThemeManager(theme_name='light')
        self.style = self.theme_manager.apply_theme(self)
        
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
        self.normalization_factor = tk.DoubleVar(value=1.0)  # Changed from IntVar for more precise control
        self.start_time = tk.StringVar(value="0:00")
        self.big_counts = tk.IntVar(value=100)  # Renamed to Biggest Peaks
        self.height_lim = tk.DoubleVar(value=20)  # Renamed to Counts Threshold
        self.distance = tk.IntVar(value=30)  # Renamed to Min. Distance Peaks
        self.rel_height = tk.DoubleVar(value=0.85)
        self.width_p = tk.StringVar(value="1,200")
        self.cutoff_value = tk.DoubleVar()
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
        self.create_menu_bar()

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
        logging.basicConfig(
            filename='performance.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PeakAnalysis')

    def calculate_auto_threshold(self):
        if self.filtered_signal is not None:
            try:
                # Update status indicator
                self.status_indicator.set_state('processing')
                self.status_indicator.set_text("Calculating threshold...")
                self.update_idletasks()
                
                # Use the new module function
                suggested_threshold = calculate_auto_threshold(self.filtered_signal)
                
                self.height_lim.set(suggested_threshold)
                
                # Update status to success
                self.status_indicator.set_state('success')
                self.status_indicator.set_text(f"Threshold set to {suggested_threshold:.1f}")
                
                self.preview_label.config(
                    text=f"Threshold automatically set to {suggested_threshold:.1f} (5σ)", 
                    foreground=self.theme_manager.get_color('success')
                )
            except Exception as e:
                # Update status to error
                self.status_indicator.set_state('error')
                self.status_indicator.set_text("Error calculating threshold")
                self.show_error("Error calculating auto threshold", e)
        else:
            # Update status to warning
            self.status_indicator.set_state('warning')
            self.status_indicator.set_text("No data available")
            
            self.preview_label.config(
                text="Please run analysis first", 
                foreground=self.theme_manager.get_color('error')
            )

    # Create all the GUI widgets
    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

        # Create left control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Add status indicator at the top
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Status indicator
        self.status_indicator = StatusIndicator(status_frame, theme_manager=self.theme_manager)
        self.status_indicator.pack(fill=tk.X, padx=5, pady=5)
        
        # Create notebook (tabbed interface) for controls
        self.tab_control = ttk.Notebook(control_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True)

        # === Data Loading Tab ===
        data_loading_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(data_loading_tab, text="Data Loading")

        # File mode selection frame
        file_mode_frame = ttk.LabelFrame(data_loading_tab, text="File Mode")
        file_mode_frame.pack(fill=tk.X, padx=5, pady=5)

        # Radio buttons for file mode
        ttk.Radiobutton(
            file_mode_frame, 
            text="Single File", 
            variable=self.file_mode, 
            value="single",
            command=self.on_file_mode_change
        ).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Radiobutton(
            file_mode_frame, 
            text="Batch Mode", 
            variable=self.file_mode, 
            value="batch",
            command=self.on_file_mode_change
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Add tooltips for file mode selection
        self.add_tooltip(
            file_mode_frame,
            "Choose between single file analysis or batch processing of multiple files"
        )

        # File selection frame
        file_frame = ttk.LabelFrame(data_loading_tab, text="File Selection")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        # Browse button with styled appearance
        self.browse_button = ttk.Button(
            file_frame, 
            text="Load File", 
            command=self.browse_file,
            style="Primary.TButton"  # Apply primary button style
        )
        self.browse_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.file_name_label = ttk.Label(file_frame, text="No file selected")
        self.file_name_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Timestamps entry for batch mode
        self.timestamps_label = ttk.Label(file_frame, text="Timestamps:")
        self.timestamps_entry = ttk.Entry(file_frame, textvariable=self.batch_timestamps)

        # Initially hide timestamps widgets
        self.timestamps_label.pack_forget()
        self.timestamps_entry.pack_forget()

        # Add tooltips for file selection controls
        self.add_tooltip(
            self.browse_button,
            "Click to select a data file (single mode) or folder (batch mode)"
        )

        self.add_tooltip(
            self.timestamps_entry,
            "Enter timestamps for batch files in format 'MM:SS,MM:SS,...'\nExample: '00:00,01:30,03:00'"
        )

        # Protocol information frame
        protocol_frame = ttk.LabelFrame(data_loading_tab, text="Protocol Information")
        protocol_frame.pack(fill=tk.X, padx=5, pady=5)

        # Protocol information entries
        protocol_entries = [
            ("Start Time:", self.protocol_start_time),
            ("Particle:", self.protocol_particle),
            ("Concentration:", self.protocol_concentration),
            ("Stamp:", self.protocol_stamp),
            ("Laser Power:", self.protocol_laser_power),
            ("Setup:", self.protocol_setup)
        ]

        # Create protocol entries first
        for row, (label_text, variable) in enumerate(protocol_entries):
            ttk.Label(protocol_frame, text=label_text).grid(row=row, column=0, padx=5, pady=2, sticky="w")
            ttk.Entry(protocol_frame, textvariable=variable).grid(row=row, column=1, padx=5, pady=2, sticky="ew")

        # Protocol tooltips
        protocol_tooltips = {
            "Start Time": "Enter the experiment start time during the day in HH:MM:SS format (e.g., '13:30:00')",
            "Particle": "Enter the type of particle or sample being analyzed",
            "Concentration": "Enter the sample concentration",
            "Stamp": "Enter any lithographic stamp name or identifier example: 'tripple-block'",
            "Laser Power": "Enter the laser power settings used as ND filter",
            "Setup": "Enter the experimental setup configuration example: 'Prototype, Old Ladom'",
            "Notes": "Enter any additional notes or observations about the experiment"
        }

        # Now apply tooltips after creating the widgets
        for row, (label_text, _) in enumerate(protocol_entries):
            label_widget = protocol_frame.grid_slaves(row=row, column=0)[0]
            entry_widget = protocol_frame.grid_slaves(row=row, column=1)[0]
            
            tooltip_text = protocol_tooltips.get(label_text.rstrip(':'), "")
            self.add_tooltip(label_widget, tooltip_text)
            self.add_tooltip(entry_widget, tooltip_text)

        # Notes field
        ttk.Label(protocol_frame, text="Notes:").grid(row=len(protocol_entries), column=0, padx=5, pady=2, sticky="w")
        notes_entry = ttk.Entry(protocol_frame, textvariable=self.protocol_notes)
        notes_entry.grid(row=len(protocol_entries), column=1, padx=5, pady=2, sticky="ew")

        # Add tooltip for notes field
        notes_label = protocol_frame.grid_slaves(row=len(protocol_entries), column=0)[0]
        self.add_tooltip(
            notes_label,
            "Enter any additional notes or observations about the experiment"
        )
        self.add_tooltip(
            notes_entry,
            "Enter any additional notes or observations about the experiment"
        )

        # Configure grid columns
        protocol_frame.columnconfigure(1, weight=1)

        # === Preprocessing Tab ===
        preprocessing_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(preprocessing_tab, text="Preprocessing")

        # Signal Filtering Frame
        filtering_frame = ttk.LabelFrame(preprocessing_tab, text="Signal Filtering")
        filtering_frame.pack(fill=tk.X, padx=5, pady=5)

        # Cutoff Frequency frame with auto-calculate button
        cutoff_frame = ttk.Frame(filtering_frame)
        cutoff_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(cutoff_frame, text="Cutoff Frequency (Hz)").pack(side=tk.LEFT)
        cutoff_entry = ttk.Entry(cutoff_frame, textvariable=self.cutoff_value, width=10)
        cutoff_entry.pack(side=tk.LEFT, padx=5)

        auto_cutoff_button = ttk.Button(
            cutoff_frame, 
            text="Auto Calculate", 
            command=self.calculate_auto_cutoff_frequency
        )
        auto_cutoff_button.pack(side=tk.LEFT, padx=5)

        # Parameters for auto calculation
        auto_params_frame = ttk.LabelFrame(filtering_frame, text="Auto Calculation Parameters")
        auto_params_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(auto_params_frame, text="Biggest Peaks").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        big_counts_entry = ttk.Entry(auto_params_frame, textvariable=self.big_counts)
        big_counts_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(auto_params_frame, text="Normalization Factor").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        norm_entry = ttk.Entry(
            auto_params_frame, 
            textvariable=self.normalization_factor,
            validate='key',
            validatecommand=(self.register(self.validate_float), '%P')
        )
        norm_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        # Action Buttons - Reordered
        action_frame = ttk.Frame(preprocessing_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            action_frame,
            text="View Raw Data",  # First button
            command=self.plot_raw_data
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame,
            text="Apply Filtering",  # Second button
            command=self.start_analysis
        ).pack(side=tk.LEFT, padx=5)

        # Add tooltips
        self.add_tooltip(
            cutoff_entry,
            "Frequency cutoff for the Butterworth low-pass filter (Hz)\nSet to 0 for automatic calculation"
        )

        self.add_tooltip(
            auto_cutoff_button,
            "Calculate optimal cutoff frequency based on peak widths"
        )

        self.add_tooltip(
            big_counts_entry,
            "Threshold for identifying largest peaks\nUsed for automatic cutoff calculation"
        )

        self.add_tooltip(
            norm_entry,
            "Factor for normalizing signal amplitude\nTypically between 0.1 and 10"
        )

        # Configure grid weights
        auto_params_frame.columnconfigure(1, weight=1)

        # === Peak Detection Tab ===
        peak_detection_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(peak_detection_tab, text="Peak Detection")

        # Peak Parameters Frame
        peak_params_frame = ttk.LabelFrame(peak_detection_tab, text="Peak Parameters")
        peak_params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Threshold frame with auto-calculate button
        row = 0
        threshold_frame = ttk.Frame(peak_params_frame)
        threshold_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=2)

        ttk.Label(threshold_frame, text="Counts Threshold").pack(side=tk.LEFT)
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.height_lim, width=10)
        threshold_entry.pack(side=tk.LEFT, padx=5)

        auto_calc_button = ttk.Button(
            threshold_frame, 
            text="Auto Calculate", 
            command=self.calculate_auto_threshold
        )
        auto_calc_button.pack(side=tk.LEFT, padx=5)

        # Add tooltip for Auto Calculate button
        self.add_tooltip(
            auto_calc_button,
            "Automatically calculate optimal threshold based on 5σ (sigma) of the filtered signal"
        )

        # Other peak parameters
        row += 1
        ttk.Label(peak_params_frame, text="Min. Distance Peaks").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        distance_entry = ttk.Entry(peak_params_frame, textvariable=self.distance)
        distance_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

        row += 1
        ttk.Label(peak_params_frame, text="Relative Height").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        rel_height_entry = ttk.Entry(peak_params_frame, textvariable=self.rel_height)
        rel_height_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

        row += 1
        ttk.Label(peak_params_frame, text="Width Range (ms)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        width_entry = ttk.Entry(peak_params_frame, textvariable=self.width_p)
        width_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

        # Action Buttons Frame
        action_frame = ttk.Frame(peak_detection_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(
            action_frame, 
            text="Detect Peaks",
            command=self.run_peak_detection
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame, 
            text="View Individual Peaks",
            command=self.plot_filtered_peaks
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame, 
            text="Next Peaks →",  # Added arrow for better UX
            command=self.show_next_peaks
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame, 
            text="Quick Save Results",
            command=self.save_peak_information_to_csv
        ).pack(side=tk.LEFT, padx=5)

        # Add tooltips
        self.add_tooltip(
            action_frame.winfo_children()[-2],  # Next Peaks button
            "Show next set of individual peaks"
        )

        # Configure grid weights
        peak_params_frame.columnconfigure(1, weight=1)

        # Add tooltips for better user guidance
        self.add_tooltip(
            threshold_entry,
            "Minimum height threshold for peak detection"
        )
        self.add_tooltip(
            distance_entry,
            "Minimum number of points between peaks"
        )
        self.add_tooltip(
            rel_height_entry,
            "Relative height from peak maximum for width calculation (0-1)"
        )
        self.add_tooltip(
            width_entry,
            "Expected peak width range in milliseconds (min,max)"
        )
        self.add_tooltip(
            action_frame.winfo_children()[0],  # Detect Peaks button
            "Run peak detection algorithm with current parameters"
        )
        self.add_tooltip(
            action_frame.winfo_children()[1],  # View Individual Peaks button
            "Display detailed view of selected individual peaks"
        )
        self.add_tooltip(
            action_frame.winfo_children()[2],  # Quick Save Results button
            "Show next set of individual peaks"
        )
        self.add_tooltip(
            action_frame.winfo_children()[-1],  # Next Peaks button
            "Save current peak detection results to CSV file"
        )

        # === Peak Analysis Tab ===
        peak_analysis_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(peak_analysis_tab, text="Peak Analysis")

        # Analysis Options Frame
        analysis_options_frame = ttk.LabelFrame(peak_analysis_tab, text="Analysis Options")
        analysis_options_frame.pack(fill=tk.X, padx=5, pady=5)

        # Time-resolved analysis button (first)
        ttk.Button(
            analysis_options_frame,
            text="Time-Resolved Analysis",  # Changed from "Plot Peak Analysis"
            command=self.plot_data
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Peak properties correlation button (second)
        ttk.Button(
            analysis_options_frame,
            text="Peak Property Correlations",  # Changed from "Plot Peak Properties"
            command=self.plot_scatter
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Export Frame
        export_frame = ttk.LabelFrame(peak_analysis_tab, text="Export Options")
        export_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            export_frame,
            text="Export Peak Data to CSV",  # Changed from "Save Peak Information"
            command=self.save_peak_information_to_csv
        ).pack(side=tk.LEFT, padx=5, pady=5)

        # Results Frame
        results_frame = ttk.LabelFrame(peak_analysis_tab, text="Results Summary")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add tooltips for better user guidance
        self.add_tooltip(
            analysis_options_frame.winfo_children()[0],  # Time-Resolved Analysis button
            "Display peak properties changes over time and throughput analysis"
        )
        self.add_tooltip(
            analysis_options_frame.winfo_children()[1],  # Peak Property Correlations button
            "Display correlation plots between peak width, height, and area"
        )
        self.add_tooltip(
            export_frame.winfo_children()[0],  # Export button
            "Save all peak information to a CSV file for further analysis"
        )

        # Results section
        results_frame = ttk.LabelFrame(control_frame, text="Analysis Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_summary = ScrolledText(
            results_frame, 
            height=10,
            wrap=tk.WORD,
            bg=self.theme_manager.get_color('card_bg'),
            fg=self.theme_manager.get_color('text'),
            insertbackground=self.theme_manager.get_color('text'),  # Cursor color
            font=self.theme_manager.get_font('default'),
            state=tk.DISABLED
        )
        self.results_summary.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Clear results button
        ttk.Button(
            results_frame,
            text="Clear Results",
            command=lambda: self.update_results_summary(preview_text="")
        ).pack(pady=5)

        # Progress bar with green color
        self.progress = ttk.Progressbar(
            control_frame, 
            mode='determinate',
            style='Green.Horizontal.TProgressbar'
        )
        self.progress.pack(fill=tk.X, padx=5, pady=5)

        # Create custom style for green progress bar
        self.style.configure(
            'Green.Horizontal.TProgressbar',
            background='green',
            troughcolor='light gray'
        )

        # Add this helper method to manage progress bar updates:
        def update_progress(self, value=0, maximum=None):
            """Update progress bar with optional maximum value"""
            try:
                if maximum is not None:
                    self.progress['maximum'] = maximum
                self.progress['value'] = value
                self.update_idletasks()
                
                # Reset to zero if completed
                if value >= self.progress['maximum']:
                    self.after(500, lambda: self.update_progress_bar(0))  # Reset after 500ms
            except Exception as e:
                print(f"Error updating progress bar: {e}")

        # Preview Frame with Plot Tabs on the right
        preview_frame = ttk.Frame(main_frame)
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=0)

        # Tab Control for Multiple Plots on the right
        self.plot_tab_control = ttk.Notebook(preview_frame)
        self.plot_tab_control.grid(row=0, column=0, sticky="nsew")

        # Create an empty frame with fixed size instead of blank image
        self.blank_tab = ttk.Frame(self.plot_tab_control, width=800, height=600)
        self.plot_tab_control.add(self.blank_tab, text="Welcome")
        
        # Add a welcome label with theme-appropriate styling
        welcome_label = ttk.Label(
            self.blank_tab, 
            text="Welcome to Peak Analysis Tool\n\nPlease load a file to begin", 
            font=("Arial", 14),
            foreground=self.theme_manager.get_color('text'),
            background=self.theme_manager.get_color('background')
        )
        welcome_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Prevent the blank tab from shrinking
        self.blank_tab.pack_propagate(False)

        # Functional Bar under plot tabs (simplified)
        functional_bar = ttk.Frame(preview_frame)
        functional_bar.grid(row=1, column=0, sticky="ew", pady=10)

        ttk.Button(functional_bar, 
                  text="Export Plot", 
                  command=self.export_plot
        ).grid(row=0, column=0, padx=5, pady=5)

        # Preview label for status messages
        self.preview_label = ttk.Label(control_frame, text="", foreground="black")
        self.preview_label.pack(fill=tk.X, padx=5, pady=5)

    def create_menu_bar(self):
        """Create the application menu bar"""
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)
        
        # File Menu
        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open File", command=self.browse_file)
        file_menu.add_command(label="Export Results", command=self.save_peak_information_to_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Export Current Plot", command=self.export_plot)
        file_menu.add_command(label="Take Screenshot", command=self.take_screenshot)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Edit Menu
        edit_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Reset Application", command=self.reset_application_state)
        edit_menu.add_command(label="Clear Results", command=lambda: self.update_results_summary(preview_text=""))
        
        # View Menu
        view_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Raw Data", command=self.plot_raw_data)
        view_menu.add_command(label="Filtered Data", command=self.start_analysis)
        view_menu.add_command(label="Detected Peaks", command=self.run_peak_detection)
        view_menu.add_separator()
        view_menu.add_command(label="Peak Analysis", command=self.plot_data)
        view_menu.add_command(label="Peak Correlations", command=self.plot_scatter)
        view_menu.add_separator()
        # Add theme toggle option
        current_theme = "Light" if self.theme_manager.current_theme == "dark" else "Dark"
        view_menu.add_command(label=f"Switch to {current_theme} Theme", command=self.toggle_theme)
        
        # Tools Menu
        tools_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Auto Calculate Threshold", command=self.calculate_auto_threshold)
        tools_menu.add_command(label="Auto Calculate Cutoff", command=self.calculate_auto_cutoff_frequency)
        tools_menu.add_separator()
        tools_menu.add_command(label="View Individual Peaks", command=self.plot_filtered_peaks)
        tools_menu.add_command(label="Next Peaks", command=self.show_next_peaks)
        
        # Help Menu
        help_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about_dialog)
    
    def show_documentation(self):
        """Show application documentation"""
        # In a real application, this might open a browser to online docs
        # or display a help window
        documentation_text = """
Peak Analysis Tool Documentation
===============================

This tool helps analyze signal data to detect and characterize peaks.

Main Features:
1. Load single or batch files for analysis
2. Apply filtering to reduce noise
3. Detect peaks with customizable parameters
4. Analyze peak properties (width, height, area)
5. Generate visualizations and reports

For detailed usage instructions, please refer to the full documentation.
"""
        help_window = tk.Toplevel(self)
        help_window.title("Documentation")
        help_window.geometry("600x400")
        
        text_widget = ScrolledText(help_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, documentation_text)
        text_widget.config(state=tk.DISABLED)
    
    def show_about_dialog(self):
        """Show about dialog with application information"""
        version = "1.0"
        about_text = f"""
Peak Analysis Tool v{version}

Developed by Lucjan & Silas

This tool provides advanced peak detection and analysis capabilities 
for signal processing and scientific data analysis.

© 2024 All rights reserved.
"""
        messagebox.showinfo("About Peak Analysis Tool", about_text)

    # Replace the old create_tooltip method with the new one
    def add_tooltip(self, widget, text):
        """Add an enhanced tooltip to a widget"""
        # Use neutral gray colors for tooltips that match our updated theme
        if self.theme_manager.current_theme == 'dark':
            bg_color = '#3a3a3a'  # Darker neutral gray
            fg_color = '#e0e0e0'  # Light gray text
        else:
            bg_color = '#707070'  # Neutral medium gray
            fg_color = '#ffffff'  # White text
            
        return create_tooltip(
            widget, 
            text,
            bg=bg_color,
            fg=fg_color,
            font=self.theme_manager.get_font('small'),
            delay=200  # Faster display for better responsiveness
        )

    def load_single_file(self, file, timestamps=None, index=0):
        """
        Helper function to load a single file
        Args:
            file (str): Path to the file to load
            timestamps (list, optional): List of timestamps for batch mode
            index (int, optional): Index of the file in the batch

        Returns:
            dict: Dictionary containing time, amplitude and index data
        """
        print(f"Loading file {index+1}: {file}")

        try:
            # Determine file type based on extension
            if file.lower().endswith(('.xls', '.xlsx')):
                # For Excel files, only read necessary columns to save memory
                df = pd.read_excel(file, usecols=[0, 1])
            else:
                # For CSV/TXT files, use more efficient options:
                # - Use engine='c' for faster parsing
                # - Only read the first two columns
                # - Use float32 instead of float64 to reduce memory usage
                # - Skip empty lines and comments
                df = pd.read_csv(
                    file, 
                    delimiter='\t',
                    usecols=[0, 1],
                    dtype={0: np.float32, 1: np.float32},
                    engine='c', 
                    skip_blank_lines=True,
                    comment='#'
                )
            
            # Get column names and handle missing headers efficiently
            cols = df.columns.tolist()
            
            # Use direct dictionary access for faster column renaming
            if len(cols) >= 2:
                # Only strip whitespace if needed
                if any(c != c.strip() for c in cols):
                    df.columns = [c.strip() for c in cols]
                
                # Most efficient way to get column names
                if 'Time - Plot 0' in df.columns and 'Amplitude - Plot 0' in df.columns:
                    time_col = 'Time - Plot 0'
                    amp_col = 'Amplitude - Plot 0'
                else:
                    # Rename columns without creating a new DataFrame
                    df.columns = ['Time - Plot 0', 'Amplitude - Plot 0']
                    time_col = 'Time - Plot 0'
                    amp_col = 'Amplitude - Plot 0'
            else:
                raise ValueError(f"File {file} doesn't have at least 2 columns")
            
            # Extract numpy arrays directly for better performance
            # and use numpy.ascontiguousarray for faster array operations later
            return {
                'time': np.ascontiguousarray(df[time_col].values),
                'amplitude': np.ascontiguousarray(df[amp_col].values),
                'index': index
            }
        except Exception as e:
            print(f"Error loading file {file}: {str(e)}")
            raise


    def reset_application_state(self):
        """Reset all application variables and plots to initial state"""
        try:
            # Reset data variables
            self.data = None
            self.t_value = None
            self.x_value = None
            self.filtered_signal = None
            self.segment_offset = 0

            # Reset variables to default values
            self.normalization_factor.set(1.0)
            self.start_time.set("0:00")
            self.big_counts.set(100)
            self.height_lim.set(20)
            self.distance.set(30)
            self.rel_height.set(0.85)
            self.width_p.set("1,200")
            self.cutoff_value.set(0)

            # Clear results summary
            self.update_results_summary(preview_text="")

            # Clear all tabs except Welcome tab
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") != "Welcome":
                    self.plot_tab_control.forget(tab)

            # Reset tab figures dictionary
            self.tab_figures.clear()

            # Reset preview label
            self.preview_label.config(text="Application state reset", foreground="blue")

            # Reset progress bar
            self.update_progress_bar(0)

        except Exception as e:
            self.data = None
            self.t_value = None
            self.x_value = None
            self.show_error("Error resetting application state", e)
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
            self.update_progress_bar(0)  # Reset on error
            
            # Update status
            self.status_indicator.set_state('error')
            self.status_indicator.set_text("Error resetting application state")

    @profile_function
    def browse_file(self):
        """Browse and load file(s) based on current mode"""
        print(f"Memory before loading: {get_memory_usage():.2f} MB")
        print(f"Current file mode: {self.file_mode.get()}")
        
        try:
            # Set status indicator to loading
            self.status_indicator.set_state('loading')
            self.status_indicator.set_text("Loading files...")
            self.update_idletasks()
            
            # Reset application state before loading new file
            self.reset_application_state()
            
            # Reset progress bar
            self.update_progress_bar(0)
            
            files = []
            if self.file_mode.get() == "single":
                files = list(filedialog.askopenfilenames(
                    title="Select Data File",
                    filetypes=(
                        ("Data files", "*.txt *.xls *.xlsx"),
                        ("All files", "*.*")
                    )
                ))
            else:  # batch mode
                folder = filedialog.askdirectory(title="Select Folder with Data Files")
                if folder:
                    # Include both text and Excel files
                    files = [os.path.join(folder, f) for f in os.listdir(folder) 
                            if f.lower().endswith(('.txt', '.xls', '.xlsx'))]
            
            if files:
                files = list(Tcl().call('lsort', '-dict', files))
                self.preview_label.config(text="Loading files...", foreground="blue")
                self.update_idletasks()
                
                # Set maximum progress
                self.update_progress_bar(0, len(files))
                
                # Store file names
                self.loaded_files = [os.path.basename(f) for f in files]
                
                # Get timestamps if in batch mode
                timestamps = []
                if self.file_mode.get() == "batch":
                    timestamps = [t.strip() for t in self.batch_timestamps.get().split(',') if t.strip()]
                
                # Use ThreadPoolExecutor for parallel file loading
                results = []
                with ThreadPoolExecutor(max_workers=min(len(files), os.cpu_count() * 2)) as executor:
                    future_to_file = {
                        executor.submit(self.load_single_file, file, timestamps, i): i 
                        for i, file in enumerate(files)
                    }
                    
                    for future in as_completed(future_to_file):
                        i = future_to_file[future]
                        try:
                            result = future.result()
                            results.append(result)
                            self.update_progress_bar(len(results))
                            
                            # Update status message
                            self.status_indicator.set_text(f"Loading files... {len(results)}/{len(files)}")
                            self.update_idletasks()
                        except Exception as e:
                            self.status_indicator.set_state('error')
                            self.status_indicator.set_text(f"Error loading file {i+1}")
                            print(f"Error loading file {i}: {str(e)}")
                            raise e

                # Sort results by index to maintain order
                results.sort(key=lambda x: x['index'])
                
                # Process results - More memory-efficient implementation
                # Pre-calculate the total size needed for the arrays
                total_time_points = sum(len(result['time']) for result in results)
                
                # Pre-allocate arrays with the correct size and data type
                combined_times = np.zeros(total_time_points, dtype=np.float32)
                combined_amplitudes = np.zeros(total_time_points, dtype=np.float32)
                
                # Copy data into the pre-allocated arrays
                start_idx = 0
                for i, result in enumerate(results):
                    time_data = result['time']
                    amplitude_data = result['amplitude']
                    n_points = len(time_data)
                    
                    # Apply time offset directly during copy
                    if self.file_mode.get() == "batch" and timestamps and i > 0:
                        if i == 0:
                            start_time = timestamps_to_seconds([timestamps[0]], timestamps[0])[0]*1e4
                        current_time = timestamps_to_seconds([timestamps[i]], timestamps[0])[0]*1e4
                        time_offset = current_time
                        combined_times[start_idx:start_idx + n_points] = time_data + time_offset
                    else:
                        if i > 0:
                            # Calculate time offset from the end of the previous segment
                            time_offset = combined_times[start_idx - 1] + (time_data[1] - time_data[0])
                            combined_times[start_idx:start_idx + n_points] = time_data + time_offset
                        else:
                            # First segment has no offset
                            combined_times[start_idx:start_idx + n_points] = time_data
                    
                    # Copy amplitude data
                    combined_amplitudes[start_idx:start_idx + n_points] = amplitude_data
                    
                    # Update the start index for the next segment
                    start_idx += n_points
                
                # Store the combined arrays
                self.t_value = combined_times
                self.x_value = combined_amplitudes
                
                print(f"Total data points after concatenation: {len(self.t_value)}")  # Debug print
                
                # Create combined DataFrame - more efficient version
                # Only create DataFrame with data that will actually be used
                self.data = pd.DataFrame({
                    'Time - Plot 0': self.t_value,
                    'Amplitude - Plot 0': self.x_value
                })
                
                # Update GUI
                if len(files) == 1:
                    self.file_path.set(files[0])
                    self.file_name_label.config(text=os.path.basename(files[0]))
                else:
                    first_file = os.path.basename(files[0])
                    self.file_path.set(files[0])
                    self.file_name_label.config(text=f"{first_file} +{len(files)-1} more")
                
                # Create preview text
                file_order_text = "\n".join([f"{i+1}. {fname}" for i, fname in enumerate(self.loaded_files)])
                if timestamps:
                    file_order_text += "\n\nTimestamps:"
                    file_order_text += "\n".join([f"{fname}: {tstamp}" 
                                                for fname, tstamp in zip(self.loaded_files, timestamps)])
                
                preview_text = (
                    f"Successfully loaded {len(files)} files\n"
                    f"Total rows: {len(self.data):,}\n"
                    f"Time range: {self.data['Time - Plot 0'].min():.2f} to {self.data['Time - Plot 0'].max():.2f}\n"
                    f"\nFiles loaded in order:\n{file_order_text}\n"
                    f"\nPreview of combined data:\n"
                    f"{self.data.head().to_string(index=False)}"
                )
                
                # Update status
                self.status_indicator.set_state('success')
                self.status_indicator.set_text(f"Loaded {len(files)} files successfully")
                
                self.preview_label.config(text="Files loaded successfully", foreground="green")
                self.update_results_summary(preview_text=preview_text)
                
            else:
                # Update status
                self.status_indicator.set_state('idle')
                self.status_indicator.set_text("No files selected")
                
        except Exception as e:
            self.data = None
            self.t_value = None
            self.x_value = None
            self.show_error("Error loading files", e)
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
            self.update_progress_bar(0)  # Reset on error
            
            # Update status
            self.status_indicator.set_state('error')
            self.status_indicator.set_text("Error loading files")

    # Function to plot the raw data
    @profile_function
    def plot_raw_data(self):
        """Ruft die ausgelagerte Funktion für optimierte Darstellung der Rohdaten auf"""
        return plot_raw_data_function(self, profiler=profile_function)

    # Function to start data analysis
    @profile_function
    def start_analysis(self):
        """Start the analysis and peak detection process"""
        return start_analysis_function(self, profile_function=profile_function)

    # Function to run peak detection
    @profile_function
    def run_peak_detection(self):
        """Run peak detection on the filtered signal"""
        return run_peak_detection_function(self, profile_function=profile_function)

    
    # Function to plot the detected filtered peaks
    @profile_function
    def plot_filtered_peaks(self):
        """Display individual peaks in a grid for detailed analysis"""
        return plot_filtered_peaks_function(self, profile_function=profile_function)

    def show_next_peaks(self):
        """Show the next set of example peaks"""
        return show_next_peaks_function(self, profile_function=profile_function)

    # Function to calculate the areas of detected peaks
    @profile_function
    def calculate_peak_areas(self):
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            # Get current parameters
            height_lim_factor = self.height_lim.get()
            distance = self.distance.get()
            rel_height = self.rel_height.get()
            width_values = self.width_p.get().strip().split(',')
            
            # Detect peaks using the PeakDetector
            self.peak_detector.detect_peaks(
                self.filtered_signal,
                self.t_value,
                height_lim_factor,
                distance,
                rel_height,
                width_values
            )
            
            # Calculate areas using the PeakDetector
            peak_area, start, end = self.peak_detector.calculate_peak_areas(self.filtered_signal)
        
            self.preview_label.config(text="Peak area calculation completed", foreground="green")
            self.update_results_summary(peak_areas=peak_area)
            return peak_area, start, end 

        except Exception as e:
            self.preview_label.config(text=f"Error calculating peak areas: {e}", foreground="red")
            self.logger.error(f"Error calculating peak areas: {str(e)}\n{traceback.format_exc()}")
            return None

    # Function to save peak information to CSV
    def save_peak_information_to_csv(self):
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            # Get current parameters
            height_lim_factor = self.height_lim.get()
            distance = self.distance.get()
            rel_height = self.rel_height.get()
            width_values = self.width_p.get().strip().split(',')
            
            # Detect peaks
            self.peak_detector.detect_peaks(
                self.filtered_signal,
                self.t_value,
                height_lim_factor,
                distance,
                rel_height,
                width_values
            )
            
            # Calculate areas
            self.peak_detector.calculate_peak_areas(self.filtered_signal)
            
            # Create protocol info dictionary
            protocol_info = {
                "Start Time": self.protocol_start_time.get(),
                "Particle": self.protocol_particle.get(),
                "Concentration": self.protocol_concentration.get(),
                "Stamp": self.protocol_stamp.get(),
                "Laser Power": self.protocol_laser_power.get(),
                "Setup": self.protocol_setup.get(),
                "Notes": self.protocol_notes.get()
            }
            
            # Create DataFrame with all peak information
            results_df = self.peak_detector.create_peak_dataframe(
                self.t_value,
                protocol_info
            )

            # Save to CSV
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Peak Information"
            )
            
            if file_path:
                results_df.to_csv(file_path, index=False)
                self.preview_label.config(text="Peak information saved successfully", foreground="green")
            else:
                self.preview_label.config(text="Save cancelled", foreground="blue")

        except Exception as e:
            self.preview_label.config(text=f"Error saving peak information: {e}", foreground="red")
            self.logger.error(f"Error saving peak information: {str(e)}\n{traceback.format_exc()}")
        
    # Function to calculate intervals between detected peaks
    def calculate_peak_intervals(self):
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            # Get current parameters
            height_lim_factor = self.height_lim.get()
            distance = self.distance.get()
            rel_height = self.rel_height.get()
            width_values = self.width_p.get().strip().split(',')
            
            # Detect peaks using the PeakDetector
            self.peak_detector.detect_peaks(
                self.filtered_signal,
                self.t_value,
                height_lim_factor,
                distance,
                rel_height,
                width_values
            )
            
            # Calculate intervals using the PeakDetector
            peaks, peaks_intervall = self.peak_detector.calculate_peak_intervals(self.t_value)
            
            self.preview_label.config(text="Peak interval calculation completed", foreground="green")
            self.update_results_summary(peak_intervals=peaks_intervall)
            return peaks, peaks_intervall
        
        except Exception as e:
            self.preview_label.config(text=f"Error calculating peak intervals: {e}", foreground="red")
            self.logger.error(f"Error calculating peak intervals: {str(e)}\n{traceback.format_exc()}")
            return None
    
    # Function to plot processed data with detected peaks
    def plot_data(self):
        """Plot peak property data over time in multiple panels"""
        return plot_data_function(self, profile_function=profile_function)

    # Function to export the current plot
    def export_plot(self):
        """Export plot with high resolution"""
        try:
            # Update status
            self.status_indicator.set_state('processing')
            self.status_indicator.set_text("Exporting plot...")
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg"),
                    ("TIFF files", "*.tiff")
                ]
            )
            if file_path:
                # Get the currently selected tab
                current_tab = self.plot_tab_control.select()
                tab_text = self.plot_tab_control.tab(current_tab, "text")
                print(f"Exporting from tab: {tab_text}")  # Debug output
                
                # Get figure from dictionary
                figure_to_export = self.tab_figures.get(tab_text)
                print(f"Found figure in dictionary: {figure_to_export is not None}")  # Debug output
                
                if figure_to_export is None:
                    self.status_indicator.set_state('warning')
                    self.status_indicator.set_text("No figure to export")
                    print("Warning: No figure found for this tab")
                    return
                
                # Always export with white background
                original_facecolor = figure_to_export.get_facecolor()
                figure_to_export.set_facecolor('white')
                
                # Make sure all axes have white background too
                for ax in figure_to_export.get_axes():
                    original_ax_facecolor = ax.get_facecolor()
                    ax.set_facecolor('white')
                
                # Save with high DPI and tight layout
                figure_to_export.savefig(
                    file_path,
                    dpi=Config.Plot.EXPORT_DPI,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none'
                )
                
                # Restore original colors if needed
                figure_to_export.set_facecolor(original_facecolor)
                for ax in figure_to_export.get_axes():
                    ax.set_facecolor(original_ax_facecolor)
                
                # Update status
                self.status_indicator.set_state('success')
                self.status_indicator.set_text(f"Plot exported to {os.path.basename(file_path)}")
                
                self.preview_label.config(
                    text=f"Plot exported successfully to {file_path}", 
                    foreground=self.theme_manager.get_color('success')
                )
                print(f"Successfully exported figure to {file_path}")  # Debug output
            else:
                # Export cancelled
                self.status_indicator.set_state('idle')
                self.status_indicator.set_text("Export cancelled")
                
        except Exception as e:
            error_msg = f"Error exporting plot: {str(e)}"
            print(f"Export error: {error_msg}")  # Debug output
            
            # Update status
            self.status_indicator.set_state('error')
            self.status_indicator.set_text("Error exporting plot")
            
            self.preview_label.config(
                text=error_msg, 
                foreground=self.theme_manager.get_color('error')
            )
            
    def take_screenshot(self):
        """Take a screenshot of the entire application window"""
        try:
            # Update status
            self.status_indicator.set_state('processing')
            self.status_indicator.set_text("Taking screenshot...")
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png")],
                title="Save Screenshot"
            )
            
            if file_path:
                # Wait for dialog to close
                self.update_idletasks()
                self.after(200)  # Short delay
                
                # Get window geometry
                x = self.winfo_rootx()
                y = self.winfo_rooty()
                width = self.winfo_width()
                height = self.winfo_height()
                
                # Take screenshot
                from PIL import ImageGrab
                screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
                screenshot.save(file_path)
                
                # Update status
                self.status_indicator.set_state('success')
                self.status_indicator.set_text(f"Screenshot saved to {os.path.basename(file_path)}")
                
                self.preview_label.config(
                    text=f"Screenshot saved to {file_path}", 
                    foreground=self.theme_manager.get_color('success')
                )
            else:
                # Screenshot cancelled
                self.status_indicator.set_state('idle')
                self.status_indicator.set_text("Screenshot cancelled")
                
        except Exception as e:
            # Update status
            self.status_indicator.set_state('error')
            self.status_indicator.set_text("Error taking screenshot")
            
            self.show_error("Error taking screenshot", e)

    # Add new method for auto cutoff calculation
    def calculate_auto_cutoff_frequency(self):
        if self.filtered_signal is None:
            print("DEBUG: No filtered signal available")
            return
        
        try:
            print("\nDEBUG: Starting auto cutoff calculation")
            print(f"DEBUG: Current normalization factor: {self.normalization_factor.get()}")
            
            # Get current parameters
            big_counts = self.big_counts.get()
            norm_factor = self.normalization_factor.get()
            
            print(f"DEBUG: Using parameters:")
            print(f"- Big counts: {big_counts}")
            print(f"- Normalization factor: {norm_factor}")
            
            # Calculate cutoff
            _, calculated_cutoff = adjust_lowpass_cutoff(
                self.x_value,  # Use original signal
                self.fs,
                big_counts,
                norm_factor
            )
            
            print(f"DEBUG: Calculated cutoff: {calculated_cutoff}")
            
            # Update the cutoff value in GUI
            self.cutoff_value.set(calculated_cutoff)
            
            self.preview_label.config(
                text=f"Cutoff frequency automatically set to {calculated_cutoff:.1f} Hz", 
                foreground="green"
            )
            
        except Exception as e:
            self.preview_label.config(
                text=f"Error in auto calculation: {str(e)}", 
                foreground="red"
            )
            print(f"DEBUG: Error in auto calculation: {str(e)}")

    

    def on_file_mode_change(self):
        """Handle file mode changes between single and batch modes"""
        if self.file_mode.get() == "batch":
            self.timestamps_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.timestamps_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
            self.browse_button.config(text="Select Folder")
        else:
            self.timestamps_label.pack_forget()
            self.timestamps_entry.pack_forget()
            self.browse_button.config(text="Load File")
        
        # Force update of the GUI
        self.update_idletasks()

    # Add this helper function at the class level
    def show_error(self, title, error):
        """
        Display error message and log error details
        
        Parameters
        ----------
        title : str
            Error title
        error : Exception
            Exception object
        """
        # Log error
        self.logger.error(f"{title}: {str(error)}")
        self.logger.error(traceback.format_exc())
        
        # Update status label
        self.preview_label.config(
            text=f"{title}: {str(error)}", 
            foreground=self.theme_manager.get_color('error')
        )
        
        # Show error message box for serious errors
        messagebox.showerror(title, str(error))

    # Add this method to the Application class
    def get_width_range(self):
        """Convert width_p string to list of integers"""
        try:
            # Get width range from the width_p StringVar
            width_str = self.width_p.get()
            
            # Split the string and convert to integers
            width_min, width_max = map(int, width_str.split(','))
            
            # Return as list
            return [width_min, width_max]
            
        except Exception as e:
            self.show_error("Error parsing width range", e)
            # Return default values if there's an error
            return [1, 200]

    # Function to plot a scatter plot of peak area vs amplitude
    def plot_scatter(self):
        """Create detailed scatter plots of peak property correlations"""
        return plot_scatter_function(self, profile_function=profile_function)

    # Function to update the results summary text box
    def update_results_summary(self, events=None, max_amp=None, peak_areas=None, peak_intervals=None, preview_text=None):
        try:
            self.results_summary.config(state=tk.NORMAL)
            self.results_summary.delete(1.0, tk.END)  # Clear existing content
            
            summary_text = ""
            if events is not None:
                summary_text += f"Number of Peaks Detected: {events}\n"
            if max_amp is not None:
                summary_text += f"Maximum Peak Amplitude: {max_amp}\n"
            if peak_areas is not None:
                summary_text += f"Peak Areas: {peak_areas[:10]}\n"
            if peak_intervals is not None:
                summary_text += f"Peak Intervals: {peak_intervals[:10]}\n"
            if preview_text is not None:
                summary_text += f"\nData Preview:\n{preview_text}\n"
            
            self.results_summary.insert(tk.END, summary_text)
            self.results_summary.config(state=tk.DISABLED)
            
        except Exception as e:
            self.show_error("Error updating results", e)
            self.results_summary.config(state=tk.DISABLED)

    # Add this validation method to your class
    def validate_float(self, value):
        """Validate that the input is a valid float"""
        if value == "":
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False

    def update_progress_bar(self, value=0, maximum=None):
        """Update progress bar with optional maximum value"""
        try:
            if maximum is not None:
                self.progress['maximum'] = maximum
            self.progress['value'] = value
            self.update_idletasks()
            
            # Reset to zero if completed
            if value >= self.progress['maximum']:
                self.after(Config.PROGRESS_RESET_DELAY, 
                          lambda: self.update_progress_bar(0))
        except Exception as e:
            print(f"Error updating progress bar: {e}")

   

    def decimate_for_plot(self, x, y, max_points=10000):
        """
        Intelligently reduce number of points for plotting while preserving important features
        
        Args:
            x: time array
            y: signal array
            max_points: maximum number of points to plot
        
        Returns:
            x_decimated, y_decimated: decimated arrays for plotting
        """
        if len(x) <= max_points:
            return x, y
        
        # More efficient decimation algorithm for very large datasets
        n_points = len(x)
        
        # Use numpy operations instead of loops for better performance
        # Calculate decimation factor
        stride = max(1, n_points // max_points)
        
        # For extremely large datasets, use a more aggressive approach
        if stride > 50:
            # Initialize mask - avoid Python loops entirely
            mask = np.zeros(n_points, dtype=bool)
            
            # Include regularly spaced points - efficient slicing
            mask[::stride] = True
            
            # Find peaks efficiently using vectorized operations
            # Use a simplified peak finding for speed - just look for local maxima
            # This is much faster than scipy.signal.find_peaks for this purpose
            if n_points > 3:  # Need at least 3 points for this method
                # Create shifted arrays for comparison
                y_left = np.empty_like(y)
                y_left[0] = -np.inf
                y_left[1:] = y[:-1]
                
                y_right = np.empty_like(y)
                y_right[-1] = -np.inf
                y_right[:-1] = y[1:]
                
                # Find local maxima
                peaks = (y > y_left) & (y > y_right)
                
                # Only keep significant peaks
                if np.any(peaks):
                    threshold = np.mean(y) + 2 * np.std(y[peaks])
                    significant_peaks = peaks & (y > threshold)
                    mask[significant_peaks] = True
                    
                    # Include points around significant peaks for better visualization
                    for offset in range(-3, 4):  # Include 3 points before and after peaks
                        shifted = np.zeros_like(mask)
                        if offset < 0:
                            shifted[:offset] = significant_peaks[-offset:]
                        elif offset > 0:
                            shifted[offset:] = significant_peaks[:-offset]
                        else:  # offset == 0, already handled
                            continue
                        mask |= shifted
            
            # Apply mask using numpy indexing
            return x[mask], y[mask]
        else:
            # For moderately large datasets, use simple stride-based decimation
            # This is much faster and works well for most visualizations
            return x[::stride], y[::stride]

    def toggle_theme(self):
        """Toggle between light and dark theme"""
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
        self.create_menu_bar()
        
        # Update status indicator
        theme_name = "Dark" if new_theme == "dark" else "Light"
        self.status_indicator.set_state('success')
        self.status_indicator.set_text(f"Switched to {theme_name} Theme")
        
        # Update preview label
        self.preview_label.config(
            text=f"Switched to {theme_name} Theme", 
            foreground=self.theme_manager.get_color('success')
        )
        
        self.update_idletasks()

def resource_path(relative_path):
    """ Get the absolute path to the resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)    

def splash_screen():
    splash = tk.Tk()
    splash.overrideredirect(True)  # Remove window decorations

    # Load and display the image
    img_path = resource_path("resources/images/startim.png")  # Adjust the path based on your structure
    img = Image.open(img_path)
    img = img.resize((400, 400), Image.LANCZOS)  # Resize to a larger square size
    photo = ImageTk.PhotoImage(img)

    label = tk.Label(splash, image=photo)
    label.pack()

    # Create a loading bar
    loading_bar = ttk.Progressbar(splash, orient="horizontal", length=300, mode="indeterminate")
    loading_bar.pack(pady=20)  # Add some vertical padding
    loading_bar.start()  # Start the loading animation

    # Set the size of the splash screen
    splash.update_idletasks()  # Update "requested size" from geometry manager
    width = splash.winfo_width()
    height = splash.winfo_height()

    # Get screen width and height
    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()

    # Calculate the position for centering
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)

    # Set the geometry of the splash screen
    splash.geometry(f"{width}x{height}+{x}+{y}")

    # Show the splash screen for 3 seconds
    splash.after(3000, lambda: [loading_bar.stop(), splash.destroy()])  # Stop the loading bar and destroy the splash screen
    splash.mainloop()

# Call the splash screen function
#splash_screen()

# Your main program code goes here

if __name__ == "__main__":
    app = Application()
    app.mainloop()