# -*- coding: utf-8 -*-
"""
Created Nov 12 02:54:48 2024

@author: Lucjan & Silas
"""

import tkinter as tk
from tkinter import filedialog, messagebox, Tcl
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, butter, filtfilt, peak_widths
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.lines import Line2D
import os
import time
import cProfile
import pstats
from functools import wraps
import logging
import psutil
import traceback

# Add after the imports, before the first function definition
def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get memory usage before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Get start time
        start_time = time.time()
        
        # Run the function
        result = func(*args, **kwargs)
        
        # Get end time
        end_time = time.time()
        
        # Get memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        # Calculate memory difference
        memory_diff = memory_after - memory_before
        
        # Log the performance data
        logging.info(f"{func.__name__} performance:")
        logging.info(f"  Time: {end_time - start_time:.2f} seconds")
        logging.info(f"  Memory before: {memory_before:.1f} MB")
        logging.info(f"  Memory after: {memory_after:.1f} MB")
        logging.info(f"  Memory difference: {memory_diff:.1f} MB")
        
        # Print to console as well
        print(f"\n{func.__name__} performance:")
        print(f"  Time: {end_time - start_time:.2f} seconds")
        print(f"  Memory usage: {memory_diff:+.1f} MB")
        
        return result
    return wrapper

# Butterworth filter application
def apply_butterworth_filter(order, Wn, btype, fs, x):
    b, a = butter(order, Wn, btype, fs=fs)
    x_f = filtfilt(b, a, x)
    print(f'Butterworth filter coefficients (b, a): {b}, {a}')
    print(f'Filtered signal: {x_f[:10]}...')  # Printing the first 10 values for brevity
    return x_f

# Find the nearest value in an array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    print(f'Nearest index: {idx}, Nearest value: {array[idx]}')
    return idx

# Convert timestamps to seconds
def timestamps_to_seconds(timestamps, start_time):
    """Convert timestamps to seconds from start time"""
    try:
        seconds = []
        start_min, start_sec = map(int, start_time.split(':'))
        start_time_seconds = start_min * 60 + start_sec
        
        for timestamp in timestamps:
            minu, sec = map(int, timestamp.split(':'))
            seconds.append(minu * 60 + sec - start_time_seconds)
            
        return seconds
    except Exception as e:
        raise ValueError(f"Error converting timestamps: {e}\n"
                        f"Format should be 'MM:SS' (e.g., '01:30')")

# Detect peaks with a sliding window and filter out invalid ones
def find_peaks_with_window(signal, width, prominence, distance, rel_height):
    profiler = cProfile.Profile()
    profiler.enable()
    
    peaks, properties = find_peaks(signal, 
                                 width=width,
                                 prominence=prominence,
                                 distance=distance,
                                 rel_height=rel_height)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)  # Print top 20 time-consuming operations
    
    return peaks, properties

# Estimate the average peak width
def estimate_peak_widths(signal, fs, big_counts):
    peaks, _ = find_peaks(signal, width=[1, 2000], prominence=big_counts, distance=1000)
    widths = peak_widths(signal, peaks, rel_height=0.5)[0]
    avg_width = np.mean(widths) / fs
    print(f'Estimated peak widths: {widths}')
    return avg_width
# Add this memory tracking function
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB
# Adjust the low-pass filter cutoff frequency
def adjust_lowpass_cutoff(signal, fs, big_counts, normalization_factor):
    print("\nDEBUG: Starting adjust_lowpass_cutoff")
    print(f"DEBUG: Input parameters:")
    print(f"- fs: {fs}")
    print(f"- big_counts: {big_counts}")
    print(f"- normalization_factor: {normalization_factor}")
    
    avg_width = estimate_peak_widths(signal, fs, big_counts)
    print(f'\nDEBUG: Average width of peaks: {avg_width} seconds')
    
    # Calculate base cutoff frequency from average width (in Hz)
    base_cutoff = 1 / avg_width  # Convert time width to frequency
    print(f'DEBUG: Base cutoff frequency (1/width): {base_cutoff:.2f} Hz')
    
    # Apply normalization factor to adjust the cutoff
    cutoff = base_cutoff * float(normalization_factor)  # Ensure normalization_factor is float
    print(f'DEBUG: After normalization:')
    print(f'- Base cutoff: {base_cutoff:.2f} Hz')
    print(f'- Normalization factor: {normalization_factor}')
    print(f'- Adjusted cutoff: {cutoff:.2f} Hz')
    
    # Limit the cutoff frequency to reasonable values
    original_cutoff = cutoff
    cutoff = max(min(cutoff, 10000), 50)
    if cutoff != original_cutoff:
        print(f'DEBUG: Cutoff was limited from {original_cutoff:.2f} to {cutoff:.2f} Hz')
    
    print(f'\nDEBUG: Final cutoff frequency: {cutoff:.2f} Hz')
    
    order = 2
    btype = 'lowpass'
    filtered_signal = apply_butterworth_filter(order, cutoff, btype, fs, signal)
    
    return filtered_signal, cutoff

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.setup_performance_logging()
        self.title("Peak Analysis Tool")
        self.geometry("1920x1080")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # Initialize figure and canvas as None
        self.figure = Figure(figsize=(8, 6))
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

        self.create_widgets()
        self.blank_tab_exists = True  # Track if the blank tab exists

        self.setup_performance_logging()

        # Initialize data plot attributes
        self.data_figure = None
        self.data_canvas = None
        self.data_original_xlims = None
        self.data_original_ylims = None

    def setup_performance_logging(self):
        logging.basicConfig(
            filename='performance.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PeakAnalysis')

    def calculate_auto_threshold(self):
        if self.filtered_signal is not None:
            signal_std = np.std(self.filtered_signal)
            suggested_threshold = 5 * signal_std  # Changed to 7-sigma
            self.height_lim.set(suggested_threshold)
            self.preview_label.config(
                text=f"Threshold automatically set to {suggested_threshold:.1f} (5σ)", 
                foreground="green"
            )
        else:
            self.preview_label.config(
                text="Please run analysis first", 
                foreground="red"
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
        self.create_tooltip(
            file_mode_frame,
            "Choose between single file analysis or batch processing of multiple files"
        )

        # File selection frame
        file_frame = ttk.LabelFrame(data_loading_tab, text="File Selection")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        # Browse button and file name label
        self.browse_button = ttk.Button(file_frame, text="Load File", command=self.browse_file)
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
        self.create_tooltip(
            self.browse_button,
            "Click to select a data file (single mode) or folder (batch mode)"
        )

        self.create_tooltip(
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
            self.create_tooltip(label_widget, tooltip_text)
            self.create_tooltip(entry_widget, tooltip_text)

        # Notes field
        ttk.Label(protocol_frame, text="Notes:").grid(row=len(protocol_entries), column=0, padx=5, pady=2, sticky="w")
        notes_entry = ttk.Entry(protocol_frame, textvariable=self.protocol_notes)
        notes_entry.grid(row=len(protocol_entries), column=1, padx=5, pady=2, sticky="ew")

        # Add tooltip for notes field
        notes_label = protocol_frame.grid_slaves(row=len(protocol_entries), column=0)[0]
        self.create_tooltip(
            notes_label,
            "Enter any additional notes or observations about the experiment"
        )
        self.create_tooltip(
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
        self.create_tooltip(
            cutoff_entry,
            "Frequency cutoff for the Butterworth low-pass filter (Hz)\nSet to 0 for automatic calculation"
        )

        self.create_tooltip(
            auto_cutoff_button,
            "Calculate optimal cutoff frequency based on peak widths"
        )

        self.create_tooltip(
            big_counts_entry,
            "Threshold for identifying largest peaks\nUsed for automatic cutoff calculation"
        )

        self.create_tooltip(
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
        self.create_tooltip(
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
            text="Detect Peaks",  # Changed from "Run Peak Detection"
            command=self.run_peak_detection
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame, 
            text="View Individual Peaks",  # Changed from "Display Example Peaks"
            command=self.plot_filtered_peaks
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            action_frame, 
            text="Quick Save Results",  # Changed from "Save Peak Information"
            command=self.save_peak_information_to_csv
        ).pack(side=tk.LEFT, padx=5)

        # Configure grid weights
        peak_params_frame.columnconfigure(1, weight=1)

        # Add tooltips for better user guidance
        self.create_tooltip(
            threshold_entry,
            "Minimum height threshold for peak detection"
        )
        self.create_tooltip(
            distance_entry,
            "Minimum number of points between peaks"
        )
        self.create_tooltip(
            rel_height_entry,
            "Relative height from peak maximum for width calculation (0-1)"
        )
        self.create_tooltip(
            width_entry,
            "Expected peak width range in milliseconds (min,max)"
        )
        self.create_tooltip(
            action_frame.winfo_children()[0],  # Detect Peaks button
            "Run peak detection algorithm with current parameters"
        )
        self.create_tooltip(
            action_frame.winfo_children()[1],  # View Individual Peaks button
            "Display detailed view of selected individual peaks"
        )
        self.create_tooltip(
            action_frame.winfo_children()[2],  # Quick Save Results button
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
        self.create_tooltip(
            analysis_options_frame.winfo_children()[0],  # Time-Resolved Analysis button
            "Display peak properties changes over time and throughput analysis"
        )
        self.create_tooltip(
            analysis_options_frame.winfo_children()[1],  # Peak Property Correlations button
            "Display correlation plots between peak width, height, and area"
        )
        self.create_tooltip(
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
        
        # Add a welcome label
        welcome_label = ttk.Label(self.blank_tab, 
                                text="Welcome to Peak Analysis Tool\n\nPlease load a file to begin", 
                                font=("Arial", 14))
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

    # Create tooltips for GUI elements
    def create_tooltip(self, widget, text):
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+0+0")
        tooltip.withdraw()
        label = ttk.Label(tooltip, text=text, background="lightyellow", relief=tk.SOLID, borderwidth=1)
        label.pack(ipadx=1)

        def show_tooltip(event):
            x = event.x_root + 10
            y = event.y_root + 10
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def hide_tooltip(event):
            tooltip.withdraw()

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)

        return tooltip

    # Function to browse and load the file
    @profile_function
    def browse_file(self):
        """Browse and load file(s) based on current mode"""
        print(f"Memory before loading: {get_memory_usage():.2f} MB")
        print(f"Current file mode: {self.file_mode.get()}")
        
        try:
            # Reset progress bar
            self.update_progress_bar(0)
            
            files = []
            if self.file_mode.get() == "single":
                files = list(filedialog.askopenfilenames(
                    title="Select Data File",
                    filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
                ))
            else:  # batch mode
                folder = filedialog.askdirectory(title="Select Folder with Data Files")
                if folder:
                    files = [os.path.join(folder, f) for f in os.listdir(folder) 
                            if f.endswith('.txt')]
            
            if files:
                files = list(Tcl().call('lsort', '-dict', files))
                self.preview_label.config(text="Loading files...", foreground="blue")
                self.update_idletasks()
                
                # Set maximum progress
                self.update_progress_bar(0, len(files))
                
                # Store file names
                self.loaded_files = [os.path.basename(f) for f in files]
                
                all_times = []
                all_amplitudes = []
                
                # Get timestamps if in batch mode
                timestamps = []
                if self.file_mode.get() == "batch":
                    timestamps = [t.strip() for t in self.batch_timestamps.get().split(',') if t.strip()]
                
                for i, file in enumerate(files):
                    print(f"Loading file {i+1}/{len(files)}: {file}")
                    
                    df = pd.read_csv(file, delimiter='\t')
                    df.columns = [col.strip() for col in df.columns]
                    
                    # Check if the file has standard time column name
                    has_timestamp = 'Time - Plot 0' in df.columns
                    
                    if has_timestamp:
                        time_col = 'Time - Plot 0'
                        amp_col = 'Amplitude - Plot 0'
                    else:
                        # Assume first column is time, second is amplitude
                        df.columns = ['Time - Plot 0', 'Amplitude - Plot 0']
                        time_col = 'Time - Plot 0'
                        amp_col = 'Amplitude - Plot 0'
                    
                    # Append data with time offset if in batch mode
                    if self.file_mode.get() == "batch" and timestamps:
                        # Convert timestamp to seconds from start
                        if i == 0:
                            start_time = timestamps_to_seconds([timestamps[0]], timestamps[0])[0]*1e4
                        current_time = timestamps_to_seconds([timestamps[i]], timestamps[0])[0]*1e4
                        time_offset = current_time
                        all_times.append(df[time_col].values + time_offset)
                    else:
                        # For single file or no timestamps, just concatenate
                        if all_times:
                            time_offset = all_times[-1][-1] + (df[time_col].iloc[1] - df[time_col].iloc[0])
                            all_times.append(df[time_col].values + time_offset)
                        else:
                            all_times.append(df[time_col].values)
                        
                    all_amplitudes.append(df[amp_col].values)
                    print(f"File {i+1} loaded. Time points: {len(df[time_col])}, Amplitude points: {len(df[amp_col])}")  # Debug print
                    
                    # Update progress
                    self.update_progress_bar(i + 1)
                
                # Combine all data
                self.t_value = np.concatenate(all_times)
                self.x_value = np.concatenate(all_amplitudes)
                
                print(f"Total data points after concatenation: {len(self.t_value)}")  # Debug print
                
                # Create combined DataFrame
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
                
                self.preview_label.config(text="Files loaded successfully", foreground="green")
                self.update_results_summary(preview_text=preview_text)
                
        except Exception as e:
            self.data = None
            self.t_value = None
            self.x_value = None
            self.show_error("Error loading files", e)
            print(f"Detailed error: {str(e)}")
            traceback.print_exc()
            self.update_progress_bar(0)  # Reset on error

    # Function to plot the raw data
    def plot_raw_data(self):
        if self.data is not None:
            try:
                # Initialize canvas if it doesn't exist
                if self.canvas is None:
                    self.canvas = FigureCanvasTkAgg(self.figure, self.plot_tab_control)

                # Clear the current figure
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                # Plot raw data with same style as smoothed data
                ax.plot(self.data['Time - Plot 0']*1e-4/60,  # Convert to minutes
                       self.data['Amplitude - Plot 0'],
                       color='black', 
                       linewidth=0.05,  # Thinner line for better visibility
                       label='Raw Data',
                       alpha=0.9)
                
                # Customize plot
                ax.set_xlabel('Time (min)', fontsize=12)
                ax.set_ylabel('Amplitude (counts)', fontsize=12)
                ax.set_title('Raw Data', fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(fontsize=10)
                
                # Adjust layout
                self.figure.tight_layout()

                # Update or create tab in plot_tab_control
                tab_exists = False
                for tab in self.plot_tab_control.tabs():
                    if self.plot_tab_control.tab(tab, "text") == "Raw Data":
                        self.plot_tab_control.select(tab)
                        tab_exists = True
                        break
                    
                if not tab_exists:
                    new_tab = ttk.Frame(self.plot_tab_control)
                    self.plot_tab_control.add(new_tab, text="Raw Data")
                    self.plot_tab_control.select(new_tab)
                    canvas = FigureCanvasTkAgg(self.figure, new_tab)
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                # Update the canvas
                self.canvas.draw_idle()
                
                self.preview_label.config(text="Raw data plotted successfully", foreground="green")

            except Exception as e:
                self.preview_label.config(text=f"Error plotting raw data: {e}", foreground="red")
        else:
            self.preview_label.config(text="No data to plot", foreground="red")

    # Function to start data analysis
    @profile_function
    def start_analysis(self):
        if self.data is None:
            self.show_error("No data loaded. Please load files first.")
            return
        
        try:
            # Initialize progress
            total_steps = 3
            self.update_progress_bar(0, total_steps)
            
            # Get parameters and prepare data
            normalization_factor = self.normalization_factor.get()
            big_counts = self.big_counts.get()
            current_cutoff = self.cutoff_value.get()
            
            t = self.data['Time - Plot 0']*1e-4
            x = self.data['Amplitude - Plot 0']
            rate = np.median(np.diff(t))
            self.fs = 1 / rate
            
            # Update progress
            self.update_progress_bar(1)
            
            # Apply filtering
            if current_cutoff > 0:
                self.filtered_signal = apply_butterworth_filter(2, current_cutoff, 'lowpass', self.fs, x)
                calculated_cutoff = current_cutoff
            else:
                self.filtered_signal, calculated_cutoff = adjust_lowpass_cutoff(x, self.fs, big_counts, normalization_factor)
                self.cutoff_value.set(calculated_cutoff)
            
            # Update progress
            self.update_progress_bar(2)
            
            # Plot results and finish up
            self.t_value = t
            self.x_value = x

            # Calculate suggested threshold
            signal_std = np.std(self.filtered_signal)
            suggested_threshold = 5 * signal_std

            # Update the height_lim entry
            self.height_lim.set(suggested_threshold)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot raw data
            ax.plot(t / 60, x, 
                    color='black', 
                    linewidth=0.05, 
                    label='Raw Data',
                    alpha=0.4)
            
            # Plot filtered data
            ax.plot(t / 60, self.filtered_signal, 
                    color='blue', 
                    linewidth=0.05, 
                    label='Filtered Data',
                    alpha=0.9)
            
            # Customize plot
            ax.set_xlabel('Time (min)', fontsize=12)
            ax.set_ylabel('Amplitude (counts)', fontsize=12)
            ax.set_title('Raw and Smoothed Signals', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            
            # Adjust layout
            self.figure.tight_layout()

            # Update or create tab in plot_tab_control
            tab_name = "Smoothed Data"
            tab_exists = False
            
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") == tab_name:
                    self.plot_tab_control.select(tab)
                    tab_exists = True
                    break
                    
            if not tab_exists:
                new_tab = ttk.Frame(self.plot_tab_control)
                self.plot_tab_control.add(new_tab, text=tab_name)
                self.plot_tab_control.select(new_tab)
                self.canvas = FigureCanvasTkAgg(self.figure, new_tab)
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Update the canvas
            self.canvas.draw_idle()

            # Save the plot
            self.figure.savefig('filtered_signal_plot.png', bbox_inches='tight')
            
            # Update progress for final step
            self.update_progress_bar(3)
            
            self.preview_label.config(
                text=f"Analysis completed successfully (Cutoff: {calculated_cutoff:.2f}, Suggested threshold: {suggested_threshold:.1f})", 
                foreground="green"
            )

        except Exception as e:
            self.show_error("Error during analysis", e)
            # Reset progress on error
            self.update_progress_bar(0)

    # Function to run peak detection
    @profile_function
    def run_peak_detection(self):
        if self.filtered_signal is None:
            self.show_error("Filtered signal not available. Please start the analysis first.")
            return
            
        try:
            # Initialize progress
            total_steps = 4
            self.update_progress_bar(0, total_steps)
            
            # Get parameters
            height_lim_factor = self.height_lim.get()
            distance = self.distance.get()
            rel_height = self.rel_height.get()
            width_p = [int(float(x) * 10) for x in self.width_p.get().split(',')]
            
            # Update progress
            self.update_progress_bar(1)
            
            # Find peaks
            peaks_x_filter, amp_x_filter = find_peaks_with_window(
                self.filtered_signal, 
                width=width_p, 
                prominence=height_lim_factor,
                distance=distance, 
                rel_height=rel_height
            )
            
            # Update progress
            self.update_progress_bar(2)
            
            # Check if any peaks were found
            if len(peaks_x_filter) == 0:
                self.show_error("No peaks found with current parameters. Try adjusting threshold or width range.")
                return
            
            events = len(peaks_x_filter)

            # Calculate peak areas
            window = np.round(amp_x_filter['widths'], 0).astype(int) + 40
            peak_areas = np.zeros(events)
            start_indices = np.zeros(events)
            end_indices = np.zeros(events)

            for i in range(events):
                # Get window indices
                start_idx = max(0, peaks_x_filter[i] - window[i])
                end_idx = min(len(self.filtered_signal), peaks_x_filter[i] + window[i])
                
                # Check if window is valid
                if start_idx >= end_idx:
                    self.show_error(f"Invalid window for peak {i}: [{start_idx}, {end_idx}]")
                    continue
                
                yData = self.filtered_signal[start_idx:end_idx]
                
                # Check if yData is not empty
                if len(yData) == 0:
                    self.show_error(f"Empty data window for peak {i}")
                    continue
                
                background = np.min(yData)

                st = int(amp_x_filter["left_ips"][i])
                en = int(amp_x_filter["right_ips"][i])
                
                # Validate indices
                if st >= en or st < 0 or en > len(self.filtered_signal):
                    self.show_error(f"Invalid peak boundaries for peak {i}: [{st}, {en}]")
                    continue

                start_indices[i] = st
                end_indices[i] = en

                peak_areas[i] = np.sum(self.filtered_signal[st:en] - background)

            # Update progress
            self.update_progress_bar(3)

            # Calculate peak intervals
            peak_times = self.t_value[peaks_x_filter]
            intervals = np.diff(peak_times)
            
            if len(intervals) > 0:  # Check if we have any intervals
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
            else:
                mean_interval = 0
                std_interval = 0

            # Update results summary
            summary_text = (
                f"Number of peaks detected: {events}\n"
                f"Average peak area: {np.mean(peak_areas):.2f} ± {np.std(peak_areas):.2f}\n"
                f"Average interval: {mean_interval:.2f} ± {std_interval:.2f} seconds\n"
                f"Peak detection threshold: {height_lim_factor}"
            )
            self.update_results_summary(summary_text)
            
            self.preview_label.config(
                text=f"Peak detection completed: {events} peaks detected", 
                foreground="green"
            )

            # Plot results
            self.plot_peak_detection_results(peaks_x_filter, peak_areas)

            # Final progress update
            self.update_progress_bar(4)

        except Exception as e:
            self.show_error("Error during peak detection", e)
            self.update_progress_bar(0)  # Reset on error

    def plot_peak_detection_results(self, peaks_x_filter, peak_areas):
        """Separate method for plotting peak detection results"""
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot raw data in black
            ax.plot(self.t_value / 60, self.x_value, 
                    color='black', 
                    linewidth=0.05, 
                    label='Raw Data',
                    alpha=0.7)
            
            # Plot filtered data in blue
            ax.plot(self.t_value / 60, self.filtered_signal, 
                    color='blue', 
                    linewidth=0.1,
                    label='Filtered Data',
                    alpha=0.8)
            
            # Plot detected peaks in red
            ax.scatter(self.t_value[peaks_x_filter] / 60, 
                      self.filtered_signal[peaks_x_filter],
                      color='red',
                      s=5,
                      label=f'Detected Peaks ({len(peaks_x_filter)})',
                      zorder=5)
            
            # Customize the plot
            ax.set_xlabel('Time (min)', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            ax.set_title('Raw and Filtered Signals with Detected Peaks', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            
            # Adjust layout
            self.figure.tight_layout()

            # Update the canvas
            self.canvas.draw_idle()

        except Exception as e:
            self.show_error("Error plotting peak detection results", e)

    # Function to plot the detected filtered peaks
    @profile_function
    def plot_filtered_peaks(self):
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            # Convert pandas Series to numpy arrays if needed
            t = np.array(self.t_value)
            filtered_signal = np.array(self.filtered_signal)
            x = np.array(self.x_value)
            
            height_threshold = self.height_lim.get()
            distance = self.distance.get()
            rel_height = self.rel_height.get()

            # Debug print
            print(f"Debug info:")
            print(f"Signal range: {np.min(filtered_signal)} to {np.max(filtered_signal)}")
            print(f"Height threshold: {height_threshold}")
            print(f"Distance: {distance}")
            print(f"Rel height: {rel_height}")

            width_p = [int(float(x) * 10) for x in self.width_p.get().split(',')]
            print(f"Width range: {width_p}")

            peaks_x_filter, amp_x_filter = find_peaks_with_window(
                filtered_signal, 
                width=width_p, 
                prominence=height_threshold,
                distance=distance, 
                rel_height=rel_height
            )

            if len(peaks_x_filter) == 0:
                self.preview_label.config(
                    text="No peaks found. Try adjusting parameters (lower threshold or wider width range)",
                    foreground="red"
                )
                return

            print(f"Number of peaks found: {len(peaks_x_filter)}")

            window = np.round(amp_x_filter['widths'], 0).astype(int) + 50
            num_peaks_to_plot = min(10, len(peaks_x_filter))
            values = np.round(np.linspace(0, len(peaks_x_filter) - 1, num_peaks_to_plot)).astype(int)
            
            new_figure = Figure(figsize=(10, 8))
            axs = []
            for i in range(2):
                row = []
                for j in range(5):
                    row.append(new_figure.add_subplot(2, 5, i*5 + j + 1))
                axs.append(row)

            handles, labels = [], []

            for idx, val in enumerate(values):
                i = val
                # Add bounds checking
                start_idx = max(0, peaks_x_filter[i] - window[i])
                end_idx = min(len(t), peaks_x_filter[i] + window[i])
                
                # Extract data slices as numpy arrays
                xData = t[start_idx:end_idx]
                yData_sub = filtered_signal[start_idx:end_idx]
                
                if len(xData) == 0:
                    continue
                    
                background = np.min(yData_sub)
                yData = yData_sub - background

                ax = axs[idx // 5][idx % 5]

                # Plot filtered data
                line1, = ax.plot((xData - xData[0]) * 1e3, yData, 
                               color='blue', 
                               label='Filtered', 
                               alpha=0.8,
                               linewidth=0.5)
                
                # Plot peak marker
                peak_time = t[peaks_x_filter[i]]
                peak_height = filtered_signal[peaks_x_filter[i]] - background
                line2, = ax.plot((peak_time - xData[0]) * 1e3, 
                               peak_height,
                               "x", 
                               color='red', 
                               ms=10, 
                               label='Peak')

                # Plot raw data
                raw_data = x[start_idx:end_idx]
                corrected_signal = raw_data - background
                line3, = ax.plot((xData - xData[0]) * 1e3, 
                               corrected_signal, 
                               color='black', 
                               label='Raw', 
                               alpha=0.5,
                               linewidth=0.3)

                # Plot width lines
                left_idx = int(amp_x_filter["left_ips"][i])
                right_idx = int(amp_x_filter["right_ips"][i])
                width_height = amp_x_filter["width_heights"][i] - background
                
                line4 = ax.hlines(y=width_height,
                                xmin=(t[left_idx] - xData[0]) * 1e3,
                                xmax=(t[right_idx] - xData[0]) * 1e3,
                                color="red",
                                linestyles='-',
                                alpha=0.8)
                line4 = Line2D([0], [0], color='red', linestyle='-', label='Peak Width')

                # Customize subplot
                ax.set_xlabel('Time (ms)', fontsize=10)
                ax.set_ylabel('Counts', fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.tick_params(axis='both', labelsize=9)
                
                # Add padding to y-axis limits
                ymin, ymax = ax.get_ylim()
                y_padding = (ymax - ymin) * 0.15
                ax.set_ylim(ymin - y_padding, ymax + y_padding)

                if idx == 0:
                    handles.extend([line3, line1, line2, line4])

            # Add legend
            new_figure.legend(handles, 
                               ['Raw Data', 'Filtered Data', 'Peak', 'Peak Width'], 
                               loc='center right',
                               bbox_to_anchor=(0.98, 0.5),
                               fontsize=10)

            new_figure.suptitle('Individual Peak Analysis', fontsize=14, y=1.02)
            new_figure.tight_layout()

            # Update or create tab in plot_tab_control
            tab_name = "Exemplary Peaks"  # Changed from "Filtered Peaks Plot"
            tab_exists = False
            
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") == tab_name:
                    self.plot_tab_control.forget(tab)  # Remove existing tab to update it
                    break
                    
            new_tab = ttk.Frame(self.plot_tab_control)
            self.plot_tab_control.add(new_tab, text=tab_name)
            self.plot_tab_control.select(new_tab)
            
            new_canvas = FigureCanvasTkAgg(new_figure, new_tab)
            new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            new_canvas.draw_idle()

            # Save the plot
            new_figure.savefig('filtered_peaks_plot.png', bbox_inches='tight')
            self.preview_label.config(
                text=f"Example peaks plotted successfully. Found {len(peaks_x_filter)} peaks.", 
                foreground="green"
            )

        except Exception as e:
            self.preview_label.config(text=f"Error plotting example peaks: {str(e)}", foreground="red")
            print(f"Detailed error: {str(e)}")
            import traceback
            traceback.print_exc()

    # Function to calculate the areas of detected peaks
    def calculate_peak_areas(self):
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            filtered_signal = self.filtered_signal
            t = self.t_value
            height_lim_factor = self.height_lim.get()
            distance = self.distance.get()
            rel_height = self.rel_height.get()

            # First convert string to float, then multiply by 10 and convert to int
            width_values = self.width_p.get().strip().split(',')
            width_p = [int(float(value.strip()) * 10) for value in width_values]
            
            peaks_x_filter, amp_x_filter = find_peaks_with_window(
                filtered_signal, 
                width=width_p,
                prominence=height_lim_factor,
                distance=distance, 
                rel_height=rel_height
            )

            window = np.round(amp_x_filter['widths'], 0).astype(int) + 40
            events = len(peaks_x_filter)
            
            peak_area = np.zeros(events)
            start = np.zeros(events)
            end = np.zeros(events)
            Datay = filtered_signal

            for i in range(events):
                yData = Datay[peaks_x_filter[i] - window[i]:peaks_x_filter[i] + window[i]]
                background = np.min(yData)

                st = int(amp_x_filter["left_ips"][i])
                en = int(amp_x_filter["right_ips"][i])

                start[i] = st
                end[i] = en

                peak_area[i] = np.sum(filtered_signal[st:en] - background)
        
            self.preview_label.config(text="Peak area calculation completed", foreground="green")
            self.update_results_summary(peak_areas=peak_area)
            return peak_area, start, end 

        except Exception as e:
            self.preview_label.config(text=f"Error calculating peak areas: {e}", foreground="red")
    
    # Function to save peak information to CSV
    def save_peak_information_to_csv(self):
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            height_lim_factor = self.height_lim.get()
            distance = self.distance.get()
            rel_height = self.rel_height.get()
            
            # Convert width values properly
            width_values = [float(x) for x in self.width_p.get().split(',')]
            width_p = [int(float(x) * 10) for x in width_values]  # Convert to samples

            peaks_x_filter, amp_x_filter = find_peaks_with_window(
                self.filtered_signal, 
                width=width_p, 
                prominence=height_lim_factor,
                distance=distance, 
                rel_height=rel_height
            )

            # Calculate peak areas
            window = np.round(amp_x_filter['widths'], 0).astype(int) + 40
            peak_areas = np.zeros(len(peaks_x_filter))
            
            for i in range(len(peaks_x_filter)):
                yData = self.filtered_signal[peaks_x_filter[i] - window[i]:peaks_x_filter[i] + window[i]]
                background = np.min(yData)
                st = int(amp_x_filter["left_ips"][i])
                en = int(amp_x_filter["right_ips"][i])
                peak_areas[i] = np.sum(self.filtered_signal[st:en] - background)

            # Create DataFrame with results
            results_df = pd.DataFrame({
                "Peak Time (min)": self.t_value[peaks_x_filter] / 60,  # Convert to minutes
                "Peak Area": peak_areas,
                "Peak Width (ms)": amp_x_filter['widths'] / 10,  # Convert to milliseconds
                "Peak Height": amp_x_filter['prominences'],
                "Start Time": self.protocol_start_time.get(),
                "Particle": self.protocol_particle.get(),
                "Concentration": self.protocol_concentration.get(),
                "Stamp": self.protocol_stamp.get(),
                "Laser Power": self.protocol_laser_power.get(),
                "Setup": self.protocol_setup.get(),
                "Notes": self.protocol_notes.get()
            })

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
        
    # Function to calculate intervals between detected peaks
    def calculate_peak_intervals(self):
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            t = self.t_value
            height_lim_factor = self.height_lim.get()
            
            # First convert string to float, then multiply by 10 and convert to int
            width_values = self.width_p.get().strip().split(',')
            width_p = [int(float(value.strip()) * 10) for value in width_values]
            
            peaks_x_filter, amp_x_filter = find_peaks_with_window(
                self.filtered_signal, 
                width=width_p,
                prominence=height_lim_factor ,
                distance=self.distance.get(), 
                rel_height=self.rel_height.get()
            )

            intervall = 10
            points = len(np.arange(1, int(max(t)), intervall))
            peaks = np.zeros(points)
            peaks_intervall = np.zeros(points)
            index = 0
            position = np.arange(1, int(max(t)), intervall)
        
            for i in position:
                peaks[index] = find_nearest(t[peaks_x_filter], i)
                if i == 0:
                    peaks_intervall[index] = peaks[index]
                    index += 1
                else:
                    peaks_intervall[index] = peaks[index] - peaks[index - 1]
                    index += 1
            
            self.preview_label.config(text="Peak interval calculation completed", foreground="green")
            self.update_results_summary(peak_intervals=peaks_intervall)
            return peaks, peaks_intervall
        
        except Exception as e:
            self.preview_label.config(text=f"Error calculating peak intervals: {e}", foreground="red")
    
    # Function to plot processed data with detected peaks
    def plot_data(self):
        """Plot processed data with peaks in a new tab."""
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            print("Starting plot_data function...")
            
            # Create a new figure for data plot
            self.data_figure = Figure(figsize=(10, 8))
            
            # Create subplots
            axes = self.data_figure.subplots(nrows=4, ncols=1, sharex=True)
            
            # Convert data to float32 for memory efficiency
            t = np.asarray(self.t_value, dtype=np.float32)
            filtered_signal = np.asarray(self.filtered_signal, dtype=np.float32)
            
            # Find peaks with optimized parameters
            width_values = self.width_p.get().strip().split(',')
            width_p = [int(float(value.strip()) * 10) for value in width_values]
            
            peaks, properties = find_peaks_with_window(
                filtered_signal,
                width=width_p,
                prominence=self.height_lim.get(),
                distance=self.distance.get(),
                rel_height=self.rel_height.get()
            )
            
            # Calculate peak properties
            widths = properties["widths"]
            prominences = properties["prominences"]
            peak_times = t[peaks]
            
            # Calculate areas under peaks
            areas = []
            for i, peak in enumerate(peaks):
                left_idx = int(properties["left_ips"][i])
                right_idx = int(properties["right_ips"][i])
                if left_idx < right_idx:
                    areas.append(np.trapz(filtered_signal[left_idx:right_idx]))
                else:
                    areas.append(0)
            areas = np.array(areas)
            
            # Plot peak heights
            axes[0].scatter(peak_times/60, prominences, s=1, alpha=0.5, color='black', label='Peak Heights')
            axes[0].set_ylabel('Peak Heights')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=8)
            axes[0].set_yscale('log')
            
            # Plot peak widths
            axes[1].scatter(peak_times/60, widths/10, s=1, alpha=0.5, color='black', label='Peak Widths (ms)')
            axes[1].set_ylabel('Peak Widths (ms)')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8)
            axes[1].set_yscale('log')
            
            # Plot peak areas
            axes[2].scatter(peak_times/60, areas, s=1, alpha=0.5, color='black', label='Peak Areas')
            axes[2].set_ylabel('Peak Areas')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(fontsize=8)
            axes[2].set_yscale('log')
            
            # Calculate and plot throughput
            interval = 10  # seconds
            bins = np.arange(0, np.max(t), interval)
            throughput, _ = np.histogram(peak_times, bins=bins)
            axes[3].plot((bins[:-1] + interval/2)/60, throughput, color='black', alpha=0.5, 
                        label=f'Throughput ({interval}s bins)')
            axes[3].set_ylabel(f'Peaks per {interval}s')
            axes[3].set_xlabel('Time (min)')
            axes[3].grid(True, alpha=0.3)
            axes[3].legend(fontsize=8)
            
            # Update title and layout
            self.data_figure.suptitle('Peak Analysis Over Time', y=0.95)
            self.data_figure.tight_layout()
            
            # Create or update the tab in plot_tab_control
            tab_name = "Peak Analysis"
            tab_exists = False
            
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") == tab_name:
                    self.plot_tab_control.forget(tab)  # Remove existing tab to update it
                    break
                    
            new_tab = ttk.Frame(self.plot_tab_control)
            self.plot_tab_control.add(new_tab, text=tab_name)
            self.plot_tab_control.select(new_tab)
            
            # Create new canvas in the tab
            self.data_canvas = FigureCanvasTkAgg(self.data_figure, new_tab)
            self.data_canvas.draw()
            self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.preview_label.config(text="Peak analysis plot created successfully", foreground="green")
            
        except Exception as e:
            self.show_error("Error in plot_data", e)
            raise  # Re-raise the exception for debugging

    def update_data_plot_display(self, title):
        """Update the data plot display with proper layout and canvas."""
        try:
            # Clear any existing data canvas
            if hasattr(self, 'data_canvas') and self.data_canvas is not None:
                self.data_canvas.get_tk_widget().pack_forget()

            # Create new canvas for data plot
            self.data_canvas = FigureCanvasTkAgg(self.data_figure, master=self)
            self.data_canvas.draw()
            
            # Configure canvas layout
            canvas_widget = self.data_canvas.get_tk_widget()
            canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Store original view limits
            axes = self.data_figure.get_axes()
            self.data_original_xlims = [ax.get_xlim() for ax in axes]
            self.data_original_ylims = [ax.get_ylim() for ax in axes]
            
            # Update title and layout
            self.data_figure.suptitle(title)
            self.data_figure.tight_layout()
            
            # Enable zoom functionality for data plot
            self.enable_data_zoom()
            
            # Update status
            self.preview_label.config(text=f"Data plot updated: {title}", foreground="green")
            
        except Exception as e:
            self.show_error("Error updating data plot display", e)

    def enable_data_zoom(self):
        """Enable zoom functionality for data plot."""
        try:
            if hasattr(self, 'data_canvas'):
                self.data_canvas.mpl_connect('key_press_event', self.on_data_key_press)
                for ax in self.data_figure.get_axes():
                    self.data_canvas.mpl_connect('button_press_event', self.on_data_mouse_click)
                    self.data_canvas.mpl_connect('button_release_event', self.on_data_mouse_release)
                    self.data_canvas.mpl_connect('motion_notify_event', self.on_data_mouse_motion)
        except Exception as e:
            self.show_error("Error enabling data zoom", e)

    def on_data_key_press(self, event):
        """Handle key press events for data plot."""
        if event.key == 'r':  # Reset zoom on 'r' key press
            for ax in self.data_figure.get_axes():
                ax.autoscale()
            self.data_canvas.draw()

   

    

    # Function to export the current plot
    def export_plot(self):
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")])
            if file_path:
                self.figure.savefig(file_path)
                self.preview_label.config(text=f"Plot exported successfully to {file_path}", foreground="green")
            else:
                self.preview_label.config(text="No file path selected", foreground="red")
        except Exception as e:
            self.preview_label.config(text=f"Error exporting plot: {e}", foreground="red")

    # Function to reset the view of the plot
 

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

    # Add new optimized plotting function
    def plot_optimized(self, x: np.ndarray, y: np.ndarray, ax: plt.Axes, **plot_kwargs):
        """Plot large datasets efficiently using decimation."""
        n_points = len(x)
        
        if n_points > 100000:
            # Decimate data for plotting
            stride = n_points // 100000
            x_plot = x[::stride]
            y_plot = y[::stride]
        else:
            x_plot, y_plot = x, y
        
        # Use more efficient line plotting
        ax.plot(x_plot, y_plot, **plot_kwargs)

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
        """Display error message in GUI and console."""
        error_message = f"=== ERROR DETAILS ===\n"
        error_message += f"Error Message: {title}: {str(error)}\n"
        error_message += f"Error Type: {type(error).__name__}\n"
        error_message += f"Traceback:\n{traceback.format_exc()}"
        error_message += f"===================\n"
        
        print(error_message)  # Print to console
        self.preview_label.config(text=f"{title}: {str(error)}", foreground="red")

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
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            # Get peaks and properties
            width_values = self.width_p.get().strip().split(',')
            width_p = [int(float(value.strip()) * 10) for value in width_values]
            
            peaks_x_filter, properties = find_peaks_with_window(
                self.filtered_signal, 
                width=width_p,
                prominence=self.height_lim.get(),
                distance=self.distance.get(), 
                rel_height=self.rel_height.get()
            )

            # Calculate peak areas
            window = np.round(properties['widths'], 0).astype(int) + 40
            peak_areas = np.zeros(len(peaks_x_filter))
            
            for i in range(len(peaks_x_filter)):
                yData = self.filtered_signal[peaks_x_filter[i] - window[i]:peaks_x_filter[i] + window[i]]
                background = np.min(yData)
                st = int(properties["left_ips"][i])
                en = int(properties["right_ips"][i])
                peak_areas[i] = np.sum(self.filtered_signal[st:en] - background)

            # Create DataFrame with all peak properties
            df_all = pd.DataFrame({
                "width": properties['widths'],
                "amplitude": properties['prominences'],
                "area": peak_areas
            })

            # Create new figure
            new_figure = Figure(figsize=(10, 8))
            ax = new_figure.subplots(nrows=2, ncols=2)

            # Create scatter plots
            ax[0, 0].scatter(df_all['width'] / 10, df_all['amplitude'], s=1, alpha=0.5, color='black')
            ax[0, 0].set_xlabel('Width (ms)')
            ax[0, 0].set_ylabel('Amplitude (counts)')
            ax[0, 0].grid(True, alpha=0.3)
            ax[0, 0].set_xscale('log')
            ax[0, 0].set_yscale('log')

            ax[0, 1].scatter(df_all['width'] / 10, df_all['area'], s=1, alpha=0.5, color='black')
            ax[0, 1].set_xlabel('Width (ms)')
            ax[0, 1].set_ylabel('Area (counts)')
            ax[0, 1].grid(True, alpha=0.3)
            ax[0, 1].set_yscale('log')
            ax[0, 1].set_xscale('log')

            ax[1, 0].scatter(df_all['amplitude'], df_all['area'], s=1, alpha=0.5, color='black')
            ax[1, 0].set_xlabel('Amplitude (counts)')
            ax[1, 0].set_ylabel('Area (counts)')
            ax[1, 0].grid(True, alpha=0.3)
            ax[1, 0].set_xscale('log')
            ax[1, 0].set_yscale('log')

            # Calculate and plot histogram
            bins = np.linspace(min(df_all['width'] / 10), max(df_all['width'] / 10), 50)
            ax[1, 1].hist(df_all['width'] / 10, bins=bins, alpha=0.5, color='black')
            ax[1, 1].set_xlabel('Width (ms)')
            ax[1, 1].set_ylabel('Counts')
            ax[1, 1].grid(True, alpha=0.3)

            new_figure.suptitle('Peak Property Correlations', y=0.95)
            new_figure.tight_layout()

            # Update or create tab in plot_tab_control
            tab_name = "Peak Properties"
            tab_exists = False
            
            for tab in self.plot_tab_control.tabs():
                if self.plot_tab_control.tab(tab, "text") == tab_name:
                    self.plot_tab_control.forget(tab)  # Remove existing tab to update it
                    break
                    
            new_tab = ttk.Frame(self.plot_tab_control)
            self.plot_tab_control.add(new_tab, text=tab_name)
            self.plot_tab_control.select(new_tab)
            
            new_canvas = FigureCanvasTkAgg(new_figure, new_tab)
            new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            new_canvas.draw_idle()

            self.preview_label.config(text="Peak properties plotted successfully", foreground="green")

        except Exception as e:
            self.preview_label.config(text=f"Error creating scatter plot: {e}", foreground="red")

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
                self.after(500, lambda: self.update_progress_bar(0))  # Reset after 500ms
        except Exception as e:
            print(f"Error updating progress bar: {e}")

if __name__ == "__main__":
    app = Application()
    app.mainloop()