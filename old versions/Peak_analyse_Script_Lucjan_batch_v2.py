# -*- coding: utf-8 -*-
"""
Created Nov 12 02:54:48 2024

@author:Lucjan & silas
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
def timestamps_to_seconds(timestamps, starttime):
    seconds = []
    start_min, start_sec = map(int, starttime.split(':'))
    start_time = start_min * 60 + start_sec 
    print(f'Start time in seconds: {start_time}')
    
    for timestamp in timestamps:
        minu, sec = map(int, timestamp.split(':'))
        seconds.append(minu * 60 + sec - start_time)
        
    print(f'Timestamps in seconds: {seconds}')
    return seconds

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
    avg_width = estimate_peak_widths(signal, fs, big_counts)
    print(f'Average width of peaks: {avg_width}')
    
    cutoff = normalization_factor / avg_width
    print(f'Normalization factor: {normalization_factor}')
    print(f'Adjusted cutoff frequency: {cutoff}')
    
    cutoff = max(min(cutoff, 10000), 50)
    print(f'Final cutoff frequency after limits applied: {cutoff}')
    
    # Set the cutoff frequency if it has been manually adjusted
    if app.cutoff_value.get() != 0:
        cutoff = app.cutoff_value.get()
        print(f'Using manual cutoff frequency: {cutoff}')
    
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
        self.normalization_factor = tk.IntVar(value=1)  # Renamed to Automatic Cutoff Factor
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

        self.create_widgets()
        self.blank_tab_exists = True  # Track if the blank tab exists

        self.setup_performance_logging()

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

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        control_frame.columnconfigure(1, weight=1)

        # Load file section
        self.browse_button = ttk.Button(control_frame, text="Load file", command=self.browse_file)
        self.browse_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Create label for filename (not using StringVar)
        self.file_name_label = ttk.Label(control_frame, text="No file selected")
        self.file_name_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Preprocessing category
        preprocessing_label = ttk.Label(control_frame, text="Preprocessing", font=("Arial", 12, "bold"))
        preprocessing_label.grid(row=1, column=0, padx=5, pady=5, sticky="w", columnspan=2)

        # Cutoff Factor
        self.normalization_factor_label = ttk.Label(control_frame, text="Automatic Cutoff Factor")
        self.normalization_factor_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.normalization_factor_entry = ttk.Entry(control_frame, textvariable=self.normalization_factor)
        self.normalization_factor_entry.grid(row=2, column=1, padx=5, pady=5)

        # Start Time
        self.start_time_label = ttk.Label(control_frame, text="Start Time (MM:SS)")
        self.start_time_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.start_time_entry = ttk.Entry(control_frame, textvariable=self.start_time)
        self.start_time_entry.grid(row=3, column=1, padx=5, pady=5)

        # Biggest Peaks
        self.big_counts_label = ttk.Label(control_frame, text="Biggest Peaks")
        self.big_counts_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.big_counts_entry = ttk.Entry(control_frame, textvariable=self.big_counts)
        self.big_counts_entry.grid(row=4, column=1, padx=5, pady=5)

        # Cutoff Value
        cutoff_frame = ttk.Frame(control_frame)
        cutoff_frame.grid(row=5, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        
        self.cutoff_label = ttk.Label(cutoff_frame, text="Cutoff Frequency")
        self.cutoff_label.pack(side="left", padx=(0,5))
        
        self.auto_cutoff_btn = ttk.Button(
            cutoff_frame, 
            text="Auto", 
            command=self.calculate_auto_cutoff_frequency,
            width=6
        )
        self.auto_cutoff_btn.pack(side="left", padx=(0,5))
        
        self.cutoff_entry = ttk.Entry(control_frame, textvariable=self.cutoff_value)
        self.cutoff_entry.grid(row=5, column=1, padx=5, pady=5)

        # Add tooltip
        self.create_tooltip(
            self.auto_cutoff_btn, 
            "Calculate suggested cutoff frequency based on peak widths"
        )

        # Buttons under Preprocessing
        self.plot_button = ttk.Button(control_frame, text="Plot Raw Data", command=self.plot_raw_data)
        self.plot_button.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.run_button = ttk.Button(control_frame, text="Run Analysis", command=self.start_analysis)
        self.run_button.grid(row=6, column=1, padx=5, pady=5)

        # Peak detection category
        postprocessing_label = ttk.Label(control_frame, text="Peak detection", font=("Arial", 12, "bold"))
        postprocessing_label.grid(row=7, column=0, padx=5, pady=5, sticky="w", columnspan=2)

        # Counts Threshold
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.grid(row=8, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        
        self.height_lim_label = ttk.Label(threshold_frame, text="Counts Threshold")
        self.height_lim_label.pack(side="left", padx=(0,5))
        
        self.auto_threshold_btn = ttk.Button(
            threshold_frame, 
            text="Auto", 
            command=self.calculate_auto_threshold,
            width=6
        )
        self.auto_threshold_btn.pack(side="left", padx=(0,5))
        
        self.height_lim_entry = ttk.Entry(control_frame, textvariable=self.height_lim)
        self.height_lim_entry.grid(row=8, column=1, padx=5, pady=5)

        # Add tooltip
        self.create_tooltip(
            self.auto_threshold_btn, 
            "Calculate suggested threshold based on signal standard deviation (5σ)"
        )

        # Min. Distance Peaks
        self.distance_label = ttk.Label(control_frame, text="Min. Distance Peaks")
        self.distance_label.grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.distance_entry = ttk.Entry(control_frame, textvariable=self.distance)
        self.distance_entry.grid(row=9, column=1, padx=5, pady=5)

        # Relative Height
        self.rel_height_label = ttk.Label(control_frame, text="Relative Height")
        self.rel_height_label.grid(row=10, column=0, padx=5, pady=5, sticky="w")
        self.rel_height_entry = ttk.Entry(control_frame, textvariable=self.rel_height)
        self.rel_height_entry.grid(row=10, column=1, padx=5, pady=5)

        # Width Range
        self.width_p_label = ttk.Label(control_frame, text="Width Range (ms, comma separated)")
        self.width_p_label.grid(row=11, column=0, padx=5, pady=5, sticky="w")
        self.width_p_entry = ttk.Entry(control_frame, textvariable=self.width_p)
        self.width_p_entry.grid(row=11, column=1, padx=5, pady=5)

        # Add tooltip for width range
        self.create_tooltip(self.width_p_entry, 
            "Enter min,max width in milliseconds (e.g. 0.1,200). Values will be converted to samples internally.")

        # Buttons under Peak Detection
        self.run_peak_button = ttk.Button(control_frame, text="Run Peak Detection", command=self.run_peak_detection)
        self.run_peak_button.grid(row=12, column=0, padx=5, pady=5, sticky="w")
        self.plot_filtered_button = ttk.Button(control_frame, text="Plot Filtered Peaks", command=self.plot_filtered_peaks)
        self.plot_filtered_button.grid(row=12, column=1, padx=5, pady=5)

        self.save_peak_button = ttk.Button(control_frame, text="Save Peak Information", command=self.save_peak_information_to_csv)
        self.save_peak_button.grid(row=13, column=1, padx=5, pady=5)

        # Time traces category
        time_traces_label = ttk.Label(control_frame, text="Time traces", font=("Arial", 12, "bold"))
        time_traces_label.grid(row=15, column=0, padx=5, pady=5, sticky="w", columnspan=2)

        self.plot_data_button = ttk.Button(control_frame, text="Plot Data", command=self.plot_data)
        self.plot_data_button.grid(row=16, column=0, padx=5, pady=5, sticky="w")
        self.plot_scatter_button = ttk.Button(control_frame, text="Plot Scatter", command=self.plot_scatter)
        self.plot_scatter_button.grid(row=16, column=1, padx=5, pady=5)

        self.preview_label = ttk.Label(control_frame, text="", foreground="blue")
        self.preview_label.grid(row=17, column=0, columnspan=2, pady=5)

        # Create a frame for the results summary
        results_frame = ttk.LabelFrame(control_frame, text="Analysis Results")
        results_frame.grid(row=25, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        results_frame.columnconfigure(0, weight=1)
        
        # Add the ScrolledText widget inside the frame
        self.results_summary = ScrolledText(
            results_frame, 
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
            background='white',
            relief=tk.SUNKEN,
            borderwidth=1
        )
        self.results_summary.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Add a clear button for the results
        ttk.Button(
            results_frame,
            text="Clear Results",
            command=lambda: self.update_results_summary(preview_text="")
        ).grid(row=1, column=0, pady=(0,5))

        # Protocol Category (adjust row numbers as needed)
        protocol_label = ttk.Label(control_frame, text="Protocol", font=("Arial", 12, "bold"))
        protocol_label.grid(row=30, column=0, padx=5, pady=5, sticky="w", columnspan=2)  # Increased row number

        # Protocol Inputs (adjust subsequent row numbers)
        self.protocol_start_time_label = ttk.Label(control_frame, text="Start Time (HH:MM:SS Date)")
        self.protocol_start_time_label.grid(row=31, column=0, padx=5, pady=5, sticky="w")
        self.protocol_start_time_entry = ttk.Entry(control_frame, textvariable=self.protocol_start_time)
        self.protocol_start_time_entry.grid(row=31, column=1, padx=5, pady=5)

        self.protocol_particle_label = ttk.Label(control_frame, text="Particle")
        self.protocol_particle_label.grid(row=32, column=0, padx=5, pady=5, sticky="w")
        self.protocol_particle_entry = ttk.Entry(control_frame, textvariable=self.protocol_particle)
        self.protocol_particle_entry.grid(row=32, column=1, padx=5, pady=5)

        self.protocol_concentration_label = ttk.Label(control_frame, text="Concentration")
        self.protocol_concentration_label.grid(row=33, column=0, padx=5, pady=5, sticky="w")
        self.protocol_concentration_entry = ttk.Entry(control_frame, textvariable=self.protocol_concentration)
        self.protocol_concentration_entry.grid(row=33, column=1, padx=5, pady=5)

        self.protocol_stamp_label = ttk.Label(control_frame, text="Stamp")
        self.protocol_stamp_label.grid(row=34, column=0, padx=5, pady=5, sticky="w")
        self.protocol_stamp_entry = ttk.Entry(control_frame, textvariable=self.protocol_stamp)
        self.protocol_stamp_entry.grid(row=34, column=1, padx=5, pady=5)

        self.protocol_laser_power_label = ttk.Label(control_frame, text="Laser Power")
        self.protocol_laser_power_label.grid(row=35, column=0, padx=5, pady=5, sticky="w")
        self.protocol_laser_power_entry = ttk.Entry(control_frame, textvariable=self.protocol_laser_power)
        self.protocol_laser_power_entry.grid(row=35, column=1, padx=5, pady=5)

        self.protocol_setup_label = ttk.Label(control_frame, text="Setup")
        self.protocol_setup_label.grid(row=36, column=0, padx=5, pady=5, sticky="w")
        self.protocol_setup_entry = ttk.Entry(control_frame, textvariable=self.protocol_setup)
        self.protocol_setup_entry.grid(row=36, column=1, padx=5, pady=5)

        self.protocol_notes_label = ttk.Label(control_frame, text="Notes")
        self.protocol_notes_label.grid(row=37, column=0, padx=5, pady=5, sticky="w")
        self.protocol_notes_entry = ttk.Entry(control_frame, textvariable=self.protocol_notes)
        self.protocol_notes_entry.grid(row=37, column=1, padx=5, pady=5)

        # Add file mode selection after the Load File button
        file_mode_frame = ttk.Frame(control_frame)
        file_mode_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Radiobutton(file_mode_frame, text="Single File", 
                       variable=self.file_mode, value="single").pack(side=tk.LEFT)
        ttk.Radiobutton(file_mode_frame, text="Batch Files", 
                       variable=self.file_mode, value="batch").pack(side=tk.LEFT)
        
        # Add timestamps entry for batch mode
        self.timestamps_label = ttk.Label(control_frame, text="Timestamps (comma-separated)")
        self.timestamps_entry = ttk.Entry(control_frame, textvariable=self.batch_timestamps)
        
        # Initially hide timestamp widgets
        self.timestamps_label.grid_remove()
        self.timestamps_entry.grid_remove()
        
        # Bind mode change
        self.file_mode.trace('w', self.on_file_mode_change)

        # Preview Frame with Functional Bar
        preview_frame = ttk.Frame(main_frame)
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=0)

        # Tab Control for Multiple Plots
        self.tab_control = ttk.Notebook(preview_frame)
        self.tab_control.grid(row=0, column=0, sticky="nsew")
        self.tab_control.rowconfigure(0, weight=1)
        self.tab_control.columnconfigure(0, weight=1)

        # Create an empty frame with fixed size instead of blank image
        self.blank_tab = ttk.Frame(self.tab_control, width=800, height=600)  # Set your desired size here
        self.tab_control.add(self.blank_tab, text="Welcome")
        
        # Add a welcome label
        welcome_label = ttk.Label(self.blank_tab, 
                                text="Welcome to Peak Analysis Tool\n\nPlease load a file to begin", 
                                font=("Arial", 14))
        welcome_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Prevent the blank tab from shrinking
        self.blank_tab.pack_propagate(False)

        # Functional Bar in Preview Window
        functional_bar = ttk.Frame(preview_frame)
        functional_bar.grid(row=1, column=0, sticky="ew", pady=10)

        ttk.Button(functional_bar, text="Zoom In", command=self.enable_zoom).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(functional_bar, text="Export Plot", command=self.export_plot).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(functional_bar, text="Reset View", command=self.reset_view).grid(row=0, column=2, padx=5, pady=5)

        self.data = None

        # Move the progress bar to the bottom
        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.grid(row=40, column=0, columnspan=2, sticky='ew', padx=5, pady=5)  # Increased row number

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
        print(f"Memory before loading: {get_memory_usage():.2f} MB")
        if self.file_mode.get() == "single":
            files = filedialog.askopenfilenames(
                title="Select Data File",
                filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
            )
        else:  # batch mode
            folder = filedialog.askdirectory(title="Select Folder with Data Files")
            if folder:
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]
            else:
                files = []

        if files:
            try:
                self.preview_label.config(text="Loading files...", foreground="blue")
                self.update_idletasks()
                
                # Initialize empty lists for data
                all_times = []
                all_amplitudes = []
                
                # Try to read first file to determine format
                test_df = pd.read_csv(files[0], delimiter='\t')
                has_timestamp = 'Time - Plot 0' in [col.strip() for col in test_df.columns]
                
                for file in files:
                    df = pd.read_csv(file, delimiter='\t')
                    df.columns = [col.strip() for col in df.columns]
                    
                    if has_timestamp:
                        time_col = 'Time - Plot 0'
                        amp_col = 'Amplitude - Plot 0'
                    else:
                        # Assume first column is time, second is amplitude
                        df.columns = ['Time - Plot 0', 'Amplitude - Plot 0']
                        time_col = 'Time - Plot 0'
                        amp_col = 'Amplitude - Plot 0'
                    
                    # Append data
                    if all_times:
                        # Adjust time values to continue from last time point
                        time_offset = all_times[-1][-1] + (df[time_col].iloc[1] - df[time_col].iloc[0])
                        all_times.append(df[time_col].values + time_offset)
                    else:
                        all_times.append(df[time_col].values)
                    all_amplitudes.append(df[amp_col].values)
                
                # Combine all data
                self.t_value = np.concatenate(all_times)
                self.x_value = np.concatenate(all_amplitudes)
                
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
                
                # Update preview with data info
                preview_text = (
                    f"Successfully loaded {len(files)} files\n"
                    f"Total data points: {len(self.data):,}\n"
                    f"Time range: {self.t_value.min():.2f} to {self.t_value.max():.2f}\n"
                    f"Preview of combined data:\n"
                    f"{self.data.head().to_string(index=False)}"
                )
                
                self.preview_label.config(text="Files loaded successfully", foreground="green")
                self.update_results_summary(preview_text=preview_text)
                
            except Exception as e:
                self.data = None
                self.t_value = None
                self.x_value = None
                self.show_error("Error loading files", e)
        else:
            self.preview_label.config(text="No files selected", foreground="red")
        print(f"Memory after loading: {get_memory_usage():.2f} MB")

    # Function to plot the raw data
    def plot_raw_data(self):
        if self.data is not None:
            try:
                # Initialize canvas if it doesn't exist
                if self.canvas is None:
                    self.canvas = FigureCanvasTkAgg(self.figure, self.tab_control)

                # Clear the current figure
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                # Use optimized plotting
                self.plot_optimized(
                    self.data['Time - Plot 0']*1e-4/60, 
                    self.data['Amplitude - Plot 0'],
                    ax,
                    linestyle='-', 
                    color='black', 
                    linewidth=0.05, 
                    alpha=0.9, 
                    label='Raw Data'
                )
                ax.set_xlabel('Time (min)')
                ax.set_ylabel('Amplitude (counts)')
                ax.set_title('Raw Data: Amplitude vs. Time')
                ax.legend()
                ax.grid(True)
                self.figure.tight_layout()
                
                # Update the existing canvas
                self.canvas.draw_idle()

                # Save the plot as an image
                self.figure.savefig('raw_data_plot.png', bbox_inches='tight')
                self.preview_label.config(text="Raw data plotted successfully", foreground="green")

                # Update or create tab
                tab_exists = False
                for tab in self.tab_control.tabs():
                    if self.tab_control.tab(tab, "text") == "Raw Data Plot":
                        self.tab_control.select(tab)
                        tab_exists = True
                        break
                        
                if not tab_exists:
                    new_tab = ttk.Frame(self.tab_control)
                    self.tab_control.add(new_tab, text="Raw Data Plot")
                    self.tab_control.select(new_tab)
                    canvas = FigureCanvasTkAgg(self.figure, new_tab)
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            except Exception as e:
                self.preview_label.config(text=f"Error plotting raw data: {e}", foreground="red")
        else:
            self.preview_label.config(text="No data to plot", foreground="red")

    # Function to start data analysis
    @profile_function
    def start_analysis(self):
        print(f"Memory before analysis: {get_memory_usage():.2f} MB")
        if self.data is None:
            self.show_error("No data loaded. Please load files first.")
            return
        
        try:
            # Get parameters from GUI
            normalization_factor = self.normalization_factor.get()
            big_counts = self.big_counts.get()
            current_cutoff = self.cutoff_value.get()


            # Use the already loaded data
            t = self.data['Time - Plot 0']*1e-4
            x = self.data['Amplitude - Plot 0']
            # Calculate sampling rate from time data
            rate = np.median(np.diff(t))  # Calculate actual sampling rate
            self.fs = 1 / rate

            # Apply filtering
            if current_cutoff > 0:
                # Use manual cutoff
                order = 2
                btype = 'lowpass'
                self.filtered_signal = apply_butterworth_filter(order, current_cutoff, btype, self.fs, x)
                calculated_cutoff = current_cutoff
            else:
                # Use automatic adjustment
                self.filtered_signal, calculated_cutoff = adjust_lowpass_cutoff(x, self.fs, big_counts, normalization_factor)
                self.cutoff_value.set(calculated_cutoff)
            
            self.t_value = t
            self.x_value = x

            # Calculate suggested threshold
            signal_std = np.std(self.filtered_signal)
            suggested_threshold = 5 * signal_std

            # Update the height_lim entry
            self.height_lim.set(suggested_threshold)

            # Plot results
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot raw data
            ax.plot(t / 60, x, 
                    color='black', 
                    linewidth=0.1, 
                    label='Raw Data',
                    alpha=0.4)
            
            # Plot filtered data
            ax.plot(t / 60, self.filtered_signal, 
                    color='blue', 
                    linewidth=0.1, 
                    label='Filtered Data',
                    alpha=0.9)
            
            # Customize plot
            ax.set_xlabel('Time (min)', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            ax.set_title('Raw and Filtered Signals', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Adjust layout
            self.figure.tight_layout()

            # Update or create tab
            tab_name = "Raw and Filtered Data Plot"
            tab_exists = False
            
            for tab in self.tab_control.tabs():
                if self.tab_control.tab(tab, "text") == tab_name:
                    self.tab_control.select(tab)
                    tab_exists = True
                    break
                    
            if not tab_exists:
                new_tab = ttk.Frame(self.tab_control)
                self.tab_control.add(new_tab, text=tab_name)
                self.tab_control.select(new_tab)
                self.canvas = FigureCanvasTkAgg(self.figure, new_tab)
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Update the canvas
            self.canvas.draw_idle()

            # Save the plot
            self.figure.savefig('filtered_signal_plot.png', bbox_inches='tight')
            
            self.preview_label.config(
                text=f"Analysis completed successfully (Cutoff: {calculated_cutoff:.2f}, Suggested threshold: {suggested_threshold:.1f})", 
                foreground="green"
            )

        except Exception as e:
            self.show_error("Error during analysis", e)
        print(f"Memory after analysis: {get_memory_usage():.2f} MB")

    # Function to run peak detection
    @profile_function
    def run_peak_detection(self):
        if self.filtered_signal is None:
            self.show_error("Filtered signal not available. Please start the analysis first.")
            return
        try:
            height_lim_factor = self.height_lim.get()
            distance = self.distance.get()
            rel_height = self.rel_height.get()
            width_p = [int(float(x) * 10) for x in self.width_p.get().split(',')]

            # Find peaks
            peaks_x_filter, amp_x_filter = find_peaks_with_window(
                self.filtered_signal, 
                width=width_p, 
                prominence=height_lim_factor,
                distance=distance, 
                rel_height=rel_height
            )
            
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

        except Exception as e:
            self.show_error("Error during peak detection", e)

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

            # Update or create tab
            tab_name = "Filtered Peaks Plot"
            for tab in self.tab_control.tabs():
                if self.tab_control.tab(tab, "text") == tab_name:
                    self.tab_control.forget(tab)
                    break
                    
            new_tab = ttk.Frame(self.tab_control)
            self.tab_control.add(new_tab, text=tab_name)
            self.tab_control.select(new_tab)
            
            new_canvas = FigureCanvasTkAgg(new_figure, new_tab)
            new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            new_canvas.draw_idle()

            # Save the plot
            new_figure.savefig('filtered_peaks_plot.png', bbox_inches='tight')
            self.preview_label.config(
                text=f"Filtered peaks plotted successfully. Found {len(peaks_x_filter)} peaks.", 
                foreground="green"
            )

        except Exception as e:
            self.preview_label.config(text=f"Error plotting filtered peaks: {str(e)}", foreground="red")
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
        if self.filtered_signal is None:
            self.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
            return

        try:
            print("Starting plot_data function...")
            # Convert to numpy array if it's a pandas Series
            t = self.t_value.to_numpy() if isinstance(self.t_value, pd.Series) else self.t_value
            height_lim = self.height_lim.get()
            
            # Debug prints
            print(f"Time array type: {type(t)}")
            print(f"Time array length: {len(t)}")
            print(f"Time array first and last values: {t[0]}, {t[-1]}")
            print(f"Filtered signal length: {len(self.filtered_signal)}")
            
            # Convert width values properly
            width_values = [float(x) for x in self.width_p.get().split(',')]
            width_p = [int(float(x) * 10) for x in width_values]
            
            peaks_x_filter, amp_x_filter = find_peaks_with_window(
                self.filtered_signal, 
                width=width_p,
                prominence=height_lim,
                distance=self.distance.get(), 
                rel_height=self.rel_height.get()
            )

            # Calculate peak areas with safe indexing
            window = np.round(amp_x_filter['widths'], 0).astype(int) + 40
            peak_areas = np.zeros(len(peaks_x_filter))
            
            for i in range(len(peaks_x_filter)):
                peak_idx = peaks_x_filter[i]
                win = window[i]
                
                # Ensure window boundaries are within signal length
                start_idx = max(0, peak_idx - win)
                end_idx = min(len(self.filtered_signal), peak_idx + win)
                
                if start_idx >= end_idx:
                    print(f"Warning: Invalid window for peak {i}")
                    continue
                
                yData = self.filtered_signal[start_idx:end_idx]
                background = np.min(yData)
                
                # Ensure peak boundaries are within signal length
                st = max(0, int(amp_x_filter["left_ips"][i]))
                en = min(len(self.filtered_signal), int(amp_x_filter["right_ips"][i]))
                
                if st < en:
                    peak_areas[i] = np.sum(self.filtered_signal[st:en] - background)
                else:
                    print(f"Warning: Invalid peak boundaries for peak {i}")

            # Calculate peak times safely
            peak_times = t[peaks_x_filter]
            intervals = np.diff(peak_times)
            
            # Create time bins for throughput
            interval = 10  # seconds
            max_time = float(t[-1]) if isinstance(t[-1], np.ndarray) else t[-1]
            bins = np.arange(0, max_time, interval)
            throughput, _ = np.histogram(peak_times, bins=bins)
            bin_centers = bins[:-1] + interval/2

            # Create DataFrame with valid data
            df_all = pd.DataFrame({
                "t": peak_times,
                "width": amp_x_filter['widths'],
                "amplitude": amp_x_filter['prominences'],
                "area": peak_areas
            })

            # Plot results
            self.figure.clear()
            ax = self.figure.subplots(nrows=4, ncols=1, sharex=True)

            # Plot with error checking
            ax[0].scatter(df_all['t']/60, df_all['amplitude'], 
                         s=1, alpha=0.5, color='black', label='Peak Heights')
            ax[0].set_ylabel('Amplitude (counts)')
            ax[0].grid(True, alpha=0.3)

            ax[1].scatter(df_all['t']/60, df_all['width']/10, 
                         s=1, alpha=0.5, color='black', label='Peak Widths')
            ax[1].set_ylabel('Width (ms)')
            ax[1].grid(True, alpha=0.3)

            ax[2].scatter(df_all['t']/60, df_all['area'], 
                         s=1, alpha=0.5, color='black', label='Peak Areas')
            ax[2].set_ylabel('Area (counts)')
            ax[2].grid(True, alpha=0.3)

            ax[3].plot(bin_centers/60, throughput, 
                      color='black', alpha=0.5, label=f'Throughput ({interval}s bins)')
            ax[3].set_ylabel(f'Peaks per {interval}s')
            ax[3].set_xlabel('Time (min)')
            ax[3].grid(True, alpha=0.3)

            # Add legends
            for a in ax:
                a.legend(fontsize=8)

            self.figure.tight_layout()

            # Update or create tab
            tab_name = "Data Plot"
            tab_exists = False
            
            for tab in self.tab_control.tabs():
                if self.tab_control.tab(tab, "text") == tab_name:
                    self.tab_control.select(tab)
                    tab_exists = True
                    break
                    
            if not tab_exists:
                new_tab = ttk.Frame(self.tab_control)
                self.tab_control.add(new_tab, text=tab_name)
                self.tab_control.select(new_tab)
                self.canvas = FigureCanvasTkAgg(self.figure, new_tab)
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Update the canvas
            self.canvas.draw_idle()

            # Save the plot
            self.figure.savefig('time_analysis.png', bbox_inches='tight')
            self.preview_label.config(
                text=f"Data plotted successfully. Found {len(peaks_x_filter)} peaks.", 
                foreground="green"
            )

        except Exception as e:
            print(f"\nError in plot_data:")
            import traceback
            traceback.print_exc()
            self.preview_label.config(text=f"Error plotting data: {str(e)}", foreground="red")

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

            # Clear existing figure
            self.figure.clear()
            ax = self.figure.subplots(nrows=2, ncols=2)

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

            self.figure.suptitle('Peak Property Correlations', y=0.95)
            self.figure.tight_layout()

            # Update or create tab
            tab_name = "Scatter Plot"
            tab_exists = False
            
            for tab in self.tab_control.tabs():
                if self.tab_control.tab(tab, "text") == tab_name:
                    self.tab_control.select(tab)
                    tab_exists = True
                    break
                    
            if not tab_exists:
                new_tab = ttk.Frame(self.tab_control)
                self.tab_control.add(new_tab, text=tab_name)
                self.tab_control.select(new_tab)
                self.canvas = FigureCanvasTkAgg(self.figure, new_tab)
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Update the canvas
            self.canvas.draw_idle()

            # Save the plot
            self.figure.savefig('scatter_plot.png', bbox_inches='tight')
            self.preview_label.config(text="Scatter plot created successfully", foreground="green")

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

    # Function to enable zoom functionality on the plot
    def enable_zoom(self):
        try:
            # First check if we have a valid figure and canvas
            if not hasattr(self, 'figure') or not hasattr(self, 'canvas'):
                self.preview_label.config(text="No plot available to zoom. Please create a plot first.", foreground="red")
                return

            # Get the current axes
            if len(self.figure.axes) == 0:
                self.preview_label.config(text="No plot available to zoom. Please create a plot first.", foreground="red")
                return
                
            ax = self.figure.gca()
            
            # Store the original limits for reset
            self.original_xlim = ax.get_xlim()
            self.original_ylim = ax.get_ylim()

            # Create new rectangle selector with transparent background
            self.rect_selector = RectangleSelector(
                ax,
                self.on_select,
                useblit=True,
                button=[1],  # Left mouse button
                minspanx=5,
                minspany=5,
                spancoords='pixels',
                interactive=True,
                props=dict(
                    facecolor='none',        # Transparent background
                    edgecolor='red',         # Red border
                    alpha=1.0,               # Full opacity for border
                    linestyle='--',          # Dashed border
                    linewidth=1              # Border width
                ),
                drag_from_anywhere=True
            )
            
            # Connect the mouse events
            self.canvas.mpl_connect('button_press_event', self.rect_selector)
            self.canvas.mpl_connect('button_release_event', self.rect_selector)
            self.canvas.mpl_connect('motion_notify_event', self.rect_selector)
            
            self.preview_label.config(text="Zoom enabled. Click and drag to select area.", foreground="green")
            
        except Exception as e:
            self.preview_label.config(text=f"Error enabling zoom: {e}", foreground="red")
            print(f"Zoom error details: {e}")  # For debugging

    # Function to handle the zoom functionality
    def on_select(self, eclick, erelease):
        try:
            if eclick.xdata is None or erelease.xdata is None:
                return
                
            ax = self.figure.gca()
            
            # Get the selected region
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])
            
            # Set the new limits
            ax.set_xlim(x1, x2)
            ax.set_ylim(y1, y2)
            
            # Update the canvas
            self.canvas.draw_idle()
            
            self.preview_label.config(text="Zoom applied. Use reset view to return to original view.", foreground="green")
            
        except Exception as e:
            self.preview_label.config(text=f"Error during zoom: {e}", foreground="red")

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
    def reset_view(self):
        try:
            if not hasattr(self, 'figure') or not hasattr(self, 'canvas'):
                self.preview_label.config(text="No plot available to reset.", foreground="red")
                return
                
            ax = self.figure.gca()
            
            if hasattr(self, 'original_xlim') and hasattr(self, 'original_ylim'):
                # Reset to original limits
                ax.set_xlim(self.original_xlim)
                ax.set_ylim(self.original_ylim)
            else:
                # If original limits not stored, use autoscale
                ax.autoscale()
                
            # Update the canvas
            self.canvas.draw_idle()
            
            # Remove the rectangle selector
            if hasattr(self, 'rect_selector'):
                self.rect_selector = None
            
            self.preview_label.config(text="View reset to original", foreground="green")
            
        except Exception as e:
            self.preview_label.config(text=f"Error resetting view: {e}", foreground="red")

    # Add new method for auto cutoff calculation
    def calculate_auto_cutoff_frequency(self):
        if not hasattr(self, 'x_value') or not hasattr(self, 'fs'):
            self.show_error("Please run analysis first")
            return
        
        try:
            # Get raw data and parameters
            raw_signal = self.x_value
            big_counts = self.big_counts.get()  # Get Biggest Peaks threshold
            
            # Find peaks using big_counts as the prominence threshold
            peaks_raw, properties = find_peaks(raw_signal, 
                                              width=[1, 2000],
                                              prominence=big_counts,
                                              distance=1000)
            
            if len(peaks_raw) == 0:
                self.preview_label.config(
                    text=f"No peaks found above {big_counts} counts. Try lowering 'Biggest Peaks' value.", 
                    foreground="red"
                )
                return
            
            # Calculate widths at half prominence
            widths_samples = peak_widths(raw_signal, peaks_raw, rel_height=0.5)[0]
            avg_width_samples = np.mean(widths_samples)
            
            # Convert to time
            avg_width_seconds = avg_width_samples / self.fs
            
            # Calculate cutoff frequency as 1/(avg_width)
            calculated_cutoff = 1 / avg_width_seconds
            
            # Limit the cutoff frequency to reasonable bounds
            calculated_cutoff = max(min(calculated_cutoff, 10000), 50)
            
            # Update the cutoff value in the GUI
            self.cutoff_value.set(calculated_cutoff)
            
            self.preview_label.config(
                text=f"Cutoff: {calculated_cutoff:.1f} Hz (avg peak width: {avg_width_seconds*1000:.1f} ms, peaks found: {len(peaks_raw)})", 
                foreground="green"
            )
            
            print(f"Auto-Cutoff Debug Info:")
            print(f"Biggest Peaks threshold: {big_counts}")
            print(f"Number of peaks found: {len(peaks_raw)}")
            print(f"Peak widths (samples): {widths_samples}")
            print(f"Average width (samples): {avg_width_samples}")
            print(f"Average width (ms): {avg_width_seconds*1000}")
            print(f"Calculated cutoff: {calculated_cutoff}")
            
        except Exception as e:
            self.show_error("Error calculating cutoff frequency", e)

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

    def on_file_mode_change(self, *args):
        if self.file_mode.get() == "batch":
            self.timestamps_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
            self.timestamps_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
            self.browse_button.config(text="Select Folder")
        else:
            self.timestamps_label.grid_remove()
            self.timestamps_entry.grid_remove()
            self.browse_button.config(text="Load File")

    # Add this helper function at the class level
    def show_error(self, message, error=None):
        """Display error in both GUI and terminal with additional debug info."""
        error_message = f"{message}: {str(error)}" if error else message
        
        # Print detailed error information to terminal
        print("\n=== ERROR DETAILS ===")
        print(f"Error Message: {error_message}")
        if error:
            print(f"Error Type: {type(error).__name__}")
            print("Traceback:")
            import traceback
            traceback.print_exc()
        print("===================\n")
        
        # Show simplified message in GUI
        self.preview_label.config(text=error_message, foreground="red")

if __name__ == "__main__":
    app = Application()
    app.mainloop()