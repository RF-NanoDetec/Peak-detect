import tkinter as tk
import numpy as np

class AppState:
    """
    Manages the state of the Peak Analysis application.

    Holds all parameters, loaded data, analysis results, and UI state flags.
    Uses Tkinter variables for parameters that are directly linked to UI widgets.
    """
    def __init__(self):
        # --- Data Loading & File Info ---
        self.file_path = tk.StringVar()  # Path to the currently loaded file/folder
        self.loaded_files = []  # List of paths for batch mode or multi-file single mode
        self.file_mode = tk.StringVar(value="single")  # "single" or "batch"
        self.batch_timestamps = tk.StringVar() # Timestamps for batch mode (e.g., "0:00,1:30")
        self.time_resolution = tk.DoubleVar(value=1e-4) # Dwell time in seconds (default: 0.1 ms)
        self.time_resolution_ms = tk.StringVar(value="0.1") # Mirrored value in ms for UI entry

        # --- Raw & Processed Data ---
        self.raw_data = None # Holds the raw signal data (numpy array)
        self.t_value = None # Time axis (numpy array)
        self.x_value = None # Signal values (numpy array) - can be raw or filtered
        self.filtered_signal = None # Holds the filtered signal data (numpy array)

        # --- Preprocessing Parameters ---
        self.filter_enabled = tk.BooleanVar(value=True) # Low-pass filter toggle
        self.cutoff_value = tk.DoubleVar(value=0) # Low-pass filter cutoff frequency (Hz), 0 for auto

        # --- Peak Detection Parameters ---
        self.height_lim = tk.DoubleVar(value=20) # Peak detection threshold (amplitude)
        self.distance = tk.IntVar(value=5) # Min distance between peaks (samples)
        self.rel_height = tk.DoubleVar(value=0.8) # Relative height for width measurement (0-1)
        self.width_p = tk.StringVar(value="0.1,50") # Peak width range (ms, e.g., "min,max")
        self.sigma_multiplier = tk.DoubleVar(value=5.0) # Sigma multiplier for auto threshold (1-10)
        # self.filter_bandwidth = tk.DoubleVar(value=0) # Not currently used? Re-evaluate if needed.
        self.prominence_ratio = tk.DoubleVar(value=0.8) # Min prominence/height ratio for filtering subpeaks

        # --- Peak Detection Results ---
        self.peaks = None # Indices of detected peaks (numpy array)
        self.peak_properties = None # Dictionary of properties from scipy.find_peaks
        self.peak_heights = None # Peak heights (often prominences)
        self.peak_widths = None # Peak widths
        self.peak_left_ips = None # Left interpolation points for width
        self.peak_right_ips = None # Right interpolation points for width
        self.peak_width_heights = None # Height at which width was measured
        self.peak_areas = None # Calculated peak areas

        # --- Protocol Information ---
        self.protocol_start_time = tk.StringVar()
        self.protocol_id_filter = tk.StringVar() # ND Filter
        self.protocol_buffer = tk.StringVar()
        self.protocol_buffer_concentration = tk.StringVar()
        self.protocol_measurement_date = tk.StringVar()
        self.protocol_sample_number = tk.StringVar()
        self.protocol_particle = tk.StringVar()
        self.protocol_concentration = tk.StringVar()
        self.protocol_stamp = tk.StringVar()
        self.protocol_laser_power = tk.StringVar()
        self.protocol_setup = tk.StringVar()
        self.protocol_notes = tk.StringVar()
        self.protocol_files = tk.StringVar() # Order of files loaded

        # --- Double Peak Analysis ---
        self.double_peak_analysis = tk.StringVar(value="0") # Mode toggle: "0"=normal, "1"=double peak
        self.double_peak_min_distance = tk.DoubleVar(value=0.001) # Min distance (s)
        self.double_peak_max_distance = tk.DoubleVar(value=0.010) # Max distance (s)
        self.double_peak_min_amp_ratio = tk.DoubleVar(value=0.1) # Min amplitude ratio
        self.double_peak_max_amp_ratio = tk.DoubleVar(value=5.0) # Max amplitude ratio
        self.double_peak_min_width_ratio = tk.DoubleVar(value=0.1) # Min width ratio
        self.double_peak_max_width_ratio = tk.DoubleVar(value=5.0) # Max width ratio
        self.double_peaks = None # Results of double peak analysis
        self.current_double_peak_page = 0 # Pagination for double peak grid view

        # --- Time-Resolved Analysis ---
        self.throughput_interval = tk.DoubleVar(value=10.0) # Interval for time-resolved analysis (s)

        # --- UI State Flags ---
        self.log_scale_enabled = tk.BooleanVar(value=True) # Plot scale toggle (log/linear)
        self.show_filtered_peaks = tk.BooleanVar(value=False) # Toggle visibility in analysis plot

    def reset(self):
        """Reset all state variables to their initial default values."""
        self.__init__() # Re-initialize to reset all variables 