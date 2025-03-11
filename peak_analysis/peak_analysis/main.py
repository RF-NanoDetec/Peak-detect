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
from peak_analysis_utils import * # Importiere alle Funktionen aus dem neuen Modul
from plot_functions import plot_raw_data as plot_raw_data_function


# Set default seaborn style
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.2)
sns.set_context("notebook", rc={"lines.linewidth": 1.0})


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.setup_performance_logging()
        self.title("Peak Analysis Tool")
        self.geometry("1920x1080")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # Initialize high-resolution figure
        self.figure = Figure(
            figsize=Config.Plot.FIGURE_SIZE, 
            dpi=Config.Plot.DPI,
            facecolor='white'
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

        self.create_widgets()
        self.blank_tab_exists = True  # Track if the blank tab exists

        self.setup_performance_logging()

        # Initialize data plot attributes
        self.data_figure = None
        self.data_canvas = None
        self.data_original_xlims = None
        self.data_original_ylims = None

        self.tab_figures = {}  # Dictionary to store figures for each tab

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
        self.create_tooltip(
            action_frame.winfo_children()[-2],  # Next Peaks button
            "Show next set of individual peaks"
        )

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
            "Show next set of individual peaks"
        )
        self.create_tooltip(
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
                df = pd.read_excel(file)
            else:
                df = pd.read_csv(file, delimiter='\t')
            
            # Strip any extra spaces in the column names
            df.columns = [col.strip() for col in df.columns]
            
            # Check if the file has the standard time column name
            if 'Time - Plot 0' in df.columns:
                time_col = 'Time - Plot 0'
                amp_col = 'Amplitude - Plot 0'
            else:
                # Assume first column is time, second is amplitude if headers don't match expected names
                df.columns = ['Time - Plot 0', 'Amplitude - Plot 0']
                time_col = 'Time - Plot 0'
                amp_col = 'Amplitude - Plot 0'
            
            return {
                'time': df[time_col].values,
                'amplitude': df[amp_col].values,
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
            self.show_error("Error resetting application state", e)

    @profile_function
    def browse_file(self):
        """Browse and load file(s) based on current mode"""
        print(f"Memory before loading: {get_memory_usage():.2f} MB")
        print(f"Current file mode: {self.file_mode.get()}")
        
        try:
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
                        except Exception as e:
                            print(f"Error loading file {i}: {str(e)}")
                            raise e

                # Sort results by index to maintain order
                results.sort(key=lambda x: x['index'])
                
                # Process results
                all_times = []
                all_amplitudes = []
                
                for i, result in enumerate(results):
                    if self.file_mode.get() == "batch" and timestamps:
                        if i == 0:
                            start_time = timestamps_to_seconds([timestamps[0]], timestamps[0])[0]*1e4
                        current_time = timestamps_to_seconds([timestamps[i]], timestamps[0])[0]*1e4
                        time_offset = current_time
                        all_times.append(result['time'] + time_offset)
                    else:
                        if all_times:
                            time_offset = all_times[-1][-1] + (result['time'][1] - result['time'][0])
                            all_times.append(result['time'] + time_offset)
                        else:
                            all_times.append(result['time'])
                    
                    all_amplitudes.append(result['amplitude'])
                
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
    @profile_function
    def plot_raw_data(self):
        """Ruft die ausgelagerte Funktion für optimierte Darstellung der Rohdaten auf"""
        return plot_raw_data_function(self, profile_function=profile_function)

    # Function to start data analysis
    @profile_function
    def start_analysis(self):
        """Optimized analysis and plotting of filtered data"""
        from plot_functions import start_analysis as start_analysis_function
        return start_analysis_function(self, profile_function=profile_function)


    # Function to run peak detection
    @profile_function
    def run_peak_detection(self):
        """Run peak detection and overlay peaks on existing plot"""
        from plot_functions import run_peak_detection as run_peak_detection_function
        return run_peak_detection_function(self, profile_function=profile_function)

    
    # Function to plot the detected filtered peaks
    @profile_function
    def plot_filtered_peaks(self):
        from plot_functions import plot_filtered_peaks as plot_filtered_peaks_function
        return plot_filtered_peaks_function(self, profile_function=profile_function)

    def show_next_peaks(self):
        """Show the next set of peaks in the filtered peaks plot"""
        from plot_functions import show_next_peaks as show_next_peaks_function
        return show_next_peaks_function(self)

    # Function to calculate the areas of detected peaks
    @profile_function
    @njit  # Apply Numba's Just-In-Time compilation
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
                start_idx = max(0, peaks_x_filter[i] - window[i]) 
                end_idx = min(len(filtered_signal), peaks_x_filter[i] + window[i])
                yData = filtered_signal[start_idx:end_idx]
                background = np.min(yData)

                st = int(amp_x_filter["left_ips"][i])
                en = int(amp_x_filter["right_ips"][i])

                start[i] = st
                end[i] = en

                peak_area[i] = np.sum(filtered_signal[st:en] - background)
        
            # Create DataFrame with results
            results_df = pd.DataFrame({
                "Peak Time (min)": self.t_value[peaks_x_filter] / 60,  # Convert to minutes
                "Peak Area": peak_area,
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
        from plot_functions import plot_data as plot_data_function
        return plot_data_function(self, profile_function=profile_function)

    # Function to export the current plot
    def export_plot(self):
        """Export plot with high resolution"""
        try:
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
                    print("Warning: No figure found for this tab")
                    return
                
                # Save with high DPI and tight layout
                figure_to_export.savefig(
                    file_path,
                    dpi=Config.Plot.EXPORT_DPI,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    edgecolor='none'
                )
                self.preview_label.config(
                    text=f"Plot exported successfully to {file_path}", 
                    foreground="green"
                )
                print(f"Successfully exported figure to {file_path}")  # Debug output
                
        except Exception as e:
            error_msg = f"Error exporting plot: {str(e)}"
            print(f"Export error: {error_msg}")  # Debug output
            self.preview_label.config(
                text=error_msg, 
                foreground="red"
            )

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
        Centralized error handling
        
        Args:
            title (str): Error title
            error (Exception): The error object
        """
        error_message = f"{title}: {str(error)}"
        print(error_message)
        self.preview_label.config(text=error_message, foreground=Config.Colors.ERROR)
        logging.error(f"{error_message}\n{traceback.format_exc()}")

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
        """Enhanced scatter plot for peak property correlations"""
        from plot_functions import plot_scatter as plot_scatter_function
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
        
        # Calculate decimation factor
        stride = len(x) // max_points
        
        # Initialize masks for important points
        mask = np.zeros(len(x), dtype=bool)
        
        # Include regularly spaced points
        mask[::stride] = True
        
        # Find peaks and include points around them
        peaks, _ = find_peaks(y, height=np.mean(y) + 3*np.std(y))
        for peak in peaks:
            start_idx = max(0, peak - 5)
            end_idx = min(len(x), peak + 6)
            mask[start_idx:end_idx] = True
        
        # Find significant changes in signal
        diff = np.abs(np.diff(y))
        significant_changes = np.where(diff > 5*np.std(diff))[0]
        for idx in significant_changes:
            mask[idx:idx+2] = True
        
        # Apply mask
        return x[mask], y[mask]
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
    img_path = resource_path("images/startim.png")  # Adjust the path based on your structure
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