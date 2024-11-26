"""
Main application window for the Peak Analysis Tool.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import logging
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from ..config import Config
from ..data_processing import DataProcessor
from ..signal_processing import adjust_lowpass_cutoff
from ..peak_detection import find_peaks_with_window, calculate_peak_areas
from . import (
    PlotManager,
    StatusBar,
    create_tooltip,
    show_error,
    show_info
)

logger = logging.getLogger(__name__)

class Application(tk.Tk):
    """Main application window for the Peak Analysis Tool."""
    
    def __init__(self):
        super().__init__()
        self.setup_window()
        self.setup_variables()
        self.setup_gui()
        self.setup_bindings()
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.plot_manager = PlotManager(self)
        
        logger.info("Application initialized")

    def setup_window(self):
        """Configure main window properties."""
        self.title("Peak Analysis Tool")
        self.geometry(f"{Config.DEFAULT_WINDOW_SIZE[0]}x{Config.DEFAULT_WINDOW_SIZE[1]}")
        
        # Configure style
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        
        # Configure grid
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

    def setup_variables(self):
        """Initialize Tkinter variables."""
        # File handling variables
        self.file_path = tk.StringVar()
        self.file_mode = tk.StringVar(value="single")
        self.batch_timestamps = tk.StringVar()
        
        # Analysis parameters
        self.normalization_factor = tk.DoubleVar(value=Config.Analysis.DEFAULT_NORMALIZATION)
        self.big_counts = tk.IntVar(value=Config.Analysis.DEFAULT_BIG_COUNTS)
        self.height_lim = tk.DoubleVar(value=Config.Analysis.DEFAULT_HEIGHT_LIM)
        self.distance = tk.IntVar(value=Config.Analysis.DEFAULT_DISTANCE)
        self.rel_height = tk.DoubleVar(value=Config.Analysis.DEFAULT_REL_HEIGHT)
        self.width_p = tk.StringVar(value=Config.Analysis.DEFAULT_WIDTH_RANGE)
        self.cutoff_value = tk.DoubleVar()
        
        # Protocol variables
        self.protocol_vars = {
            'start_time': tk.StringVar(),
            'particle': tk.StringVar(),
            'concentration': tk.StringVar(),
            'stamp': tk.StringVar(),
            'laser_power': tk.StringVar(),
            'setup': tk.StringVar(),
            'notes': tk.StringVar()
        }
        
        # Analysis results
        self.filtered_signal = None
        self.peaks = None
        self.peak_properties = None

    def setup_gui(self):
        """Create and configure GUI elements."""
        # Create main frames
        self.control_frame = ttk.Frame(self)
        self.control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        # Create notebook for controls
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_data_loading_tab()
        self.create_preprocessing_tab()
        self.create_peak_detection_tab()
        self.create_analysis_tab()
        
        # Create status bar
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        # Initialize plot area
        self.setup_plot_area()

    def create_data_loading_tab(self):
        """Create the data loading tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Data Loading")
        
        # File mode selection
        mode_frame = ttk.LabelFrame(tab, text="File Mode")
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="Single File",
            variable=self.file_mode,
            value="single",
            command=self.on_file_mode_change
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            mode_frame,
            text="Batch Mode",
            variable=self.file_mode,
            value="batch",
            command=self.on_file_mode_change
        ).pack(side=tk.LEFT, padx=5)
        
        # File selection
        file_frame = ttk.LabelFrame(tab, text="File Selection")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.browse_button = ttk.Button(
            file_frame,
            text="Load File",
            command=self.browse_file
        )
        self.browse_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Protocol information
        self.create_protocol_frame(tab)

    def create_preprocessing_tab(self):
        """Create the preprocessing tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Preprocessing")
        
        # Filtering frame
        filter_frame = ttk.LabelFrame(tab, text="Signal Filtering")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Cutoff frequency
        cutoff_frame = ttk.Frame(filter_frame)
        cutoff_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(cutoff_frame, text="Cutoff Frequency (Hz)").pack(side=tk.LEFT)
        ttk.Entry(
            cutoff_frame,
            textvariable=self.cutoff_value,
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            cutoff_frame,
            text="Auto Calculate",
            command=self.calculate_auto_cutoff
        ).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        action_frame = ttk.Frame(tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            action_frame,
            text="View Raw Data",
            command=self.plot_raw_data
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            action_frame,
            text="Apply Filtering",
            command=self.start_analysis
        ).pack(side=tk.LEFT, padx=5)

    def create_peak_detection_tab(self):
        """Create the peak detection tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Peak Detection")
        
        # Peak parameters
        params_frame = ttk.LabelFrame(tab, text="Peak Parameters")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create parameter entries
        self.create_parameter_entry(
            params_frame, "Height Threshold", self.height_lim
        )
        self.create_parameter_entry(
            params_frame, "Min Distance", self.distance
        )
        self.create_parameter_entry(
            params_frame, "Relative Height", self.rel_height
        )
        self.create_parameter_entry(
            params_frame, "Width Range", self.width_p
        )
        
        # Action buttons
        action_frame = ttk.Frame(tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            action_frame,
            text="Detect Peaks",
            command=self.run_peak_detection
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            action_frame,
            text="View Individual Peaks",
            command=self.plot_individual_peaks
        ).pack(side=tk.LEFT, padx=5)

    def create_analysis_tab(self):
        """Create the analysis tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Analysis")
        
        # Analysis options
        options_frame = ttk.LabelFrame(tab, text="Analysis Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            options_frame,
            text="Peak Statistics",
            command=self.show_peak_statistics
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        ttk.Button(
            options_frame,
            text="Export Results",
            command=self.export_results
        ).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Results display
        results_frame = ttk.LabelFrame(tab, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(
            results_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_plot_area(self):
        """Initialize the plotting area."""
        self.figure = Figure(figsize=Config.Plot.FIGURE_SIZE, dpi=Config.Plot.DPI)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_bindings(self):
        """Setup keyboard and event bindings."""
        self.bind('<Control-o>', lambda e: self.browse_file())
        self.bind('<Control-s>', lambda e: self.export_results())
        self.bind('<Escape>', lambda e: self.quit())
        
        # Window close protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.quit()

    # Add your existing method implementations here
    def browse_file(self):
        """File browsing implementation."""
        pass  # Implement your existing browse_file logic

    def start_analysis(self):
        """Analysis implementation."""
        pass  # Implement your existing analysis logic

    def run_peak_detection(self):
        """Peak detection implementation."""
        pass  # Implement your existing peak detection logic

    # ... Add all other necessary methods ...

if __name__ == "__main__":
    app = Application()
    app.mainloop()
