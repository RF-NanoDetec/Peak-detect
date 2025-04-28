"""
UI Components for Peak Analysis Tool

This module contains functions to create the main UI components used in the application.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np
from config.settings import Config

def validate_float_entry(value):
    """Validate that entry value is a valid float or empty string"""
    if value == "" or value == "-":
        return True
    try:
        # Allow exponential notation and decimal points
        if 'e' in value.lower() or 'E' in value:
            # Handle scientific notation
            parts = value.lower().split('e')
            if len(parts) != 2:
                return False
            try:
                float(parts[0])
                int(parts[1])
                return True
            except ValueError:
                return False
        else:
            float(value)
        return True
    except ValueError:
        return False

def create_menu_bar(app):
    """Create the application menu bar"""
    menu_bar = tk.Menu(app)
    app.config(menu=menu_bar)
    
    # File Menu
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open File", command=app.browse_file)
    file_menu.add_command(label="Export Results", command=app.save_peak_information_to_csv)
    file_menu.add_separator()
    file_menu.add_command(label="Export Current Plot", command=app.export_plot)
    file_menu.add_command(label="Take Screenshot", command=app.take_screenshot)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=app.quit)
    
    # Edit Menu
    edit_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Edit", menu=edit_menu)
    edit_menu.add_command(label="Reset Application", command=app.reset_application_state)
    
    # View Menu
    view_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Raw Data", command=lambda: app.plot_raw_data())
    view_menu.add_command(label="Filtered Data", command=lambda: app.start_analysis())
    view_menu.add_command(label="Detected Peaks", command=lambda: app.run_peak_detection())
    view_menu.add_separator()
    view_menu.add_command(label="Peak Analysis", command=lambda: app.plot_data())
    view_menu.add_command(label="Peak Correlations", command=lambda: app.plot_scatter())
    view_menu.add_separator()
    # Add theme toggle option
    current_theme = "Light" if app.theme_manager.current_theme == "dark" else "Dark"
    view_menu.add_command(label=f"Switch to {current_theme} Theme", command=app.toggle_theme)
    
    # Tools Menu
    tools_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Tools", menu=tools_menu)
    tools_menu.add_command(label="Auto Calculate Threshold", command=app.calculate_auto_threshold)
    tools_menu.add_command(label="Auto Calculate Cutoff", command=app.calculate_auto_cutoff_frequency)
    tools_menu.add_separator()
    tools_menu.add_command(label="View Individual Peaks", command=lambda: app.plot_filtered_peaks())
    tools_menu.add_command(label="Next Peaks", command=lambda: app.show_next_peaks())
    
    # Help Menu
    help_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Documentation", command=app.show_documentation)
    help_menu.add_command(label="About", command=app.show_about_dialog)
    
    return menu_bar

def create_control_panel(app, main_frame):
    """Create the control panel with tabs"""
    control_frame = ttk.Frame(main_frame)
    control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    # Add status indicator at the top
    status_frame = ttk.Frame(control_frame)
    status_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Status indicator
    app.status_indicator = app.status_indicator_class(status_frame, theme_manager=app.theme_manager)
    app.status_indicator.pack(fill=tk.X, padx=5, pady=5)
    
    # Create notebook (tabbed interface) for controls
    app.tab_control = ttk.Notebook(control_frame)
    app.tab_control.pack(fill=tk.BOTH, expand=True)

    # Create tabs
    create_data_loading_tab(app, app.tab_control)
    create_preprocessing_tab(app, app.tab_control)
    create_peak_detection_tab(app, app.tab_control)
    create_peak_analysis_tab(app, app.tab_control)
    
    # Add double peak analysis tab if enabled
    if app.double_peak_analysis.get() == "1":
        create_double_peak_analysis_tab(app, app.tab_control)

    # Progress bar with green color
    app.progress = ttk.Progressbar(
        control_frame, 
        mode='determinate',
        style='Green.Horizontal.TProgressbar'
    )
    app.progress.pack(fill=tk.X, padx=5, pady=5)

    # Preview label for status messages
    app.preview_label = ttk.Label(control_frame, text="", foreground="black")
    app.preview_label.pack(fill=tk.X, padx=5, pady=5)
    
    return control_frame

def create_preview_frame(app, main_frame):
    """Create the preview frame with plot tabs"""
    preview_frame = ttk.Frame(main_frame)
    preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)
    preview_frame.rowconfigure(1, weight=0)

    # Tab Control for Multiple Plots on the right
    app.plot_tab_control = ttk.Notebook(preview_frame)
    app.plot_tab_control.grid(row=0, column=0, sticky="nsew")

    # Create an empty frame with fixed size instead of blank image
    app.blank_tab = ttk.Frame(app.plot_tab_control, width=800, height=600)
    app.plot_tab_control.add(app.blank_tab, text="Welcome")
    
    # Add a welcome label with theme-appropriate styling
    welcome_label = ttk.Label(
        app.blank_tab, 
        text="Welcome to Peak Analysis Tool\n\nPlease load a file to begin", 
        font=("Arial", 14),
        foreground=app.theme_manager.get_color('text'),
        background=app.theme_manager.get_color('background')
    )
    welcome_label.place(relx=0.5, rely=0.5, anchor="center")
    
    # Prevent the blank tab from shrinking
    app.blank_tab.pack_propagate(False)

    # Functional Bar under plot tabs
    functional_bar = ttk.Frame(preview_frame)
    functional_bar.grid(row=1, column=0, sticky="ew", pady=10)

    ttk.Button(functional_bar, 
              text="Export Plot", 
              command=app.export_plot
    ).grid(row=0, column=0, padx=5, pady=5)
    
    # Add scale toggle button
    scale_toggle_btn = ttk.Button(
        functional_bar,
        text="Toggle Scale (Log/Linear)",
        command=app.toggle_scale_mode
    )
    scale_toggle_btn.grid(row=0, column=1, padx=5, pady=5)
    
    # Add tooltip for scale toggle button
    app.add_tooltip(
        scale_toggle_btn,
        "Toggle between logarithmic and linear scales for peak analysis plots"
    )
    
    return preview_frame

def create_data_loading_tab(app, tab_control):
    """Create the data loading tab"""
    data_loading_tab = ttk.Frame(tab_control)
    tab_control.add(data_loading_tab, text="Data Loading")

    # File mode selection frame
    file_mode_frame = ttk.LabelFrame(data_loading_tab, text="File Mode")
    file_mode_frame.pack(fill=tk.X, padx=5, pady=5)

    # Radio buttons for file mode
    ttk.Radiobutton(
        file_mode_frame, 
        text="Standard Mode", 
        variable=app.file_mode, 
        value="single",
        command=app.on_file_mode_change
    ).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Radiobutton(
        file_mode_frame, 
        text="Timestamp Mode", 
        variable=app.file_mode, 
        value="batch",
        command=app.on_file_mode_change
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Add tooltips for file mode selection
    app.add_tooltip(
        file_mode_frame,
        "Standard Mode: Load single or multiple files with automatic time sequencing\nTimestamp Mode: Load multiple files with custom timestamps"
    )

    # Add peak analysis mode frame (normal vs double peak)
    peak_mode_frame = ttk.LabelFrame(data_loading_tab, text="Peak Analysis Mode")
    peak_mode_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Radio buttons for normal vs double peak analysis
    peak_mode_container = ttk.Frame(peak_mode_frame)
    peak_mode_container.pack(padx=5, pady=5)
    
    ttk.Label(peak_mode_container, text="Analysis Type:").pack(side=tk.LEFT, padx=5)
    
    # Radio buttons for normal vs double peak analysis
    ttk.Radiobutton(
        peak_mode_container,
        text="Normal Analysis",
        variable=app.double_peak_analysis,
        value="0",
        command=app.on_double_peak_mode_change
    ).pack(side=tk.LEFT, padx=10)
    
    ttk.Radiobutton(
        peak_mode_container,
        text="Double Peak Analysis",
        variable=app.double_peak_analysis,
        value="1",
        command=app.on_double_peak_mode_change
    ).pack(side=tk.LEFT, padx=10)
    
    # Add tooltip for peak analysis mode
    app.add_tooltip(
        peak_mode_frame,
        "Select 'Double Peak Analysis' to enable additional canvas for analyzing double peaks"
    )

    # File selection frame
    file_frame = ttk.LabelFrame(data_loading_tab, text="File Selection")
    file_frame.pack(fill=tk.X, padx=5, pady=5)

    # Browse button with styled appearance
    app.browse_button = ttk.Button(
        file_frame, 
        text="Load File", 
        command=app.browse_file,
        style="Primary.TButton"  # Apply primary button style
    )
    app.browse_button.pack(side=tk.LEFT, padx=5, pady=5)

    app.file_name_label = ttk.Label(file_frame, text="No file selected")
    app.file_name_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

    # Timestamps entry for batch mode
    app.timestamps_label = ttk.Label(file_frame, text="Timestamps:")
    app.timestamps_entry = ttk.Entry(file_frame, textvariable=app.batch_timestamps)

    # Initially hide timestamps widgets
    app.timestamps_label.pack_forget()
    app.timestamps_entry.pack_forget()
    
    # Time resolution configuration - make it more prominent
    time_res_frame = ttk.LabelFrame(data_loading_tab, text="⚠️ Dwell Time - Critical Setting ⚠️")
    time_res_frame.pack(fill=tk.X, padx=5, pady=10, ipady=5)
    
    # Create a container for the explanation text
    app.explanation_frame = ttk.Frame(time_res_frame)
    app.explanation_frame.pack(fill=tk.X, padx=5, pady=2)
    
    explanation_text = (
        "This value represents the time interval between data points and is crucial for correct peak width calculations.\n"
        "For most measurements, the default value of 0.1 milliseconds is appropriate."
    )
    app.explanation_label = ttk.Label(app.explanation_frame, text=explanation_text, wraplength=380, justify=tk.LEFT)
    app.explanation_label.pack(anchor=tk.W, padx=5, pady=2)
    
    # Create entry container frame
    entry_frame = ttk.Frame(time_res_frame)
    entry_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create colored frame to highlight importance
    app.highlight_frame = ttk.Frame(entry_frame, style="Accent.TFrame")
    app.highlight_frame.pack(fill=tk.X, padx=2, pady=2)
    
    time_res_label = ttk.Label(app.highlight_frame, text="Dwell Time (ms):", font=("TkDefaultFont", 10, "bold"))
    time_res_label.pack(side=tk.LEFT, padx=10, pady=5)
    
    # Create a StringVar for displaying in milliseconds
    app.time_resolution_ms = tk.StringVar()
    
    # Function to update time resolution in seconds when ms value changes
    def update_time_resolution_seconds(*args):
        try:
            # Convert from ms to seconds
            ms_value = float(app.time_resolution_ms.get())
            app.time_resolution.set(ms_value / 1000.0)
            print(f"Time resolution updated: {ms_value} ms = {app.time_resolution.get()} seconds")
        except ValueError:
            # Handle invalid input
            pass
    
    # Function to initialize the ms display from the seconds value
    def init_time_resolution_ms():
        # Check if time_resolution is a Tkinter variable or a float
        if hasattr(app.time_resolution, 'get'):
            seconds_value = app.time_resolution.get()
        else:
            seconds_value = app.time_resolution
        app.time_resolution_ms.set(f"{seconds_value * 1000:.1f}")
    
    # Set up the trace to update the seconds value when ms changes
    app.time_resolution_ms.trace_add("write", update_time_resolution_seconds)
    
    # Initialize the ms display
    init_time_resolution_ms()
    
    app.time_res_entry = ttk.Entry(
        app.highlight_frame, 
        textvariable=app.time_resolution_ms,
        width=10,
        font=("TkDefaultFont", 10, "bold"),
        validate="key", 
        validatecommand=(app.register(lambda P: validate_float_entry(P)), "%P")
    )
    app.time_res_entry.pack(side=tk.LEFT, padx=5, pady=5)
    
    units_label = ttk.Label(app.highlight_frame, text="milliseconds", font=("TkDefaultFont", 10))
    units_label.pack(side=tk.LEFT, padx=5, pady=5)
    
    # Create a "Reset to Default" button
    def reset_to_default():
        # Check if time_resolution is a Tkinter variable or a float
        if hasattr(app.time_resolution, 'set'):
            app.time_resolution.set(1e-4)  # 0.1 ms in seconds
        else:
            # If it's a float, recreate it as a Tkinter variable
            app.time_resolution = tk.DoubleVar(value=1e-4)
            print("Recreated time_resolution as a Tkinter variable")
        init_time_resolution_ms()
    
    reset_button = ttk.Button(
        entry_frame, 
        text="Reset to Default (0.1 ms)",
        command=reset_to_default
    )
    reset_button.pack(side=tk.RIGHT, padx=5, pady=5)
    
    app.add_tooltip(
        time_res_frame,
        "Dwell time is the time interval between consecutive data points.\n\n"
        "This value is critical for all calculations involving time, especially peak widths.\n"
        "Default is 0.1 milliseconds (0.0001 seconds) which is correct for most measurements.\n\n"
        "Examples:\n"
        "- 0.1 ms: 10,000 points = 1 second\n"
        "- 1.0 ms: 1,000 points = 1 second\n"
        "- 0.01 ms: 100,000 points = 1 second\n\n"
        "Using the wrong dwell time will result in incorrect peak width values!"
    )

    # Add tooltips for file selection controls
    app.add_tooltip(
        app.browse_button,
        "Click to select a data file (single mode) or folder (batch mode)"
    )

    app.add_tooltip(
        app.timestamps_entry,
        "Enter timestamps for batch files in format 'MM:SS,MM:SS,...'\nExample: '00:00,01:30,03:00'"
    )

    # Protocol information frame
    protocol_frame = ttk.LabelFrame(data_loading_tab, text="Protocol Information")
    protocol_frame.pack(fill=tk.X, padx=5, pady=5)

    # Protocol information entries
    protocol_entries = [
        ("Measurement Date:", app.protocol_measurement_date),
        ("Start Time:", app.protocol_start_time),
        ("Setup:", app.protocol_setup),
        ("Particle:", app.protocol_particle),
        ("Particle Concentration:", app.protocol_concentration),
        ("Buffer:", app.protocol_buffer),
        ("Buffer Concentration:", app.protocol_buffer_concentration),
        ("ND Filter:", app.protocol_id_filter),
        ("Laser Power:", app.protocol_laser_power),
        ("Stamp:", app.protocol_stamp)
    ]

    # Create protocol entries first
    for row, (label_text, variable) in enumerate(protocol_entries):
        ttk.Label(protocol_frame, text=label_text).grid(row=row, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(protocol_frame, textvariable=variable).grid(row=row, column=1, padx=5, pady=2, sticky="ew")

    # Protocol tooltips
    protocol_tooltips = {
        "Measurement Date": "Enter the date of measurement in YYYY-MM-DD format",
        "Start Time": "Enter the experiment start time during the day in HH:MM:SS format (e.g., '13:30:00')",
        "Setup": "Enter the experimental setup configuration example: 'Prototype, Old Ladom'",
        "Particle": "Enter the type of particle or sample being analyzed",
        "Particle Concentration": "Enter the concentration of the particles or sample",
        "Buffer": "Enter the buffer solution used in the experiment",
        "Buffer Concentration": "Enter the concentration of the buffer solution",
        "ND Filter": "Enter the neutral density (ND) filter value used in the experiment",
        "Laser Power": "Enter the laser power settings used",
        "Stamp": "Enter any lithographic stamp name or identifier example: 'tripple-block'",
        "Notes": "Enter any additional notes or observations about the experiment"
    }

    # Now apply tooltips after creating the widgets
    for row, (label_text, _) in enumerate(protocol_entries):
        label_widget = protocol_frame.grid_slaves(row=row, column=0)[0]
        entry_widget = protocol_frame.grid_slaves(row=row, column=1)[0]
        
        tooltip_text = protocol_tooltips.get(label_text.rstrip(':'), "")
        app.add_tooltip(label_widget, tooltip_text)
        app.add_tooltip(entry_widget, tooltip_text)

    # Notes field
    ttk.Label(protocol_frame, text="Notes:").grid(row=len(protocol_entries), column=0, padx=5, pady=2, sticky="w")
    notes_entry = ttk.Entry(protocol_frame, textvariable=app.protocol_notes)
    notes_entry.grid(row=len(protocol_entries), column=1, padx=5, pady=2, sticky="ew")

    # Add tooltip for notes field
    notes_label = protocol_frame.grid_slaves(row=len(protocol_entries), column=0)[0]
    app.add_tooltip(
        notes_label,
        "Enter any additional notes or observations about the experiment"
    )
    app.add_tooltip(
        notes_entry,
        "Enter any additional notes or observations about the experiment"
    )

    # Configure grid columns
    protocol_frame.columnconfigure(1, weight=1)

def create_preprocessing_tab(app, tab_control):
    """Create the preprocessing tab"""
    preprocessing_tab = ttk.Frame(tab_control)
    tab_control.add(preprocessing_tab, text="Preprocessing")

    # Create a more compact and modern processing mode selector
    mode_frame = ttk.LabelFrame(preprocessing_tab, text="Signal Processing Mode")
    mode_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Add description text
    description_text = (
        "Signal preprocessing helps improve peak detection by reducing noise and enhancing signal quality.\n"
        "You can choose between filtered data (recommended for most signals) or raw data (preserves original characteristics)."
    )
    description_label = ttk.Label(
        mode_frame, 
        text=description_text,
        wraplength=380, 
        justify=tk.LEFT,
        padding=(5, 5)
    )
    description_label.pack(fill=tk.X, padx=5, pady=5)
    
    # Create a more compact toggle switch
    toggle_frame = ttk.Frame(mode_frame)
    toggle_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create horizontal layout for mode selection
    mode_selector = ttk.Frame(toggle_frame)
    mode_selector.pack(pady=5)
    
    # Radio buttons with icons/colors for better visual representation
    filter_radio = ttk.Radiobutton(
        mode_selector,
        text="Filtered Data",
        variable=app.filter_enabled,
        value=True,
        command=lambda: update_filter_state(True)
    )
    filter_radio.pack(side=tk.LEFT, padx=20)
    
    # Create a color chip to show filtered data is smoothed
    # Store reference to the filter color indicator
    app.filter_color_indicator = ttk.Label(
        mode_selector,
        text="   ",
        background="#0078D7",  # Initial color (will be updated by theme)
        relief=tk.RAISED,
        borderwidth=2
    )
    app.filter_color_indicator.pack(side=tk.LEFT, padx=(0, 30))
    
    raw_radio = ttk.Radiobutton(
        mode_selector,
        text="Raw Data",
        variable=app.filter_enabled,
        value=False,
        command=lambda: update_filter_state(False)
    )
    raw_radio.pack(side=tk.LEFT, padx=20)
    
    # Create a color chip to show raw data is noisy
    # Store reference to the raw color indicator
    app.raw_color_indicator = ttk.Label(
        mode_selector,
        text="   ",
        background="#333333",  # Initial color (will be updated by theme)
        relief=tk.RAISED,
        borderwidth=2
    )
    app.raw_color_indicator.pack(side=tk.LEFT)
    
    # Add visual comparison of filtered vs raw data
    comparison_frame = ttk.Frame(mode_frame)
    comparison_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Create canvas for comparison visualization
    # Store reference to the comparison canvas
    canvas_height = 80
    canvas_width = 380
    app.preprocessing_comparison_canvas = tk.Canvas(
        comparison_frame, 
        height=canvas_height, 
        width=canvas_width,
        bg=app.theme_manager.get_color('card_bg'),
        highlightthickness=0
    )
    app.preprocessing_comparison_canvas.pack(pady=5)
    
    # Draw a comparison of raw vs filtered data
    baseline_y = canvas_height // 2
    
    # Draw axis
    app.preprocessing_comparison_canvas.create_line(
        10, baseline_y, canvas_width-10, baseline_y,
        fill="#aaaaaa", dash=(4, 4), width=1
    )
    
    # Draw raw data (noisy)
    raw_points = []
    np.random.seed(42)  # For consistent random noise
    for x in range(10, canvas_width-10, 3):
        # Create a noisy sine wave
        noise = np.random.normal(0, 6) if x % 9 != 0 else np.random.normal(0, 2)
        y = baseline_y - 15 * np.sin((x-10) / 30) + noise
        raw_points.append(x)
        raw_points.append(int(y))
    
    # Create raw data curve
    app.preprocessing_comparison_canvas.create_line(raw_points, fill="#333333", width=1.5, smooth=False)
    
    # Draw filtered data (smooth)
    filtered_points = []
    for x in range(10, canvas_width-10, 3):
        # Create a smooth sine wave
        y = baseline_y - 15 * np.sin((x-10) / 30)
        filtered_points.append(x)
        filtered_points.append(int(y))
    
    app.preprocessing_comparison_canvas.create_line(filtered_points, fill="#0078D7", width=2, smooth=True)
    
    # Function to update UI based on filter state
    def update_filter_state(is_filtered):
        app.filter_enabled.set(is_filtered)
        
        # Update button text
        if is_filtered:
            process_btn.configure(text="Apply Filtering")
        else:
            process_btn.configure(text="Process Raw Data")
            
        # Update filtering section visibility
        if is_filtered:
            filtering_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            filtering_frame.pack_forget()
    
    # Add tooltips for radio buttons
    app.add_tooltip(
        filter_radio, 
        "Apply Butterworth low-pass filter to smooth the signal and reduce noise.\n"
        "Recommended for most signals to improve peak detection."
    )
    app.add_tooltip(
        raw_radio,
        "Use raw unprocessed data without any filtering.\n"
        "Preserves original signal characteristics but may include more noise."
    )
    
    # Create a horizontal separator
    ttk.Separator(preprocessing_tab, orient="horizontal").pack(fill=tk.X, padx=10, pady=10)
    
    # Filtering parameters section in its own frame
    filtering_frame = ttk.LabelFrame(preprocessing_tab, text="Filtering Parameters")
    
    # Add explanation text for filtering
    filter_description = (
        "Signal filtering uses a Butterworth low-pass filter to remove high-frequency noise while\n"
        "preserving important signal features. The cutoff frequency determines which frequencies are removed."
    )
    ttk.Label(
        filtering_frame, 
        text=filter_description,
        wraplength=380, 
        justify=tk.LEFT,
        padding=(5, 5)
    ).pack(fill=tk.X, padx=5, pady=5)

    # Cutoff Frequency section with better layout
    cutoff_frame = ttk.Frame(filtering_frame)
    cutoff_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(
        cutoff_frame, 
        text="Cutoff Frequency (Hz):",
        font=("TkDefaultFont", 10, "bold")
    ).pack(side=tk.LEFT, padx=(5, 10))
    
    cutoff_entry = ttk.Entry(
        cutoff_frame, 
        textvariable=app.cutoff_value, 
        width=8,
        font=("TkDefaultFont", 10)
    )
    cutoff_entry.pack(side=tk.LEFT)
    
    ttk.Label(
        cutoff_frame, 
        text="Hz",
        font=("TkDefaultFont", 10)
    ).pack(side=tk.LEFT, padx=(2, 10))

    auto_cutoff_button = ttk.Button(
        cutoff_frame, 
        text="Auto Calculate",
        style="Accent.TButton", 
        command=app.calculate_auto_cutoff_frequency
    )
    auto_cutoff_button.pack(side=tk.LEFT, padx=5)
    
    # Help text for cutoff frequency with improved explanation of automatic calculation
    cutoff_help = ttk.Label(
        filtering_frame,
        text="Set to 0 for automatic calculation. Auto-calculation finds the highest signal value and uses 70% of this as a threshold to determine appropriate peak frequency.",
        wraplength=380,
        foreground=app.theme_manager.get_color('secondary'),
        font=("TkDefaultFont", 8),
        justify=tk.LEFT
    )
    cutoff_help.pack(fill=tk.X, padx=15, pady=(0, 5))

    # Add auto-calculation explanation frame with more details
    auto_calc_explanation = ttk.LabelFrame(filtering_frame, text="Auto-Calculation Method")
    auto_calc_explanation.pack(fill=tk.X, padx=5, pady=5)
    
    auto_calc_text = (
        "The automatic cutoff frequency is calculated by:\n\n"
        "1. Finding the highest signal value in the data\n"
        "2. Using 70% of this value as a threshold (30% below maximum)\n"
        "3. Detecting peaks above this threshold\n"
        "4. Measuring the average width of these peaks\n"
        "5. Setting the cutoff frequency based on this width\n\n"
        "This approach ensures that the filter preserves actual signal peaks while removing noise."
    )
    
    ttk.Label(
        auto_calc_explanation,
        text=auto_calc_text,
        wraplength=380,
        justify=tk.LEFT,
        padding=(5, 5)
    ).pack(fill=tk.X, padx=5, pady=5)

    # Add enhanced tooltips
    app.add_tooltip(
        cutoff_entry,
        "Cutoff frequency for the low-pass filter in Hertz.\n\n"
        "• Lower values (1-5 Hz): More aggressive filtering, smoother signals\n"
        "• Medium values (10-50 Hz): Balanced filtering for most data\n"
        "• Higher values (>100 Hz): Light filtering, preserves most signal details\n\n"
        "Set to 0 for automatic calculation based on signal characteristics."
    )

    app.add_tooltip(
        auto_cutoff_button,
        "Automatically calculate the optimal cutoff frequency based on signal characteristics.\n"
        "The calculation finds the highest peaks in the signal and determines the ideal cutoff frequency."
    )

    # Show/hide filtering frame based on current mode
    if app.filter_enabled.get():
        filtering_frame.pack(fill=tk.X, padx=5, pady=5)
    else:
        filtering_frame.pack_forget()

    # Action Buttons with improved layout
    action_frame = ttk.LabelFrame(preprocessing_tab, text="Processing Actions")
    action_frame.pack(fill=tk.X, padx=5, pady=(10, 5))

    # Button container for better spacing
    button_container = ttk.Frame(action_frame)
    button_container.pack(fill=tk.X, padx=5, pady=10)

    view_raw_btn = ttk.Button(
        button_container,
        text="View Raw Data",
        command=app.plot_raw_data,
        style="Primary.TButton"
    )
    view_raw_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    process_btn = ttk.Button(
        button_container,
        text="Apply Filtering" if app.filter_enabled.get() else "Process Raw Data",
        command=app.start_analysis,
        style="Accent.TButton"
    )
    process_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # Add tooltips for action buttons
    app.add_tooltip(
        view_raw_btn,
        "Display the original unprocessed data without any filtering"
    )
    
    app.add_tooltip(
        process_btn,
        "Apply the selected processing mode to the data:\n"
        "• Filtered: Applies Butterworth filter with specified cutoff\n"
        "• Raw: Processes data without filtering"
    )

def create_peak_detection_tab(app, tab_control):
    """Create the peak detection tab"""
    peak_detection_tab = ttk.Frame(tab_control)
    tab_control.add(peak_detection_tab, text="Peak Detection")

    # Create a main container with scrollbar
    main_container = ttk.Frame(peak_detection_tab)
    main_container.pack(fill=tk.BOTH, expand=True)
    
    # Create canvas and scrollbar
    # Store reference to the main canvas for the tab
    app.peak_detection_main_canvas = tk.Canvas(main_container, bg=app.theme_manager.get_color('background'))
    scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=app.peak_detection_main_canvas.yview)
    scrollable_frame = ttk.Frame(app.peak_detection_main_canvas) # Add frame to the app's canvas
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: app.peak_detection_main_canvas.configure(scrollregion=app.peak_detection_main_canvas.bbox("all"))
    )
    
    app.peak_detection_main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    app.peak_detection_main_canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack the scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    app.peak_detection_main_canvas.pack(side="left", fill="both", expand=True)

    # Create a dedicated Auto Threshold frame with clear explanation
    auto_threshold_frame = ttk.LabelFrame(scrollable_frame, text="Automatic Threshold Detection")
    auto_threshold_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Add description text
    description_text = (
        "The automatic threshold is calculated using statistical properties of the signal:\n\n"
        "Threshold = σ × Standard Deviation of Signal\n\n"
        "where σ (sigma) controls sensitivity. Higher values make peak detection more selective,\n"
        "requiring larger peaks to exceed the threshold. Lower values detect more peaks including smaller ones."
    )
    description_label = ttk.Label(
        auto_threshold_frame, 
        text=description_text,
        wraplength=380, 
        justify=tk.LEFT,
        padding=(5, 5)
    )
    description_label.pack(fill=tk.X, padx=5, pady=5)
    
    # Add visual diagram to help explain the concept
    diagram_frame = ttk.Frame(auto_threshold_frame)
    diagram_frame.pack(fill=tk.X, padx=10, pady=5)
    
    canvas_height = 80
    canvas_width = 380
    # Store reference to the canvas
    app.threshold_diagram_canvas = tk.Canvas(
        diagram_frame, 
        height=canvas_height, 
        width=canvas_width,
        bg=app.theme_manager.get_color('card_bg'),
        highlightthickness=0
    )
    app.threshold_diagram_canvas.pack()
    
    # Draw a sine-like signal to represent data
    signal_color = "#0078D7"  # Use blue directly instead of theme's primary color
    baseline_y = canvas_height // 2 + 15  # Move baseline down to show peaks better
    
    # Create a single data line with baseline noise and peaks
    data_points = []
    np.random.seed(42)
    for x in range(10, canvas_width-10, 4):
        y = baseline_y
        
        # Add peaks at specific locations
        if 70 <= x <= 90:
            # First peak
            peak_height = 35
            y = baseline_y - peak_height * np.exp(-0.02 * (x - 80) ** 2)
        elif 180 <= x <= 200:
            # Second peak (taller)
            peak_height = 45
            y = baseline_y - peak_height * np.exp(-0.02 * (x - 190) ** 2)
        elif 270 <= x <= 290:
            # Third peak (medium)
            peak_height = 25
            y = baseline_y - peak_height * np.exp(-0.02 * (x - 280) ** 2)
        
        # Add noise to the entire signal
        y += np.random.normal(0, 3)
            
        data_points.append(x)
        data_points.append(int(y))
    
    # Create the single signal curve
    app.threshold_diagram_canvas.create_line(data_points, fill=signal_color, width=2, smooth=True)
    
    # Draw threshold lines for different sigma values
    low_thresh_y = baseline_y - 15  # Low threshold (catches small peaks too)
    med_thresh_y = baseline_y - 25  # Medium threshold (balanced)
    high_thresh_y = baseline_y - 40  # High threshold (only the largest peaks)
    
    # Low sigma (e.g., σ=2) - will detect all peaks including some noise
    app.threshold_diagram_canvas.create_line(
        10, low_thresh_y, canvas_width-10, low_thresh_y,
        fill="#4CAF50", width=1, dash=(2, 2)
    )
    app.threshold_diagram_canvas.create_text(
        canvas_width-15, low_thresh_y-8, 
        text="σ=2", 
        fill="#4CAF50", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # Medium sigma (e.g., σ=5) - balanced threshold
    app.threshold_diagram_canvas.create_line(
        10, med_thresh_y, canvas_width-10, med_thresh_y,
        fill="#FF9800", width=1, dash=(2, 2)
    )
    app.threshold_diagram_canvas.create_text(
        canvas_width-15, med_thresh_y-8, 
        text="σ=5", 
        fill="#FF9800", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # High sigma (e.g., σ=8) - only detects the largest peak
    app.threshold_diagram_canvas.create_line(
        10, high_thresh_y, canvas_width-10, high_thresh_y,
        fill="#F44336", width=1, dash=(2, 2)
    )
    app.threshold_diagram_canvas.create_text(
        canvas_width-15, high_thresh_y-8, 
        text="σ=8", 
        fill="#F44336", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # Add markers to show which peaks are detected at each threshold
    # For low threshold (detects all peaks)
    for x_pos in [80, 190, 280]:
        app.threshold_diagram_canvas.create_oval(
            x_pos-3, low_thresh_y-3, 
            x_pos+3, low_thresh_y+3, 
            fill="#4CAF50", outline=""
        )
        
    # For medium threshold (detects medium and large peaks)
    for x_pos in [190, 280]:
        app.threshold_diagram_canvas.create_oval(
            x_pos-3, med_thresh_y-3, 
            x_pos+3, med_thresh_y+3, 
            fill="#FF9800", outline=""
        )
        
    # For high threshold (detects only the largest peak)
    app.threshold_diagram_canvas.create_oval(
        190-3, high_thresh_y-3, 
        190+3, high_thresh_y+3, 
        fill="#F44336", outline=""
    )
    
    # Add explanatory caption
    caption = ttk.Label(
        diagram_frame,
        text="Lower threshold (green) detects more peaks including noise, higher threshold (red) detects only prominent peaks.",
        wraplength=380,
        justify=tk.CENTER,
        font=("TkDefaultFont", 8)
    )
    caption.pack(pady=(0, 5))
    
    # Sigma multiplier slider in its own container
    sigma_container = ttk.Frame(auto_threshold_frame)
    sigma_container.pack(fill=tk.X, padx=5, pady=5)
    
    # Enhance sigma slider with better layout
    ttk.Label(
        sigma_container, 
        text="Sensitivity (σ):",
        font=("TkDefaultFont", 10, "bold")
    ).pack(side=tk.LEFT, padx=5)
    
    # Current value display with higher visibility
    sigma_value_label = ttk.Label(
        sigma_container, 
        text=f"{app.sigma_multiplier.get():.1f}",
        width=4,
        font=("TkDefaultFont", 10, "bold"),
        foreground=app.theme_manager.get_color('primary')
    )
    sigma_value_label.pack(side=tk.LEFT, padx=5)
    
    # Create a container for the slider to allow better styling
    slider_frame = ttk.Frame(sigma_container)
    slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Store reference to the slider
    app.sigma_slider = tk.Scale(
        slider_frame, 
        from_=1.0, 
        to=10.0, 
        resolution=0.1,
        orient=tk.HORIZONTAL,
        variable=app.sigma_multiplier,
        length=250,
        bg=app.theme_manager.get_color('card_bg'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background'),
        showvalue=False  # Hide the default value display
    )
    app.sigma_slider.pack(fill=tk.X, expand=True)
    
    # Add min/max labels under slider
    slider_labels = ttk.Frame(slider_frame)
    slider_labels.pack(fill=tk.X, expand=True)
    
    ttk.Label(
        slider_labels, 
        text="Lower (1.0)\nMore peaks",
        font=("TkDefaultFont", 8),
        justify=tk.LEFT
    ).pack(side=tk.LEFT)
    
    ttk.Label(
        slider_labels, 
        text="Higher (10.0)\nFewer peaks",
        font=("TkDefaultFont", 8),
        justify=tk.RIGHT
    ).pack(side=tk.RIGHT)
    
    # Update label when slider changes
    def update_sigma_label(*args):
        sigma_value_label.config(text=f"{app.sigma_multiplier.get():.1f}")
    
    app.sigma_multiplier.trace_add("write", update_sigma_label)
    
    # Add buttons container
    buttons_container = ttk.Frame(auto_threshold_frame)
    buttons_container.pack(fill=tk.X, padx=5, pady=5)
    
    # Current threshold display
    threshold_display = ttk.Frame(buttons_container)
    threshold_display.pack(side=tk.LEFT, padx=5)
    
    ttk.Label(
        threshold_display, 
        text="Current Threshold:",
        font=("TkDefaultFont", 9)
    ).pack(side=tk.LEFT)
    
    threshold_entry = ttk.Entry(
        threshold_display, 
        textvariable=app.height_lim, 
        width=8,
        font=("TkDefaultFont", 9, "bold")
    )
    threshold_entry.pack(side=tk.LEFT, padx=5)

    # Calculation button with improved style
    auto_calc_button = ttk.Button(
        buttons_container, 
        text="Calculate Threshold",
        command=app.calculate_auto_threshold,
        style="Accent.TButton"
    )
    auto_calc_button.pack(side=tk.RIGHT, padx=5)
    
    # Add tooltips with detailed explanations
    app.add_tooltip(
        app.sigma_slider,
        "Adjust sensitivity of peak detection:\n"
        "• Lower values (1-3): More sensitive, detects smaller peaks\n"
        "• Medium values (4-6): Balanced detection for most data\n"
        "• Higher values (7-10): Less sensitive, only detects prominent peaks"
    )
    
    app.add_tooltip(
        auto_calc_button,
        "Calculate threshold based on the current sigma value and signal statistics.\n"
        "The formula used is: Threshold = σ × Standard Deviation of Signal"
    )
    
    app.add_tooltip(
        threshold_entry,
        "Current threshold value for peak detection.\n"
        "You can manually edit this value or use auto-calculation."
    )
    
    # Horizontal separator to visually separate auto threshold from other parameters
    ttk.Separator(scrollable_frame, orient="horizontal").pack(fill=tk.X, padx=10, pady=10)
    
    # Create a collapsible frame for manual peak detection parameters
    manual_params_frame = ttk.LabelFrame(scrollable_frame, text="Manual Peak Detection Parameters")
    manual_params_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Add a toggle button for the manual parameters section
    toggle_button = ttk.Button(
        manual_params_frame,
        text="▼ Show Manual Parameters",
        command=lambda: toggle_section(manual_params_container, toggle_button)
    )
    toggle_button.pack(fill=tk.X, padx=5, pady=5)
    
    # Container for manual parameters (initially hidden)
    manual_params_container = ttk.Frame(manual_params_frame)
    
    # Function to toggle section visibility
    def toggle_section(container, button):
        if container.winfo_viewable():
            container.pack_forget()
            button.config(text="▼ Show Manual Parameters")
        else:
            container.pack(fill=tk.X, padx=5, pady=5)
            button.config(text="▲ Hide Manual Parameters")
    
    # Create a single comprehensive visualization frame
    visualization_frame = ttk.LabelFrame(manual_params_container, text="Peak Detection Parameters Visualization")
    visualization_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Add visual diagram with realistic signal
    # Store reference to the manual diagram canvas
    app.manual_diagram_canvas = tk.Canvas(
        visualization_frame,
        height=120,
        width=380,
        bg=app.theme_manager.get_color('card_bg'),
        highlightthickness=0
    )
    app.manual_diagram_canvas.pack(fill=tk.X, padx=5, pady=5)
    
    # Draw a realistic signal with two clear peaks
    signal_color = "#0078D7"  # Use blue directly instead of theme's primary color
    baseline_y = 60  # Center the signal vertically
    
    # Create a more realistic signal with two clear peaks
    data_points = []
    np.random.seed(42)  # For consistent random noise
    
    # Define two clear peaks with different properties
    peaks = [
        {'x': 100, 'height': 50, 'width': 30},  # First peak - larger and wider
        {'x': 250, 'height': 35, 'width': 25}   # Second peak
    ]
    
    # Generate the signal
    for x in range(10, canvas_width-10, 2):
        y = baseline_y
        
        # Add peaks (now positive)
        for peak in peaks:
            # Gaussian peak shape
            if abs(x - peak['x']) < peak['width'] * 2:
                y -= peak['height'] * np.exp(-0.5 * ((x - peak['x']) / (peak['width']/2))**2)
        
        # Add minimal noise
        y += np.random.normal(0, 0.5)
        
        data_points.append(x)
        data_points.append(int(y))
    
    # Create the signal curve
    app.manual_diagram_canvas.create_line(data_points, fill=signal_color, width=2, smooth=True)
    
    # Add parameter indicators with better spacing and colors
    # 1. Minimum Distance between Peaks (between peak centers)
    distance_y = baseline_y + 15  # Move distance indicator up
    app.manual_diagram_canvas.create_line(
        peaks[0]['x'], distance_y,
        peaks[1]['x'], distance_y,
        fill="#FF6B6B", width=1, dash=(2, 2)
    )
    app.manual_diagram_canvas.create_text(
        (peaks[0]['x'] + peaks[1]['x'])/2, distance_y + 10,
        text="Distance between peaks",
        fill="black",
        font=("TkDefaultFont", 8)
    )
    
    # 2. Relative Height (measured from peak top)
    rel_height_y = baseline_y - peaks[0]['height'] * 0.2  # 20% from baseline (80% from top)
    # Draw line from peak top to width measurement height
    app.manual_diagram_canvas.create_line(
        peaks[0]['x'], baseline_y - peaks[0]['height'],  # Start from peak top
        peaks[0]['x'], rel_height_y,  # End at width measurement height
        fill="#4ECDC4", width=1, dash=(2, 2)
    )
    app.manual_diagram_canvas.create_text(
        peaks[0]['x'], rel_height_y - 10,
        text="Relative Height (0.8 = 80% from top)",
        fill="black",
        font=("TkDefaultFont", 8)
    )
    
    # 3. Width Range (measured at relative height) - only for first peak
    width_y = baseline_y + 5
    # Horizontal line at width measurement height
    app.manual_diagram_canvas.create_line(
        peaks[0]['x'] - peaks[0]['width'], width_y,
        peaks[0]['x'] + peaks[0]['width'], width_y,
        fill="#45B7D1", width=1, dash=(2, 2)
    )
    # Vertical lines to show width measurement
    app.manual_diagram_canvas.create_line(
        peaks[0]['x'] - peaks[0]['width'], width_y,
        peaks[0]['x'] - peaks[0]['width'], rel_height_y,
        fill="#45B7D1", width=1, dash=(2, 2)
    )
    app.manual_diagram_canvas.create_line(
        peaks[0]['x'] + peaks[0]['width'], width_y,
        peaks[0]['x'] + peaks[0]['width'], rel_height_y,
        fill="#45B7D1", width=1, dash=(2, 2)
    )
    
    # Add width range text for first peak
    app.manual_diagram_canvas.create_text(
        peaks[0]['x'], width_y + 10,
        text="Width Range",
        fill="black",
        font=("TkDefaultFont", 8)
    )
    
    # Add explanatory caption with clearer description
    caption = ttk.Label(
        visualization_frame,
        text="Peak Detection Parameters:\n"
             "• Distance: Minimum points between peak centers (prevents detecting multiple peaks too close together)\n"
             "• Height: Relative height from peak top (0.8 = measure width at 80% from peak top)\n"
             "• Width: Allowed peak width range in milliseconds (e.g., '0.1,50' means only peaks between 0.1 and 50ms are kept)",
        wraplength=380,
        justify=tk.LEFT,
        font=("TkDefaultFont", 8)
    )
    caption.pack(pady=(0, 5))
    
    # Parameters Frame - now directly below visualization
    params_frame = ttk.Frame(manual_params_container)
    params_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # 1. Minimum Distance between Peaks
    distance_container = ttk.Frame(params_frame)
    distance_container.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(distance_container, text="Distance:").pack(side=tk.LEFT, padx=5)
    # Store reference to distance slider
    app.distance_slider = tk.Scale(
        distance_container,
        from_=1,
        to=100,
        orient=tk.HORIZONTAL,
        variable=app.distance,
        length=250,
        bg=app.theme_manager.get_color('card_bg'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background')
    )
    app.distance_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    distance_entry = ttk.Entry(distance_container, textvariable=app.distance, width=6)
    distance_entry.pack(side=tk.LEFT, padx=5)
    
    # 2. Relative Height
    rel_height_container = ttk.Frame(params_frame)
    rel_height_container.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(rel_height_container, text="Height:").pack(side=tk.LEFT, padx=5)
    # Store reference to height slider
    app.rel_height_slider = tk.Scale(
        rel_height_container,
        from_=0.1,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        variable=app.rel_height,
        length=250,
        bg=app.theme_manager.get_color('card_bg'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background')
    )
    app.rel_height_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    rel_height_entry = ttk.Entry(rel_height_container, textvariable=app.rel_height, width=6)
    rel_height_entry.pack(side=tk.LEFT, padx=5)
    
    # 3. Width Range
    width_container = ttk.Frame(params_frame)
    width_container.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(width_container, text="Width Range (ms):").pack(side=tk.LEFT, padx=5)
    width_entry = ttk.Entry(width_container, textvariable=app.width_p, width=15)
    width_entry.pack(side=tk.LEFT, padx=5)
    
    # Add tooltips for better user guidance
    app.add_tooltip(
        app.distance_slider,
        "Minimum number of points between peak centers.\n"
        "Higher values prevent detecting multiple peaks too close together."
    )
    app.add_tooltip(
        app.rel_height_slider,
        "Relative height (0-1) at which peak width is measured.\n"
        "Example: 0.5 = width at half maximum height, 0.9 = width near peak top"
    )
    app.add_tooltip(
        width_entry,
        "Enter exact peak width range in milliseconds (min,max).\n"
        "Example: '0.1,50' means only peaks between 0.1 and 50ms are kept"
    )
    
    # Action Buttons Frame
    peak_detection_frame = ttk.LabelFrame(scrollable_frame, text="Peak Detection Actions")
    peak_detection_frame.pack(fill=tk.X, padx=5, pady=10)

    # Create a more visually appealing button layout
    buttons_frame = ttk.Frame(peak_detection_frame)
    buttons_frame.pack(fill=tk.X, padx=5, pady=10)

    detect_btn = ttk.Button(
        buttons_frame, 
        text="Detect Peaks",
        command=app.run_peak_detection,
        style="Primary.TButton"
    )
    detect_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    view_btn = ttk.Button(
        buttons_frame, 
        text="View Peaks",
        command=app.plot_filtered_peaks
    )
    view_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    next_btn = ttk.Button(
        buttons_frame,
        text="Next Peaks",
        command=app.show_next_peaks
    )
    next_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    save_btn = ttk.Button(
        buttons_frame, 
        text="Save Results",
        command=app.save_peak_information_to_csv
    )
    save_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    # Add tooltips for action buttons
    app.add_tooltip(
        detect_btn,
        "Run peak detection algorithm with current parameters.\n"
        "This will identify peaks based on threshold and other settings."
    )
    app.add_tooltip(
        view_btn,
        "Display detailed view of selected individual peaks.\n"
        "This helps validate your peak detection settings."
    )
    app.add_tooltip(
        next_btn,
        "Navigate to the next set of peaks in the visualization."
    )
    app.add_tooltip(
        save_btn,
        "Save current peak detection results to CSV file for further analysis."
    )

def create_peak_analysis_tab(app, tab_control):
    """Create the peak analysis tab"""
    peak_analysis_tab = ttk.Frame(tab_control)
    tab_control.add(peak_analysis_tab, text="Peak Analysis")

    # Analysis Options Frame
    analysis_options_frame = ttk.LabelFrame(peak_analysis_tab, text="Analysis Options")
    analysis_options_frame.pack(fill=tk.X, padx=5, pady=5)

    # Time-resolved analysis button (first)
    ttk.Button(
        analysis_options_frame,
        text="Time-Resolved Analysis",  # Changed from "Plot Peak Analysis"
        command=app.plot_data
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Peak properties correlation button (second)
    ttk.Button(
        analysis_options_frame,
        text="Peak Property Correlations",  # Changed from "Plot Peak Properties"
        command=app.plot_scatter
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Results Frame
    results_frame = ttk.LabelFrame(peak_analysis_tab, text="Results Summary")
    results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Add tooltips for better user guidance
    app.add_tooltip(
        analysis_options_frame.winfo_children()[0],  # Time-Resolved Analysis button
        "Display peak properties changes over time and throughput analysis"
    )
    app.add_tooltip(
        analysis_options_frame.winfo_children()[1],  # Peak Property Correlations button
        "Display correlation plots between peak width, height, and area"
    )

def create_double_peak_analysis_tab(app, tab_control):
    """Create the double peak analysis tab"""
    double_peak_tab = ttk.Frame(tab_control)
    tab_control.add(double_peak_tab, text="Double Peak Analysis")
    
    # Parameter frame for double peak detection
    param_frame = ttk.LabelFrame(double_peak_tab, text="Double Peak Detection Parameters")
    param_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Distance parameters with sliders and entry fields
    distance_frame = ttk.LabelFrame(param_frame, text="Peak Distance Range (ms)")
    distance_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create variables to store the slider values in milliseconds
    min_distance_ms = tk.DoubleVar(value=app.double_peak_min_distance.get() * 1000)
    max_distance_ms = tk.DoubleVar(value=app.double_peak_max_distance.get() * 1000)
    
    # Function to update the application variables from the ms values
    def update_app_distance_values():
        app.double_peak_min_distance.set(min_distance_ms.get() / 1000)
        app.double_peak_max_distance.set(max_distance_ms.get() / 1000)
        # No longer trigger analysis automatically
    
    # Functions to synchronize entry and slider (but don't trigger automatic refresh)
    def sync_min_slider_to_entry(*args):
        try:
            value = float(min_entry.get())
            if value >= 0.1 and value <= max_distance_ms.get():
                min_slider.set(value)
                min_distance_ms.set(value)
                # Only update the values, don't trigger analysis
                app.double_peak_min_distance.set(value / 1000)
            else:
                # Reset entry to slider value if out of range
                min_entry.delete(0, tk.END)
                min_entry.insert(0, f"{min_distance_ms.get():.1f}")
        except ValueError:
            # Reset entry to slider value if invalid
            min_entry.delete(0, tk.END)
            min_entry.insert(0, f"{min_distance_ms.get():.1f}")

    def sync_max_slider_to_entry(*args):
        try:
            value = float(max_entry.get())
            if value >= min_distance_ms.get() and value <= 50.0:
                max_slider.set(value)
                max_distance_ms.set(value)
                # Only update the values, don't trigger analysis
                app.double_peak_max_distance.set(value / 1000)
            else:
                # Reset entry to slider value if out of range
                max_entry.delete(0, tk.END)
                max_entry.insert(0, f"{max_distance_ms.get():.1f}")
        except ValueError:
            # Reset entry to slider value if invalid
            max_entry.delete(0, tk.END)
            max_entry.insert(0, f"{max_distance_ms.get():.1f}")
    
    # Function to update entry from slider
    def update_min_entry(val):
        val = float(val)
        min_entry.delete(0, tk.END)
        min_entry.insert(0, f"{val:.1f}")
        min_distance_ms.set(val)
        # Only update the values, don't trigger analysis
        app.double_peak_min_distance.set(val / 1000)
        
    def update_max_entry(val):
        val = float(val)
        max_entry.delete(0, tk.END)
        max_entry.insert(0, f"{val:.1f}")
        max_distance_ms.set(val)
        # Only update the values, don't trigger analysis
        app.double_peak_max_distance.set(val / 1000)
    
    # Min distance slider and entry
    min_slider_frame = ttk.Frame(distance_frame)
    min_slider_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(min_slider_frame, text="Min:").pack(side=tk.LEFT, padx=5)
    min_slider = ttk.Scale(
        min_slider_frame, 
        from_=0.1, 
        to=25.0,
        variable=min_distance_ms, 
        orient=tk.HORIZONTAL,
        command=update_min_entry
    )
    min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Text entry for precise min distance
    min_entry = ttk.Entry(min_slider_frame, width=6)
    min_entry.pack(side=tk.LEFT, padx=5)
    min_entry.insert(0, f"{min_distance_ms.get():.1f}")
    min_entry.bind("<Return>", sync_min_slider_to_entry)
    min_entry.bind("<FocusOut>", sync_min_slider_to_entry)
    
    ttk.Label(min_slider_frame, text="ms").pack(side=tk.LEFT)
    
    # Max distance slider and entry
    max_slider_frame = ttk.Frame(distance_frame)
    max_slider_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(max_slider_frame, text="Max:").pack(side=tk.LEFT, padx=5)
    max_slider = ttk.Scale(
        max_slider_frame, 
        from_=1.0, 
        to=50.0,
        variable=max_distance_ms, 
        orient=tk.HORIZONTAL,
        command=update_max_entry
    )
    max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Text entry for precise max distance
    max_entry = ttk.Entry(max_slider_frame, width=6)
    max_entry.pack(side=tk.LEFT, padx=5)
    max_entry.insert(0, f"{max_distance_ms.get():.1f}")
    max_entry.bind("<Return>", sync_max_slider_to_entry)
    max_entry.bind("<FocusOut>", sync_max_slider_to_entry)
    
    ttk.Label(max_slider_frame, text="ms").pack(side=tk.LEFT)
    
    # Amplitude ratio parameters
    amp_frame = ttk.LabelFrame(param_frame, text="Amplitude Ratio")
    amp_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Grid layout for amplitude ratio controls
    ttk.Label(amp_frame, text="Range:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    min_amp_entry = ttk.Entry(amp_frame, textvariable=app.double_peak_min_amp_ratio, width=8)
    min_amp_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    ttk.Label(amp_frame, text="to").grid(row=0, column=2, sticky="ew", padx=2, pady=2)
    max_amp_entry = ttk.Entry(amp_frame, textvariable=app.double_peak_max_amp_ratio, width=8)
    max_amp_entry.grid(row=0, column=3, sticky="ew", padx=5, pady=2)
    
    # Histogram frame using grid
    amp_hist_frame = ttk.Frame(amp_frame)
    amp_hist_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=5, pady=(2, 5))
    
    # Create small figure for amplitude ratio histogram with adjusted size and tight layout
    amp_hist_fig = Figure(figsize=(2.5, 1.0), dpi=100)
    amp_hist_fig.set_tight_layout(True)  # Enable tight layout
    amp_hist_canvas = FigureCanvasTkAgg(amp_hist_fig, amp_hist_frame)
    
    # Create and store the axis with adjusted position
    app.amp_hist_ax = amp_hist_fig.add_subplot(111)
    app.amp_hist_ax.set_xlim(0, 5)
    app.amp_hist_ax.set_ylim(0, 1)
    app.amp_hist_ax.set_xticks([0, 1, 2, 3, 4, 5])
    app.amp_hist_ax.set_yticks([])
    app.amp_hist_ax.grid(True, alpha=0.3)
    app.amp_hist_ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Adjust the subplot parameters to prevent cutoff
    amp_hist_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.95)
    
    amp_hist_canvas.draw()
    amp_hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    # Store the canvas for later updates
    app.amp_hist_canvas = amp_hist_canvas
    
    # Configure grid columns for amplitude ratio frame
    amp_frame.columnconfigure(1, weight=1)
    amp_frame.columnconfigure(3, weight=1)
    amp_frame.rowconfigure(1, weight=1)
    
    # Width ratio parameters
    width_frame = ttk.LabelFrame(param_frame, text="Width Ratio")
    width_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Grid layout for width ratio controls
    ttk.Label(width_frame, text="Range:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    min_width_entry = ttk.Entry(width_frame, textvariable=app.double_peak_min_width_ratio, width=8)
    min_width_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    ttk.Label(width_frame, text="to").grid(row=0, column=2, sticky="ew", padx=2, pady=2)
    max_width_entry = ttk.Entry(width_frame, textvariable=app.double_peak_max_width_ratio, width=8)
    max_width_entry.grid(row=0, column=3, sticky="ew", padx=5, pady=2)
    
    # Width ratio histogram frame
    width_hist_frame = ttk.Frame(width_frame)
    width_hist_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=5, pady=(2, 5))
    
    # Create small figure for width ratio histogram with adjusted size and tight layout
    width_hist_fig = Figure(figsize=(2.5, 1.0), dpi=100)
    width_hist_fig.set_tight_layout(True)  # Enable tight layout
    width_hist_canvas = FigureCanvasTkAgg(width_hist_fig, width_hist_frame)
    
    # Create and store the axis with adjusted position
    app.width_hist_ax = width_hist_fig.add_subplot(111)
    app.width_hist_ax.set_xlim(0, 5)
    app.width_hist_ax.set_ylim(0, 1)
    app.width_hist_ax.set_xticks([0, 1, 2, 3, 4, 5])
    app.width_hist_ax.set_yticks([])
    app.width_hist_ax.grid(True, alpha=0.3)
    app.width_hist_ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Adjust the subplot parameters to prevent cutoff
    width_hist_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.95)
    
    width_hist_canvas.draw()
    width_hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    
    # Store the canvas for later updates
    app.width_hist_canvas = width_hist_canvas
    
    # Configure grid columns for width ratio frame
    width_frame.columnconfigure(1, weight=1)
    width_frame.columnconfigure(3, weight=1)
    width_frame.rowconfigure(1, weight=1)
    
    # Add explanation text
    explanation_text = (
        "Double peak analysis identifies pairs of peaks that meet specific criteria:\n\n"
        "• Distance Range: Time separation between peaks (in milliseconds)\n"
        "• Amplitude Ratio: Ratio of secondary to primary peak amplitude\n"
        "• Width Ratio: Ratio of secondary to primary peak width\n\n"
        "These parameters help identify the primary peak and its associated secondary peak in a double peak pair.\n"
        "The primary peak can be either the higher or lower amplitude peak - this is determined by the flow dynamics\n"
        "of your experiment. The amplitude and width ratios help distinguish between true double peaks and\n"
        "random peak pairs by ensuring the peaks have similar characteristics.\n\n"
        "Note: Adjust parameters using sliders or text entries. Changes will not affect the visualization until you\n"
        "click 'Analyze Double Peaks' to update the results."
    )
    explanation = ttk.Label(
        param_frame, 
        text=explanation_text,
        wraplength=380, 
        justify=tk.LEFT,
        padding=(5, 5)
    )
    explanation.pack(fill=tk.X, padx=5, pady=5)
    
    # Action buttons
    action_frame = ttk.LabelFrame(double_peak_tab, text="Actions")
    action_frame.pack(fill=tk.X, padx=5, pady=5)
    
    button_frame = ttk.Frame(action_frame)
    button_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Analyze button
    analyze_btn = ttk.Button(
        button_frame,
        text="Analyze Double Peaks",
        command=app.analyze_double_peaks,
        style="Primary.TButton"
    )
    analyze_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    # Show grid button
    show_grid_btn = ttk.Button(
        button_frame,
        text="Show Grid View",
        command=app.show_double_peaks_grid
    )
    show_grid_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    # Export button
    export_btn = ttk.Button(
        button_frame,
        text="Export Double Peak Data",
        command=app.save_double_peak_information_to_csv
    )
    export_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    # Navigation frame
    nav_frame = ttk.Frame(action_frame)
    nav_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Previous page button
    prev_btn = ttk.Button(
        nav_frame,
        text="Previous Page",
        command=app.show_prev_double_peaks_page
    )
    prev_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    # Next page button
    next_btn = ttk.Button(
        nav_frame,
        text="Next Page",
        command=app.show_next_double_peaks_page
    )
    next_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    # Add tooltips
    app.add_tooltip(
        analyze_btn,
        "Update visualization and detect double peaks using current parameters.\nClick this button after adjusting any parameters to see changes."
    )
    app.add_tooltip(
        show_grid_btn,
        "Show grid view of detected double peak pairs"
    )
    app.add_tooltip(
        export_btn,
        "Save double peak information including distances and width ratios to a CSV file"
    )
    app.add_tooltip(
        prev_btn,
        "Show previous page of double peak pairs"
    )
    app.add_tooltip(
        next_btn,
        "Show next page of double peak pairs"
    )

def create_export_options_dialog(parent):
    """
    Create a dialog window for selecting export options.
    
    Parameters
    ----------
    parent : tkinter.Tk or tkinter.Toplevel
        Parent window
        
    Returns
    -------
    tuple
        (file_format, delimiter, include_metadata)
    """
    dialog = tk.Toplevel(parent)
    dialog.title("Export Options")
    dialog.transient(parent)
    dialog.grab_set()
    
    # Center dialog on parent window
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
    y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
    dialog.geometry(f"+{x}+{y}")
    
    # Format selection
    ttk.Label(dialog, text="File Format:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
    format_var = tk.StringVar(value="csv")
    format_combo = ttk.Combobox(dialog, textvariable=format_var, state="readonly")
    format_combo['values'] = ("csv", "txt")
    format_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    
    # Delimiter selection
    ttk.Label(dialog, text="Delimiter:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
    delimiter_var = tk.StringVar(value=",")
    delimiter_combo = ttk.Combobox(dialog, textvariable=delimiter_var, state="readonly")
    delimiter_combo['values'] = (",", ";", "\t", "|")
    delimiter_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    
    # Metadata checkbox
    metadata_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(dialog, text="Include metadata header", variable=metadata_var).grid(
        row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w"
    )
    
    # Result variable
    result = [None, None, None]
    
    def on_ok():
        result[0] = format_var.get()
        result[1] = delimiter_var.get()
        result[2] = metadata_var.get()
        dialog.destroy()
    
    def on_cancel():
        dialog.destroy()
    
    # Buttons
    button_frame = ttk.Frame(dialog)
    button_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=10)
    
    ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
    
    # Wait for dialog to close
    dialog.wait_window()
    
    return tuple(result) if result[0] is not None else ("csv", ",", True) 