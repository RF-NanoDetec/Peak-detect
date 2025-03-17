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
    edit_menu.add_command(label="Clear Results", command=lambda: app.update_results_summary(preview_text=""))
    
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
    
    # Results section
    results_frame = ttk.LabelFrame(control_frame, text="Analysis Results")
    results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    app.results_summary = ScrolledText(
        results_frame, 
        height=10,
        wrap=tk.WORD,
        bg=app.theme_manager.get_color('card_bg'),
        fg=app.theme_manager.get_color('text'),
        insertbackground=app.theme_manager.get_color('text'),  # Cursor color
        font=app.theme_manager.get_font('default'),
        state=tk.DISABLED
    )
    app.results_summary.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Clear results button
    ttk.Button(
        results_frame,
        text="Clear Results",
        command=lambda: app.update_results_summary(preview_text="")
    ).pack(pady=5)

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
        text="Single File", 
        variable=app.file_mode, 
        value="single",
        command=app.on_file_mode_change
    ).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Radiobutton(
        file_mode_frame, 
        text="Batch Mode", 
        variable=app.file_mode, 
        value="batch",
        command=app.on_file_mode_change
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Add tooltips for file mode selection
    app.add_tooltip(
        file_mode_frame,
        "Choose between single file analysis or batch processing of multiple files"
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
    explanation_frame = ttk.Frame(time_res_frame)
    explanation_frame.pack(fill=tk.X, padx=5, pady=2)
    
    explanation_text = (
        "This value represents the time interval between data points and is crucial for correct peak width calculations.\n"
        "For most measurements, the default value of 0.1 milliseconds is appropriate."
    )
    explanation_label = ttk.Label(explanation_frame, text=explanation_text, wraplength=380, justify=tk.LEFT)
    explanation_label.pack(anchor=tk.W, padx=5, pady=2)
    
    # Create entry container frame
    entry_frame = ttk.Frame(time_res_frame)
    entry_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create colored frame to highlight importance
    highlight_frame = ttk.Frame(entry_frame, style="Accent.TFrame")
    highlight_frame.pack(fill=tk.X, padx=2, pady=2)
    
    time_res_label = ttk.Label(highlight_frame, text="Dwell Time (ms):", font=("TkDefaultFont", 10, "bold"))
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
    
    time_res_entry = ttk.Entry(
        highlight_frame, 
        textvariable=app.time_resolution_ms,
        width=10,
        font=("TkDefaultFont", 10, "bold"),
        validate="key", 
        validatecommand=(app.register(lambda P: validate_float_entry(P)), "%P")
    )
    time_res_entry.pack(side=tk.LEFT, padx=5, pady=5)
    
    units_label = ttk.Label(highlight_frame, text="milliseconds", font=("TkDefaultFont", 10))
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
        ("Start Time:", app.protocol_start_time),
        ("Particle:", app.protocol_particle),
        ("Concentration:", app.protocol_concentration),
        ("Stamp:", app.protocol_stamp),
        ("Laser Power:", app.protocol_laser_power),
        ("Setup:", app.protocol_setup)
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
    filter_color = ttk.Label(
        mode_selector,
        text="   ",
        background="#0078D7",  # Use blue directly instead of primary theme color
        relief=tk.RAISED,
        borderwidth=2
    )
    filter_color.pack(side=tk.LEFT, padx=(0, 30))
    
    raw_radio = ttk.Radiobutton(
        mode_selector,
        text="Raw Data",
        variable=app.filter_enabled,
        value=False,
        command=lambda: update_filter_state(False)
    )
    raw_radio.pack(side=tk.LEFT, padx=20)
    
    # Create a color chip to show raw data is noisy
    raw_color = ttk.Label(
        mode_selector,
        text="   ",
        background="#333333",  # Change from #FF6B6B to dark gray
        relief=tk.RAISED,
        borderwidth=2
    )
    raw_color.pack(side=tk.LEFT)
    
    # Add visual comparison of filtered vs raw data
    comparison_frame = ttk.Frame(mode_frame)
    comparison_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Create canvas for comparison visualization
    canvas_height = 80
    canvas_width = 380
    comparison = tk.Canvas(
        comparison_frame, 
        height=canvas_height, 
        width=canvas_width,
        bg=app.theme_manager.get_color('card_bg'),
        highlightthickness=0
    )
    comparison.pack(pady=5)
    
    # Draw a comparison of raw vs filtered data
    baseline_y = canvas_height // 2
    
    # Draw axis
    comparison.create_line(
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
    comparison.create_line(raw_points, fill="#333333", width=1.5, smooth=False)
    
    # Draw filtered data (smooth)
    filtered_points = []
    for x in range(10, canvas_width-10, 3):
        # Create a smooth sine wave
        y = baseline_y - 15 * np.sin((x-10) / 30)
        filtered_points.append(x)
        filtered_points.append(int(y))
    
    # Create filtered data curve
    comparison.create_line(filtered_points, fill="#0078D7", width=2, smooth=True)
    
    # Add labels
    comparison.create_text(
        30, 15, 
        text="Filtered: Smoother signal, reduced noise", 
        fill="#0078D7",  # Use blue directly 
        anchor=tk.W,
        font=("TkDefaultFont", 8, "bold")
    )
    comparison.create_text(
        30, 30, 
        text="Raw: Original signal with noise", 
        fill="#333333",  # Change from #FF6B6B to dark gray
        anchor=tk.W,
        font=("TkDefaultFont", 8, "bold")
    )
    
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

    # Create a dedicated Auto Threshold frame with clear explanation
    auto_threshold_frame = ttk.LabelFrame(peak_detection_tab, text="Automatic Threshold Detection")
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
    diagram = tk.Canvas(
        diagram_frame, 
        height=canvas_height, 
        width=canvas_width,
        bg=app.theme_manager.get_color('card_bg'),
        highlightthickness=0
    )
    diagram.pack()
    
    # Draw a sine-like signal to represent data
    signal_color = "#0078D7"  # Use blue directly instead of theme's primary color
    baseline_y = canvas_height // 2 + 15  # Move baseline down to show peaks better
    
    # Create a single data line with baseline noise and peaks
    data_points = []
    np.random.seed(42)  # For consistent noise pattern
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
    diagram.create_line(data_points, fill=signal_color, width=2, smooth=True)
    
    # Draw threshold lines for different sigma values
    low_thresh_y = baseline_y - 15  # Low threshold (catches small peaks too)
    med_thresh_y = baseline_y - 25  # Medium threshold (balanced)
    high_thresh_y = baseline_y - 40  # High threshold (only the largest peaks)
    
    # Low sigma (e.g., σ=2) - will detect all peaks including some noise
    diagram.create_line(
        10, low_thresh_y, canvas_width-10, low_thresh_y,
        fill="#4CAF50", width=1, dash=(2, 2)
    )
    diagram.create_text(
        canvas_width-15, low_thresh_y-8, 
        text="σ=2", 
        fill="#4CAF50", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # Medium sigma (e.g., σ=5) - balanced threshold
    diagram.create_line(
        10, med_thresh_y, canvas_width-10, med_thresh_y,
        fill="#FF9800", width=1, dash=(2, 2)
    )
    diagram.create_text(
        canvas_width-15, med_thresh_y-8, 
        text="σ=5", 
        fill="#FF9800", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # High sigma (e.g., σ=8) - only detects the largest peak
    diagram.create_line(
        10, high_thresh_y, canvas_width-10, high_thresh_y,
        fill="#F44336", width=1, dash=(2, 2)
    )
    diagram.create_text(
        canvas_width-15, high_thresh_y-8, 
        text="σ=8", 
        fill="#F44336", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # Add markers to show which peaks are detected at each threshold
    # For low threshold (detects all peaks)
    for x_pos in [80, 190, 280]:
        diagram.create_oval(
            x_pos-3, low_thresh_y-3, 
            x_pos+3, low_thresh_y+3, 
            fill="#4CAF50", outline=""
        )
        
    # For medium threshold (detects medium and large peaks)
    for x_pos in [190, 280]:
        diagram.create_oval(
            x_pos-3, med_thresh_y-3, 
            x_pos+3, med_thresh_y+3, 
            fill="#FF9800", outline=""
        )
        
    # For high threshold (detects only the largest peak)
    diagram.create_oval(
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
    
    # Improved slider with labels for values
    sigma_slider = tk.Scale(
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
    sigma_slider.pack(fill=tk.X, expand=True)
    
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
        sigma_slider,
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
    ttk.Separator(peak_detection_tab, orient="horizontal").pack(fill=tk.X, padx=10, pady=10)
    
    # Peak Parameters Frame - now separate from auto threshold
    peak_params_frame = ttk.LabelFrame(peak_detection_tab, text="Manual Peak Detection Parameters")
    peak_params_frame.pack(fill=tk.X, padx=5, pady=5)

    # Other peak parameters
    row = 0
    ttk.Label(peak_params_frame, text="Min. Distance Peaks").grid(row=row, column=0, sticky="w", padx=5, pady=2)
    distance_entry = ttk.Entry(peak_params_frame, textvariable=app.distance)
    distance_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

    row += 1
    ttk.Label(peak_params_frame, text="Relative Height").grid(row=row, column=0, sticky="w", padx=5, pady=2)
    rel_height_entry = ttk.Entry(peak_params_frame, textvariable=app.rel_height)
    rel_height_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

    row += 1
    ttk.Label(peak_params_frame, text="Width Range (ms)").grid(row=row, column=0, sticky="w", padx=5, pady=2)
    width_entry = ttk.Entry(peak_params_frame, textvariable=app.width_p)
    width_entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)

    # Action Buttons Frame
    peak_detection_frame = ttk.LabelFrame(peak_detection_tab, text="Peak Detection Actions")
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

    # Configure grid weights
    peak_params_frame.columnconfigure(1, weight=1)

    # Add tooltips for better user guidance
    app.add_tooltip(
        distance_entry,
        "Minimum number of points between peaks.\n"
        "Higher values prevent detecting multiple peaks too close together."
    )
    app.add_tooltip(
        rel_height_entry,
        "Relative height (0-1) at which peak width is measured.\n"
        "Example: 0.5 = width at half maximum height, 0.9 = width near peak top"
    )
    app.add_tooltip(
        width_entry,
        "Expected peak width range in milliseconds (min,max).\n"
        "Example: '1,200' means only keep peaks between 1-200ms wide"
    )
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

    # Export Frame
    export_frame = ttk.LabelFrame(peak_analysis_tab, text="Export Options")
    export_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Button(
        export_frame,
        text="Export Peak Data to CSV",  # Changed from "Save Peak Information"
        command=app.save_peak_information_to_csv
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
    app.add_tooltip(
        export_frame.winfo_children()[0],  # Export button
        "Save all peak information to a CSV file for further analysis"
    ) 