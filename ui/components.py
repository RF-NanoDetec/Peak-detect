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

def create_menu_bar(parent_window, callbacks, current_theme_name):
    """Create the application menu bar"""
    menu_bar = tk.Menu(parent_window)
    parent_window.config(menu=menu_bar)
    
    # File Menu
    file_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open File", command=callbacks['browse_file'])
    file_menu.add_command(label="Export Results", command=callbacks['save_peak_info'])
    file_menu.add_separator()
    file_menu.add_command(label="Export Current Plot", command=callbacks['export_plot'])
    file_menu.add_command(label="Take Screenshot", command=callbacks['take_screenshot'])
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=callbacks['quit'])
    
    # Edit Menu
    edit_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Edit", menu=edit_menu)
    edit_menu.add_command(label="Reset Application", command=callbacks['reset_state'])
    
    # View Menu
    view_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="View", menu=view_menu)
    view_menu.add_command(label="Raw Data", command=callbacks['plot_raw_data'])
    view_menu.add_command(label="Filtered Data", command=callbacks['start_analysis'])
    view_menu.add_command(label="Detected Peaks", command=callbacks['run_peak_detection'])
    view_menu.add_separator()
    view_menu.add_command(label="Peak Analysis", command=callbacks['plot_data'])
    view_menu.add_command(label="Peak Correlations", command=callbacks['plot_scatter'])
    view_menu.add_separator()
    # Add theme toggle option
    theme_toggle_label = f"Switch to {'Light' if current_theme_name == 'dark' else 'Dark'} Theme"
    view_menu.add_command(label=theme_toggle_label, command=callbacks['toggle_theme'])
    
    # Tools Menu
    tools_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Tools", menu=tools_menu)
    tools_menu.add_command(label="Auto Calculate Threshold", command=callbacks['calc_auto_threshold'])
    tools_menu.add_command(label="Auto Calculate Cutoff", command=callbacks['calc_auto_cutoff'])
    tools_menu.add_separator()
    tools_menu.add_command(label="View Individual Peaks", command=callbacks['plot_filtered_peaks'])
    tools_menu.add_command(label="Next Peaks", command=callbacks['show_next_peaks'])
    
    # Help Menu
    help_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Help", menu=help_menu)
    help_menu.add_command(label="Documentation", command=callbacks['show_documentation'])
    help_menu.add_command(label="About", command=callbacks['show_about'])
    
    return menu_bar

def create_control_panel(parent_frame, state, theme_manager, callbacks, status_indicator_class):
    """Create the control panel with tabs"""
    control_frame = ttk.Frame(parent_frame)
    control_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    # Set a fixed width for the control panel
    control_frame.grid_propagate(False)  # Prevent the frame from resizing to its children
    control_frame.configure(width=400)   # Set a fixed width
    
    # Add status indicator at the top
    status_frame = ttk.Frame(control_frame)
    status_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Status indicator
    status_indicator = status_indicator_class(status_frame, theme_manager=theme_manager)
    status_indicator.pack(fill=tk.X, padx=5, pady=5)
    
    # Create notebook (tabbed interface) for controls
    tab_control = ttk.Notebook(control_frame)
    tab_control.pack(fill=tk.BOTH, expand=True)

    # Create tabs
    create_data_loading_tab(tab_control, state, theme_manager, callbacks)
    preprocessing_widgets = create_preprocessing_tab(tab_control, state, theme_manager, callbacks)
    peak_detection_widgets = create_peak_detection_tab(tab_control, state, theme_manager, callbacks)
    peak_analysis_widgets = create_peak_analysis_tab(tab_control, state, theme_manager, callbacks)
    
    # Add double peak analysis tab if enabled
    double_peak_widgets = None
    if state.double_peak_analysis.get() == "1":
        double_peak_widgets = create_double_peak_analysis_tab(tab_control, state, theme_manager, callbacks)

    # Progress bar with green color
    progress = ttk.Progressbar(
        control_frame, 
        mode='determinate',
        style='Green.Horizontal.TProgressbar'
    )
    progress.pack(fill=tk.X, padx=5, pady=5)

    # Preview label - created locally
    preview_label = ttk.Label(control_frame, text="", foreground="black") # Color needs theme awareness later
    preview_label.pack(fill=tk.X, padx=5, pady=5)
    
    # Return references to the created widgets needed by the main application
    return {
        "frame": control_frame,
        "status_indicator": status_indicator,
        "tab_control": tab_control,
        "progress_bar": progress,
        "preview_label": preview_label,
        "peak_detection_widgets": peak_detection_widgets,
        "double_peak_widgets": double_peak_widgets
    }
    # return control_frame

# Updated signature: accepts parent, theme_manager, callbacks
def create_preview_frame(parent_frame, theme_manager, callbacks):
    """Create the preview frame with plot tabs"""
    preview_frame = ttk.Frame(parent_frame) # Use parent_frame
    preview_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    preview_frame.columnconfigure(0, weight=1)
    preview_frame.rowconfigure(0, weight=1)
    preview_frame.rowconfigure(1, weight=0)
    
    # Configure parent_frame to give more weight to the preview column
    # This assumes the caller (Application.create_widgets) correctly configures its own layout
    # parent_frame.columnconfigure(0, weight=0) # Control panel doesn't need to expand
    # parent_frame.columnconfigure(1, weight=1) # Preview frame gets all extra space
    
    # Tab Control for Multiple Plots on the right - create locally
    plot_tab_control = ttk.Notebook(preview_frame)
    plot_tab_control.grid(row=0, column=0, sticky="nsew")

    # Create an empty frame with fixed size instead of blank image - create locally
    blank_tab = ttk.Frame(plot_tab_control, width=800, height=600)
    plot_tab_control.add(blank_tab, text="Welcome")
    
    # Add a welcome label with theme-appropriate styling - use theme_manager arg
    welcome_label = ttk.Label(
        blank_tab, 
        text="Welcome to Peak Analysis Tool\n\nPlease load a file to begin", 
        font=("Arial", 14),
        foreground=theme_manager.get_color('text'),
        background=theme_manager.get_color('background')
    )
    welcome_label.place(relx=0.5, rely=0.5, anchor="center")
    
    # Prevent the blank tab from shrinking
    blank_tab.pack_propagate(False)

    # Functional Bar under plot tabs
    functional_bar = ttk.Frame(preview_frame)
    functional_bar.grid(row=1, column=0, sticky="ew", pady=10)

    # Use callbacks dict for commands
    ttk.Button(functional_bar, 
              text="Export Plot", 
              command=callbacks['export_plot'] 
    ).grid(row=0, column=0, padx=5, pady=5)
    
    # Add scale toggle button
    scale_toggle_btn = ttk.Button(
        functional_bar,
        text="Toggle Scale (Log/Linear)",
        command=callbacks['toggle_scale_mode']
    )
    scale_toggle_btn.grid(row=0, column=1, padx=5, pady=5)
    
    # Add tooltip for scale toggle button - use callback
    add_tooltip_callback = callbacks.get('add_tooltip') # Use .get for safety
    if add_tooltip_callback:
        add_tooltip_callback(
            scale_toggle_btn,
            "Toggle between logarithmic and linear scales for peak analysis plots"
        )
    
    # Return references needed by Application
    return {
        "frame": preview_frame,
        "plot_tab_control": plot_tab_control,
        "blank_tab": blank_tab
    }
    # return preview_frame

# Updated signature and logic
def create_data_loading_tab(parent_tab_control, state, theme_manager, callbacks):
    """Create the data loading tab"""
    data_loading_tab = ttk.Frame(parent_tab_control)
    parent_tab_control.add(data_loading_tab, text="Data Loading")

    add_tooltip_callback = callbacks.get('add_tooltip')

    # File mode selection frame
    file_mode_frame = ttk.LabelFrame(data_loading_tab, text="File Mode")
    file_mode_frame.pack(fill=tk.X, padx=5, pady=5)

    # Radio buttons for file mode - Use state and callbacks
    ttk.Radiobutton(
        file_mode_frame,
        text="Standard Mode",
        variable=state.file_mode, # Use state
        value="single",
        command=callbacks['on_file_mode_change'] # Use callback
    ).pack(side=tk.LEFT, padx=5, pady=5)
    ttk.Radiobutton(
        file_mode_frame,
        text="Timestamp Mode",
        variable=state.file_mode, # Use state
        value="batch",
        command=callbacks['on_file_mode_change'] # Use callback
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Add tooltips for file mode selection - Use callback
    if add_tooltip_callback:
        add_tooltip_callback(
            file_mode_frame,
            "Standard Mode: Load single or multiple files with automatic time sequencing\nTimestamp Mode: Load multiple files with custom timestamps"
        )

    # Add peak analysis mode frame (normal vs double peak)
    peak_mode_frame = ttk.LabelFrame(data_loading_tab, text="Peak Analysis Mode")
    peak_mode_frame.pack(fill=tk.X, padx=5, pady=5)

    peak_mode_container = ttk.Frame(peak_mode_frame)
    peak_mode_container.pack(padx=5, pady=5, anchor=tk.W)

    # Radio buttons for normal vs double peak analysis - Use state and callbacks
    ttk.Radiobutton(
        peak_mode_container,
        text="Normal Analysis",
        variable=state.double_peak_analysis, # Use state
        value="0",
        command=callbacks['on_double_peak_mode_change'] # Use callback
    ).pack(side=tk.LEFT, padx=10)
    ttk.Radiobutton(
        peak_mode_container,
        text="Double Peak Analysis",
        variable=state.double_peak_analysis, # Use state
        value="1",
        command=callbacks['on_double_peak_mode_change'] # Use callback
    ).pack(side=tk.LEFT, padx=10)

    # Add tooltip for peak analysis mode - Use callback
    if add_tooltip_callback:
        add_tooltip_callback(
            peak_mode_frame,
            "Select 'Double Peak Analysis' to enable additional canvas for analyzing double peaks"
        )

    # File selection frame
    file_frame = ttk.LabelFrame(data_loading_tab, text="File Selection")
    file_frame.pack(fill=tk.X, padx=5, pady=5)

    # Browse button - Use callback
    browse_button = ttk.Button(
        file_frame,
        text="Load File",
        command=callbacks['browse_file'], # Use callback
        style="Primary.TButton"
    )
    browse_button.pack(side=tk.LEFT, padx=5, pady=5)

    file_name_label = ttk.Label(file_frame, text="No file selected")
    file_name_label.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

    # Timestamps entry for batch mode - Use state
    timestamps_label = ttk.Label(file_frame, text="Timestamps:")
    timestamps_entry = ttk.Entry(file_frame, textvariable=state.batch_timestamps) # Use state

    # Initially hide timestamps widgets (App needs to manage this based on mode change)
    timestamps_label.pack_forget()
    timestamps_entry.pack_forget()

    # Time resolution configuration
    time_res_frame = ttk.LabelFrame(data_loading_tab, text="⚠️ Dwell Time - Critical Setting ⚠️")
    time_res_frame.pack(fill=tk.X, padx=5, pady=10, ipady=5)

    explanation_frame = ttk.Frame(time_res_frame)
    explanation_frame.pack(fill=tk.X, padx=5, pady=2)
    explanation_text = (
        "This value represents the time interval between data points..."
    )
    explanation_label = ttk.Label(explanation_frame, text=explanation_text, wraplength=380, justify=tk.LEFT)
    explanation_label.pack(fill=tk.X, padx=5, pady=2)

    # Time resolution entry - Use state
    time_res_entry_frame = ttk.Frame(time_res_frame)
    time_res_entry_frame.pack(fill=tk.X, padx=5, pady=2)
    ttk.Label(time_res_entry_frame, text="Dwell Time (s):").pack(side=tk.LEFT, padx=5)
    time_res_entry = ttk.Entry(time_res_entry_frame, textvariable=state.time_resolution, width=10)
    time_res_entry.pack(side=tk.LEFT, padx=5)

    if add_tooltip_callback:
        add_tooltip_callback(time_res_frame, "Dwell time is the time interval...")
        add_tooltip_callback(browse_button, "Click to select a data file...")
        add_tooltip_callback(timestamps_entry, "Enter timestamps for batch files...")

    # Protocol information frame - Use state
    protocol_frame = ttk.LabelFrame(data_loading_tab, text="Protocol Information")
    protocol_frame.pack(fill=tk.X, padx=5, pady=5)

    protocol_entries = [
        ("Measurement Date:", state.protocol_measurement_date),
        ("Start Time:", state.protocol_start_time),
        ("Setup:", state.protocol_setup),
        ("Sample Number:", state.protocol_sample_number),
        ("Particle:", state.protocol_particle),
        ("Particle Concentration:", state.protocol_concentration),
        ("Buffer:", state.protocol_buffer),
        ("Buffer Concentration:", state.protocol_buffer_concentration),
        ("ND Filter:", state.protocol_id_filter),
        ("Laser Power:", state.protocol_laser_power),
        ("Stamp:", state.protocol_stamp)
    ]
    protocol_tooltips = {
        "Measurement Date": "Enter the date of measurement in YYYY-MM-DD format",
        "Start Time": "Enter the experiment start time during the day in HH:MM:SS format (e.g., '13:30:00')",
        "Setup": "Enter the experimental setup configuration example: 'Prototype, Old Ladom'",
        "Sample Number": "Enter the sample number or identifier",
        "Particle": "Enter the type of particle or sample being analyzed",
        "Particle Concentration": "Enter the concentration of the particles or sample",
        "Buffer": "Enter the buffer solution used in the experiment",
        "Buffer Concentration": "Enter the concentration of the buffer solution",
        "ND Filter": "Enter the neutral density (ND) filter value used in the experiment",
        "Laser Power": "Enter the laser power settings used",
        "Stamp": "Enter any lithographic stamp name or identifier example: 'tripple-block'",
        "Notes": "Enter any additional notes or observations about the experiment"
    }

    for row, (label_text, variable) in enumerate(protocol_entries):
        ttk.Label(protocol_frame, text=label_text).grid(row=row, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(protocol_frame, textvariable=variable).grid(row=row, column=1, padx=5, pady=2, sticky="ew")
        # Add tooltips using callback
        if add_tooltip_callback:
             label_widget = protocol_frame.grid_slaves(row=row, column=0)[0]
             entry_widget = protocol_frame.grid_slaves(row=row, column=1)[0]
             tooltip_text = protocol_tooltips.get(label_text.rstrip(':'), "")
             add_tooltip_callback(label_widget, tooltip_text)
             add_tooltip_callback(entry_widget, tooltip_text)

    ttk.Label(protocol_frame, text="Notes:").grid(row=len(protocol_entries), column=0, padx=5, pady=2, sticky="w")
    notes_entry = ttk.Entry(protocol_frame, textvariable=state.protocol_notes)
    notes_entry.grid(row=len(protocol_entries), column=1, padx=5, pady=2, sticky="ew")
    if add_tooltip_callback:
        notes_label = protocol_frame.grid_slaves(row=len(protocol_entries), column=0)[0]
        notes_tooltip = "Enter any additional notes or observations about the experiment"
        add_tooltip_callback(notes_label, notes_tooltip)
        add_tooltip_callback(notes_entry, notes_tooltip)

    protocol_frame.columnconfigure(1, weight=1)

def create_preprocessing_tab(parent_tab_control, state, theme_manager, callbacks):
    """Create the preprocessing tab"""
    preprocessing_tab = ttk.Frame(parent_tab_control)
    parent_tab_control.add(preprocessing_tab, text="Preprocessing")

    add_tooltip_callback = callbacks.get('add_tooltip')

    # Create mode selector
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
    
    # Radio buttons for mode selection
    filter_radio = ttk.Radiobutton(
        toggle_frame,
        text="Filtered Data",
        variable=state.filter_enabled,
        value=True,
        command=lambda: update_filter_state(True)
    )
    filter_radio.pack(side=tk.LEFT, padx=10)
    
    raw_radio = ttk.Radiobutton(
        toggle_frame,
        text="Raw Data",
        variable=state.filter_enabled,
        value=False,
        command=lambda: update_filter_state(False)
    )
    raw_radio.pack(side=tk.LEFT, padx=10)

    # Create filtering parameters frame
    filtering_frame = ttk.LabelFrame(preprocessing_tab, text="Filtering Parameters")
    
    # Create comparison canvas
    comparison_canvas = tk.Canvas(
        mode_frame,
        width=380,
        height=100,
        bg=theme_manager.get_color('canvas_bg'),
        highlightthickness=0
    )
    comparison_canvas.pack(padx=5, pady=10)
    
    # Draw baseline
    baseline_y = 50
    canvas_width = 380
    comparison_canvas.create_line(
        10, baseline_y, canvas_width-10, baseline_y,
        fill=theme_manager.get_color('text_secondary'),
        dash=(2, 4)
    )
    
    # Draw raw data (noisy)
    raw_points = []
    
    # For consistent random noise
    for x in range(10, canvas_width-10, 3):
        # Create a noisy sine wave
        noise = np.random.normal(0, 6) if x % 9 != 0 else np.random.normal(0, 2)
        y = baseline_y - 15 * np.sin((x-10) / 30) + noise
        raw_points.append(x)
        raw_points.append(int(y))
    
    # Create raw data curve
    comparison_canvas.create_line(raw_points, fill=theme_manager.get_color('text_secondary'), width=1.5, smooth=False)
    
    # Draw filtered data (smooth)
    filtered_points = []
    for x in range(10, canvas_width-10, 3):
        # Create a smooth sine wave
        y = baseline_y - 15 * np.sin((x-10) / 30)
        filtered_points.append(x)
        filtered_points.append(int(y))
    
    comparison_canvas.create_line(filtered_points, fill=theme_manager.get_color('primary'), width=2, smooth=True)
    
    # Function to update UI based on filter state
    def update_filter_state(is_filtered):
        state.filter_enabled.set(is_filtered)
        
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
    if add_tooltip_callback:
        add_tooltip_callback(
            filter_radio,
            "Apply Butterworth low-pass filter to smooth the signal and reduce noise.\n"
            "Recommended for most signals to improve peak detection."
        )
        add_tooltip_callback(
            raw_radio,
            "Use raw unprocessed data without any filtering.\n"
            "Preserves original signal characteristics but may include more noise."
        )
    
    # Create a horizontal separator
    ttk.Separator(preprocessing_tab, orient="horizontal").pack(fill=tk.X, padx=10, pady=10)
    
    # Filtering parameters section
    filter_params_frame = ttk.Frame(filtering_frame)
    filter_params_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Cutoff frequency
    cutoff_frame = ttk.Frame(filter_params_frame)
    cutoff_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(cutoff_frame, text="Cutoff Frequency (Hz):").pack(side=tk.LEFT, padx=5)
    cutoff_entry = ttk.Entry(cutoff_frame, textvariable=state.cutoff_value, width=10)
    cutoff_entry.pack(side=tk.LEFT, padx=5)
    
    # Normalization factor
    norm_frame = ttk.Frame(filter_params_frame)
    norm_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(norm_frame, text="Normalization Factor:").pack(side=tk.LEFT, padx=5)
    norm_entry = ttk.Entry(norm_frame, textvariable=state.normalization_factor, width=10)
    norm_entry.pack(side=tk.LEFT, padx=5)
    
    # Auto-calculate button
    auto_calc_frame = ttk.Frame(filter_params_frame)
    auto_calc_frame.pack(fill=tk.X, padx=5, pady=5)
    
    auto_calc_btn = ttk.Button(
        auto_calc_frame,
        text="Auto-Calculate Cutoff",
        command=callbacks['calc_auto_cutoff'],
        style="Accent.TButton"
    )
    auto_calc_btn.pack(side=tk.LEFT, padx=5)
    
    # Add tooltips for filtering parameters
    if add_tooltip_callback:
        add_tooltip_callback(
            cutoff_entry,
            "Frequency cutoff for the Butterworth low-pass filter.\n"
            "Lower values = more smoothing but may lose peak details.\n"
            "Higher values = less smoothing but keeps more noise."
        )
        add_tooltip_callback(
            norm_entry,
            "Factor to adjust the auto-calculated cutoff frequency.\n"
            "Values > 1 increase cutoff (less filtering)\n"
            "Values < 1 decrease cutoff (more filtering)"
        )
        add_tooltip_callback(
            auto_calc_btn,
            "Automatically calculate optimal cutoff frequency based on signal characteristics.\n"
            "Uses peak width estimation to determine appropriate filtering level."
        )
    
    # Action Buttons with improved layout
    action_frame = ttk.LabelFrame(preprocessing_tab, text="Processing Actions")
    action_frame.pack(fill=tk.X, padx=5, pady=(10, 5))
    
    # Button container for better spacing
    button_container = ttk.Frame(action_frame)
    button_container.pack(fill=tk.X, padx=5, pady=10)
    
    view_raw_btn = ttk.Button(
        button_container,
        text="View Raw Data",
        command=callbacks['plot_raw_data'],
        style="Primary.TButton"
    )
    view_raw_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    process_btn = ttk.Button(
        button_container,
        text="Apply Filtering" if state.filter_enabled.get() else "Process Raw Data",
        command=callbacks['start_analysis'],
        style="Accent.TButton"
    )
    process_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    # Add tooltips for action buttons
    if add_tooltip_callback:
        add_tooltip_callback(
            view_raw_btn,
            "Display the original unprocessed data without any filtering"
        )
        add_tooltip_callback(
            process_btn,
            "Apply the selected processing mode to the data:\n"
            "• Filtered: Applies Butterworth filter with specified cutoff\n"
            "• Raw: Processes data without filtering"
        )
    
    # Return references to widgets that need theme updates
    return {
        "comparison_canvas": comparison_canvas,
        "filter_radio": filter_radio,
        "raw_radio": raw_radio,
        "cutoff_entry": cutoff_entry,
        "norm_entry": norm_entry,
        "auto_calc_btn": auto_calc_btn,
        "view_raw_btn": view_raw_btn,
        "process_btn": process_btn
    }

def create_peak_detection_tab(parent_tab_control, state, theme_manager, callbacks):
    """Create the peak detection tab"""
    peak_detection_tab = ttk.Frame(parent_tab_control)
    parent_tab_control.add(peak_detection_tab, text="Peak Detection")

    add_tooltip_callback = callbacks.get('add_tooltip')
    show_tooltip_popup_callback = callbacks.get('show_tooltip_popup') # Needed for info button

    # Scrollable frame setup
    main_container = ttk.Frame(peak_detection_tab)
    main_container.pack(fill=tk.BOTH, expand=True)
    peak_detection_main_canvas = tk.Canvas(main_container, bg=theme_manager.get_color('background'))
    scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=peak_detection_main_canvas.yview)
    scrollable_frame = ttk.Frame(peak_detection_main_canvas)
    scrollable_frame.bind("<Configure>", lambda e: peak_detection_main_canvas.configure(scrollregion=peak_detection_main_canvas.bbox("all")))
    peak_detection_main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    peak_detection_main_canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    peak_detection_main_canvas.pack(side="left", fill="both", expand=True)

    # Auto Threshold frame
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
    # Create the canvas locally
    threshold_diagram_canvas = tk.Canvas(
        diagram_frame, 
        height=canvas_height, 
        width=canvas_width,
        bg=theme_manager.get_color('card_bg'),
        highlightthickness=0
    )
    threshold_diagram_canvas.pack()
    
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
    threshold_diagram_canvas.create_line(data_points, fill=signal_color, width=2, smooth=True)
    
    # Draw threshold lines for different sigma values
    low_thresh_y = baseline_y - 15  # Low threshold (catches small peaks too)
    med_thresh_y = baseline_y - 25  # Medium threshold (balanced)
    high_thresh_y = baseline_y - 40  # High threshold (only the largest peaks)
    
    # Low sigma (e.g., σ=2) - will detect all peaks including some noise
    threshold_diagram_canvas.create_line(
        10, low_thresh_y, canvas_width-10, low_thresh_y,
        fill="#4CAF50", width=1, dash=(2, 2)
    )
    threshold_diagram_canvas.create_text(
        canvas_width-15, low_thresh_y-8, 
        text="σ=2", 
        fill="#4CAF50", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # Medium sigma (e.g., σ=5) - balanced threshold
    threshold_diagram_canvas.create_line(
        10, med_thresh_y, canvas_width-10, med_thresh_y,
        fill="#FF9800", width=1, dash=(2, 2)
    )
    threshold_diagram_canvas.create_text(
        canvas_width-15, med_thresh_y-8, 
        text="σ=5", 
        fill="#FF9800", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # High sigma (e.g., σ=8) - only detects the largest peak
    threshold_diagram_canvas.create_line(
        10, high_thresh_y, canvas_width-10, high_thresh_y,
        fill="#F44336", width=1, dash=(2, 2)
    )
    threshold_diagram_canvas.create_text(
        canvas_width-15, high_thresh_y-8, 
        text="σ=8", 
        fill="#F44336", 
        anchor=tk.E,
        font=("TkDefaultFont", 8)
    )
    
    # Add markers to show which peaks are detected at each threshold
    # For low threshold (detects all peaks)
    for x_pos in [80, 190, 280]:
        threshold_diagram_canvas.create_oval(
            x_pos-3, low_thresh_y-3, 
            x_pos+3, low_thresh_y+3, 
            fill="#4CAF50", outline=""
        )
        
    # For medium threshold (detects medium and large peaks)
    for x_pos in [190, 280]:
        threshold_diagram_canvas.create_oval(
            x_pos-3, med_thresh_y-3, 
            x_pos+3, med_thresh_y+3, 
            fill="#FF9800", outline=""
        )
        
    # For high threshold (detects only the largest peak)
    threshold_diagram_canvas.create_oval(
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
        text=f"{state.sigma_multiplier.get():.1f}",
        width=4,
        font=("TkDefaultFont", 10, "bold"),
        foreground=theme_manager.get_color('primary')
    )
    sigma_value_label.pack(side=tk.LEFT, padx=5)
    
    # Create a container for the slider to allow better styling
    slider_frame = ttk.Frame(sigma_container)
    slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Create the slider locally
    sigma_slider = tk.Scale(
        slider_frame, 
        from_=1.0, 
        to=10.0, 
        resolution=0.1,
        orient=tk.HORIZONTAL,
        variable=state.sigma_multiplier,
        length=250,
        bg=theme_manager.get_color('card_bg'),
        fg=theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=theme_manager.get_color('background'),
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
        sigma_value_label.config(text=f"{state.sigma_multiplier.get():.1f}")
    
    state.sigma_multiplier.trace_add("write", update_sigma_label)
    
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
        textvariable=state.height_lim, 
        width=8,
        font=("TkDefaultFont", 9, "bold")
    )
    threshold_entry.pack(side=tk.LEFT, padx=5)

    # Calculation button with improved style
    auto_calc_button = ttk.Button(
        buttons_container, 
        text="Calculate Threshold",
        command=callbacks['calc_auto_threshold'],
        style="Accent.TButton"
    )
    auto_calc_button.pack(side=tk.RIGHT, padx=5)
    
    # Add tooltips with detailed explanations
    if add_tooltip_callback:
        add_tooltip_callback(
            sigma_slider,
            "Adjust sensitivity of peak detection:\n"
            "• Lower values (1-3): More sensitive, detects smaller peaks\n"
            "• Medium values (4-6): Balanced detection for most data\n"
            "• Higher values (7-10): Less sensitive, only detects prominent peaks"
        )
    
    add_tooltip_callback(
        auto_calc_button,
        "Calculate threshold based on the current sigma value and signal statistics.\n"
        "The formula used is: Threshold = σ × Standard Deviation of Signal"
    )
    
    add_tooltip_callback(
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
    
    # Create the manual diagram canvas locally
    manual_diagram_canvas = tk.Canvas(
        visualization_frame,
        height=120,
        width=380,
        bg=theme_manager.get_color('card_bg'),
        highlightthickness=0
    )
    manual_diagram_canvas.pack(fill=tk.X, padx=5, pady=5)
    
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
    manual_diagram_canvas.create_line(data_points, fill=signal_color, width=2, smooth=True)
    
    # Add parameter indicators with better spacing and colors
    # 1. Minimum Distance between Peaks (between peak centers)
    distance_y = baseline_y + 15  # Move distance indicator up
    manual_diagram_canvas.create_line(
        peaks[0]['x'], distance_y,
        peaks[1]['x'], distance_y,
        fill="#FF6B6B", width=1, dash=(2, 2)
    )
    manual_diagram_canvas.create_text(
        (peaks[0]['x'] + peaks[1]['x'])/2, distance_y + 10,
        text="Distance between peaks",
        fill="black",
        font=("TkDefaultFont", 8)
    )
    
    # 2. Relative Height (measured from peak top)
    rel_height_y = baseline_y - peaks[0]['height'] * 0.2  # 20% from baseline (80% from top)
    # Draw line from peak top to width measurement height
    manual_diagram_canvas.create_line(
        peaks[0]['x'], baseline_y - peaks[0]['height'],  # Start from peak top
        peaks[0]['x'], rel_height_y,  # End at width measurement height
        fill="#4ECDC4", width=1, dash=(2, 2)
    )
    manual_diagram_canvas.create_text(
        peaks[0]['x'], rel_height_y - 10,
        text="Relative Height (0.8 = 80% from top)",
        fill="black",
        font=("TkDefaultFont", 8)
    )
    
    # 3. Width Range (measured at relative height) - only for first peak
    width_y = baseline_y + 5
    # Horizontal line at width measurement height
    manual_diagram_canvas.create_line(
        peaks[0]['x'] - peaks[0]['width'], width_y,
        peaks[0]['x'] + peaks[0]['width'], width_y,
        fill="#45B7D1", width=1, dash=(2, 2)
    )
    # Vertical lines to show width measurement
    manual_diagram_canvas.create_line(
        peaks[0]['x'] - peaks[0]['width'], width_y,
        peaks[0]['x'] - peaks[0]['width'], rel_height_y,
        fill="#45B7D1", width=1, dash=(2, 2)
    )
    manual_diagram_canvas.create_line(
        peaks[0]['x'] + peaks[0]['width'], width_y,
        peaks[0]['x'] + peaks[0]['width'], rel_height_y,
        fill="#45B7D1", width=1, dash=(2, 2)
    )
    
    # Add width range text for first peak
    manual_diagram_canvas.create_text(
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
    # Create distance slider locally
    distance_slider = tk.Scale(
        distance_container,
        from_=1,
        to=100,
        orient=tk.HORIZONTAL,
        variable=state.distance,
        length=250,
        bg=theme_manager.get_color('card_bg'),
        fg=theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=theme_manager.get_color('background')
    )
    distance_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    distance_entry = ttk.Entry(distance_container, textvariable=state.distance, width=6)
    distance_entry.pack(side=tk.LEFT, padx=5)
    
    # 2. Relative Height
    rel_height_container = ttk.Frame(params_frame)
    rel_height_container.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(rel_height_container, text="Height:").pack(side=tk.LEFT, padx=5)
    # Create height slider locally
    rel_height_slider = tk.Scale(
        rel_height_container,
        from_=0.1,
        to=1.0,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        variable=state.rel_height,
        length=250,
        bg=theme_manager.get_color('card_bg'),
        fg=theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=theme_manager.get_color('background')
    )
    rel_height_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    rel_height_entry = ttk.Entry(rel_height_container, textvariable=state.rel_height, width=6)
    rel_height_entry.pack(side=tk.LEFT, padx=5)
    
    # 3. Width Range
    width_container = ttk.Frame(params_frame)
    width_container.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(width_container, text="Width Range (ms):").pack(side=tk.LEFT, padx=5)
    width_entry = ttk.Entry(width_container, textvariable=state.width_p, width=15)
    width_entry.pack(side=tk.LEFT, padx=5)
    
    # Add tooltips for better user guidance
    if add_tooltip_callback:
        add_tooltip_callback(
            distance_slider,
            "Minimum number of points between peak centers.\n"
            "Higher values prevent detecting multiple peaks too close together."
        )
    add_tooltip_callback(
        rel_height_slider,
        "Relative height (0-1) at which peak width is measured.\n"
        "Example: 0.5 = width at half maximum height, 0.9 = width near peak top"
    )
    add_tooltip_callback(
        width_entry,
        "Enter exact peak width range in milliseconds (min,max).\n"
        "Example: '0.1,50' means only peaks between 0.1 and 50ms are kept"
    )
    
    # Prominence ratio threshold slider
    prominence_ratio_frame = ttk.Frame(manual_params_container)
    prominence_ratio_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(prominence_ratio_frame, text="Prominence Ratio Threshold:").pack(side=tk.LEFT, padx=5)
    
    prominence_ratio_slider = tk.Scale(
        prominence_ratio_frame,
        from_=0.0,
        to=1.0,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        variable=state.prominence_ratio,
        bg=theme_manager.get_color('card_bg'),
        fg=theme_manager.get_color('text'),
        troughcolor=theme_manager.get_color('background')
    )
    prominence_ratio_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Create diagram to illustrate the concept
    prominence_diagram_canvas = tk.Canvas(
        manual_params_container, 
        width=380, 
        height=120, 
        bg=theme_manager.get_color('background'),
        highlightthickness=0
    )
    prominence_diagram_canvas.pack(pady=5)
    
    # Drawing will be done when the canvas is visible
    def draw_prominence_diagram():
        canvas = prominence_diagram_canvas
        canvas.delete("all")
        
        # Colors
        text_color = theme_manager.get_color('text')
        signal_color = theme_manager.get_color('primary')
        prominence_color = "#4CAF50"  # Green
        height_color = "#FF9800"      # Orange
        subpeak_color = "#F44336"     # Red
        
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
    
    # Schedule drawing when the canvas becomes visible
    prominence_diagram_canvas.after(100, draw_prominence_diagram)
    
    # Add tooltip for prominence ratio slider
    if add_tooltip_callback:
        add_tooltip_callback(
            prominence_ratio_slider,
            "Controls the filtering of subpeaks using the prominence-to-height ratio:\n"
            "• High filtering (0.9-1.0): Very strict, keeps only the most prominent peaks\n"
            "• Medium filtering (0.8-0.9): Balanced filtering (recommended)\n"
            "• Low filtering (0.0-0.8): More permissive, keeps more peaks\n\n"
            "The default value of 0.8 works well for most measurements.\n\n"
            "Peaks with ratio < threshold are filtered out."
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
        command=callbacks['run_peak_detection'],
        style="Primary.TButton"
    )
    detect_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    view_btn = ttk.Button(
        buttons_frame, 
        text="View Peaks",
        command=callbacks['plot_filtered_peaks']
    )
    view_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    next_btn = ttk.Button(
        buttons_frame,
        text="Next Peaks",
        command=callbacks['show_next_peaks']
    )
    next_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    save_btn = ttk.Button(
        buttons_frame, 
        text="Save Results",
        command=callbacks['save_peak_info']
    )
    save_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    # Add tooltips for action buttons
    if add_tooltip_callback:
        add_tooltip_callback(
            detect_btn,
            "Run peak detection algorithm with current parameters.\n"
            "This will identify peaks based on threshold and other settings."
        )
        add_tooltip_callback(
            view_btn,
            "Display detailed view of selected individual peaks.\n"
            "This helps validate your peak detection settings."
        )
        add_tooltip_callback(
            next_btn,
            "Navigate to the next set of peaks in the visualization."
        )
        add_tooltip_callback(
            save_btn,
            "Save current peak detection results to CSV file for further analysis."
        )

    # Return the widgets that the Application needs access to (for theme updates)
    return {
        "threshold_diagram_canvas": threshold_diagram_canvas,
        "sigma_slider": sigma_slider,
        "manual_diagram_canvas": manual_diagram_canvas,
        "distance_slider": distance_slider,
        "rel_height_slider": rel_height_slider
    }

def create_peak_analysis_tab(parent_tab_control, state, theme_manager, callbacks):
    """Create the peak analysis tab"""
    peak_analysis_tab = ttk.Frame(parent_tab_control)
    parent_tab_control.add(peak_analysis_tab, text="Peak Analysis")

    add_tooltip_callback = callbacks.get('add_tooltip')
    show_tooltip_popup_callback = callbacks.get('show_tooltip_popup')

    main_container = ttk.Frame(peak_analysis_tab)
    main_container.pack(fill=tk.X, padx=5, pady=5, anchor="w")
    analysis_options_frame = ttk.LabelFrame(main_container, text="Analysis Options")
    analysis_options_frame.pack(fill=tk.X, padx=5, pady=5, anchor="w")
    button_container = ttk.Frame(analysis_options_frame)
    button_container.pack(fill=tk.X, padx=5, pady=5, anchor="w")

    # Buttons - Use callbacks
    time_resolved_btn = ttk.Button(button_container, text="Time-Resolved Analysis", command=callbacks['plot_data'])
    time_resolved_btn.pack(side=tk.LEFT, padx=5, pady=5)
    correlation_btn = ttk.Button(button_container, text="Peak Property Correlations", command=callbacks['plot_scatter'])
    correlation_btn.pack(side=tk.LEFT, padx=5, pady=5)

    # Filter Controls Frame
    filter_frame = ttk.LabelFrame(main_container, text="Peak Filtering")
    filter_frame.pack(fill=tk.X, padx=5, pady=5, anchor="w")
    filter_controls_row = ttk.Frame(filter_frame)
    filter_controls_row.pack(fill=tk.X, padx=5, pady=5, anchor="w")

    # Peak display options - Use state and callbacks
    peaks_display_frame = ttk.Frame(filter_controls_row)
    peaks_display_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
    ttk.Label(peaks_display_frame, text="Display:").pack(anchor="w", padx=5, pady=2)
    ttk.Radiobutton(
        peaks_display_frame, text="All Peaks", variable=state.show_filtered_peaks, value=False,
        command=callbacks['toggle_filtered_peaks_visibility']
    ).pack(anchor="w", padx=5, pady=2)
    filtered_peaks_radio = ttk.Radiobutton(
        peaks_display_frame, text="Show Filtered Peaks", variable=state.show_filtered_peaks, value=True,
        command=callbacks['toggle_filtered_peaks_visibility']
    )
    filtered_peaks_radio.pack(anchor="w", padx=5, pady=2)
    indicator_frame = ttk.Frame(peaks_display_frame)
    indicator_frame.pack(fill=tk.X, padx=5, pady=2)

    # Prominence ratio controls - Use state, theme_manager, callbacks
    prominence_frame = ttk.Frame(filter_controls_row)
    prominence_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=2)
    label_row = ttk.Frame(prominence_frame)
    label_row.pack(fill=tk.X, padx=0, pady=0)
    ttk.Label(label_row, text="Prominence Ratio:").pack(side=tk.LEFT, padx=(0,2), pady=0)
    info_button = ttk.Button(
        label_row, text="?", width=2,
        command=lambda: show_tooltip_popup_callback("Prominence Ratio", "Controls the filtering...") if show_tooltip_popup_callback else None
    )
    info_button.pack(side=tk.LEFT, padx=(0,5), pady=0)
    controls_row = ttk.Frame(prominence_frame)
    controls_row.pack(fill=tk.X, padx=0, pady=0)
    prominence_ratio_slider = tk.Scale(
        controls_row, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=state.prominence_ratio, length=140,
        bg=theme_manager.get_color('card_bg'), fg=theme_manager.get_color('text'), troughcolor=theme_manager.get_color('background')
    )
    prominence_ratio_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5), pady=0)
    ratio_entry = ttk.Entry(controls_row, textvariable=state.prominence_ratio, width=4)
    ratio_entry.pack(side=tk.LEFT, padx=(0,5), pady=0)
    apply_feedback_row = ttk.Frame(prominence_frame)
    apply_feedback_row.pack(fill=tk.X, padx=0, pady=(5,0), anchor="s")
    apply_button = ttk.Button(apply_feedback_row, text="Apply", command=callbacks['on_apply_prominence_ratio'])
    apply_button.pack(side=tk.LEFT, padx=(5,0), pady=0)
    # Feedback label needs to be updated via callback/return value
    filtered_peaks_feedback = ttk.Label(apply_feedback_row, text="", foreground="blue")
    filtered_peaks_feedback.pack(side=tk.RIGHT, padx=(0,10), pady=0)

    if add_tooltip_callback:
        add_tooltip_callback(apply_button, "Apply the current prominence ratio...")
        add_tooltip_callback(filtered_peaks_radio, "When enabled, peaks that would be...")
        add_tooltip_callback(prominence_ratio_slider, "Controls the filtering of subpeaks...")
        add_tooltip_callback(time_resolved_btn, "Display peak properties changes...")
        add_tooltip_callback(correlation_btn, "Display correlation plots...")

    # Throughput Interval Control - Use state
    interval_frame = ttk.Frame(analysis_options_frame)
    interval_frame.pack(fill=tk.X, padx=5, pady=2, anchor="w")
    ttk.Label(interval_frame, text="Throughput Interval (s):").pack(side=tk.LEFT, padx=(0,5))
    interval_entry = ttk.Entry(interval_frame, textvariable=state.throughput_interval, width=6)
    interval_entry.pack(side=tk.LEFT, padx=(0,5))
    interval_slider = tk.Scale(
        interval_frame, from_=1, to=100, orient=tk.HORIZONTAL, resolution=1,
        variable=state.throughput_interval, length=180, showvalue=False
    )
    interval_slider.pack(side=tk.LEFT, padx=(0,5))
    if add_tooltip_callback:
        tooltip_text = "Set the time window (in seconds) for throughput calculation..."
        add_tooltip_callback(interval_entry, tooltip_text)
        add_tooltip_callback(interval_slider, tooltip_text)

def create_double_peak_analysis_tab(parent_tab_control, state, theme_manager, callbacks):
    """Create the double peak analysis tab"""
    double_peak_tab = ttk.Frame(parent_tab_control)
    parent_tab_control.add(double_peak_tab, text="Double Peak Analysis")

    add_tooltip_callback = callbacks.get('add_tooltip')

    param_frame = ttk.LabelFrame(double_peak_tab, text="Double Peak Detection Parameters")
    param_frame.pack(fill=tk.X, padx=5, pady=5)
    distance_frame = ttk.LabelFrame(param_frame, text="Peak Distance Range (ms)")
    distance_frame.pack(fill=tk.X, padx=5, pady=5)

    # Use state vars
    min_distance_ms = tk.DoubleVar(value=state.double_peak_min_distance.get() * 1000)
    max_distance_ms = tk.DoubleVar(value=state.double_peak_max_distance.get() * 1000)

    # Update functions use state
    def update_app_distance_values():
        state.double_peak_min_distance.set(min_distance_ms.get()/1000)
        state.double_peak_max_distance.set(max_distance_ms.get()/1000)

    def sync_min_slider_to_entry(*args): pass # Simplified - assumes validation elsewhere
    def sync_max_slider_to_entry(*args): pass # Simplified - assumes validation elsewhere

    def update_min_entry(val):
        min_entry.delete(0, tk.END)
        min_entry.insert(0, f"{float(val):.1f}")
        min_distance_ms.set(float(val))
        state.double_peak_min_distance.set(float(val)/1000)

    def update_max_entry(val):
        max_entry.delete(0, tk.END)
        max_entry.insert(0, f"{float(val):.1f}")
        max_distance_ms.set(float(val))
        state.double_peak_max_distance.set(float(val)/1000)

    # Min distance widgets
    min_slider_frame = ttk.Frame(distance_frame); min_slider_frame.pack(fill=tk.X, padx=5, pady=2)
    ttk.Label(min_slider_frame, text="Min:").pack(side=tk.LEFT, padx=5)
    min_slider = ttk.Scale(min_slider_frame, from_=0.1, to=25.0, variable=min_distance_ms, orient=tk.HORIZONTAL, command=update_min_entry)
    min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    min_entry = ttk.Entry(min_slider_frame, width=6); min_entry.pack(side=tk.LEFT, padx=5); min_entry.insert(0, f"{min_distance_ms.get():.1f}")
    # min_entry.bind("<Return>", sync_min_slider_to_entry) # Binding removed for simplicity
    # min_entry.bind("<FocusOut>", sync_min_slider_to_entry)
    ttk.Label(min_slider_frame, text="ms").pack(side=tk.LEFT)

    # Max distance widgets
    max_slider_frame = ttk.Frame(distance_frame); max_slider_frame.pack(fill=tk.X, padx=5, pady=2)
    ttk.Label(max_slider_frame, text="Max:").pack(side=tk.LEFT, padx=5)
    max_slider = ttk.Scale(max_slider_frame, from_=1.0, to=50.0, variable=max_distance_ms, orient=tk.HORIZONTAL, command=update_max_entry)
    max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    max_entry = ttk.Entry(max_slider_frame, width=6); max_entry.pack(side=tk.LEFT, padx=5); max_entry.insert(0, f"{max_distance_ms.get():.1f}")
    # max_entry.bind("<Return>", sync_max_slider_to_entry)
    # max_entry.bind("<FocusOut>", sync_max_slider_to_entry)
    ttk.Label(max_slider_frame, text="ms").pack(side=tk.LEFT)

    # Amplitude ratio - Use state
    amp_frame = ttk.LabelFrame(param_frame, text="Amplitude Ratio"); amp_frame.pack(fill=tk.X, padx=5, pady=5)
    ttk.Label(amp_frame, text="Range:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    min_amp_entry = ttk.Entry(amp_frame, textvariable=state.double_peak_min_amp_ratio, width=8); min_amp_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    ttk.Label(amp_frame, text="to").grid(row=0, column=2, sticky="ew", padx=2, pady=2)
    max_amp_entry = ttk.Entry(amp_frame, textvariable=state.double_peak_max_amp_ratio, width=8); max_amp_entry.grid(row=0, column=3, sticky="ew", padx=5, pady=2)
    amp_hist_frame = ttk.Frame(amp_frame); amp_hist_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=5, pady=(2, 5))
    # Histogram canvas - Created locally, will be returned
    amp_hist_fig = Figure(figsize=(2.5, 1.0), dpi=100); amp_hist_fig.set_tight_layout(True)
    amp_hist_canvas = FigureCanvasTkAgg(amp_hist_fig, amp_hist_frame)
    amp_hist_ax = amp_hist_fig.add_subplot(111); amp_hist_ax.set_xlim(0, 5); amp_hist_ax.set_ylim(0, 1); amp_hist_ax.set_xticks([0, 1, 2, 3, 4, 5]); amp_hist_ax.set_yticks([]); amp_hist_ax.grid(True, alpha=0.3); amp_hist_ax.tick_params(axis='both', which='major', labelsize=6)
    amp_hist_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.95)
    amp_hist_canvas.draw(); amp_hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    amp_frame.columnconfigure(1, weight=1); amp_frame.columnconfigure(3, weight=1); amp_frame.rowconfigure(1, weight=1)

    # Width ratio - Use state
    width_frame = ttk.LabelFrame(param_frame, text="Width Ratio"); width_frame.pack(fill=tk.X, padx=5, pady=5)
    ttk.Label(width_frame, text="Range:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    min_width_entry = ttk.Entry(width_frame, textvariable=state.double_peak_min_width_ratio, width=8); min_width_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
    ttk.Label(width_frame, text="to").grid(row=0, column=2, sticky="ew", padx=2, pady=2)
    max_width_entry = ttk.Entry(width_frame, textvariable=state.double_peak_max_width_ratio, width=8); max_width_entry.grid(row=0, column=3, sticky="ew", padx=5, pady=2)
    width_hist_frame = ttk.Frame(width_frame); width_hist_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=5, pady=(2, 5))
    # Histogram canvas - Created locally, will be returned
    width_hist_fig = Figure(figsize=(2.5, 1.0), dpi=100); width_hist_fig.set_tight_layout(True)
    width_hist_canvas = FigureCanvasTkAgg(width_hist_fig, width_hist_frame)
    width_hist_ax = width_hist_fig.add_subplot(111); width_hist_ax.set_xlim(0, 5); width_hist_ax.set_ylim(0, 1); width_hist_ax.set_xticks([0, 1, 2, 3, 4, 5]); width_hist_ax.set_yticks([]); width_hist_ax.grid(True, alpha=0.3); width_hist_ax.tick_params(axis='both', which='major', labelsize=6)
    width_hist_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.25, top=0.95)
    width_hist_canvas.draw(); width_hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
    width_frame.columnconfigure(1, weight=1); width_frame.columnconfigure(3, weight=1); width_frame.rowconfigure(1, weight=1)

    # Explanation text
    explanation_text = """Double peak analysis identifies pairs of peaks within a specified distance range.

It filters these pairs based on amplitude and width ratios to isolate true double peaks.

• Distance Range: Time between the centers of the two peaks (ms).
• Amplitude Ratio: Ratio of the second peak's height to the first peak's height.
• Width Ratio: Ratio of the second peak's width to the first peak's width.

Histograms show distributions of these ratios for detected pairs before filtering."""
    explanation = ttk.Label(param_frame, text=explanation_text, wraplength=380, justify=tk.LEFT, padding=(5, 5))
    explanation.pack(fill=tk.X, padx=5, pady=5)

    # Action buttons - Use callbacks
    action_frame = ttk.LabelFrame(double_peak_tab, text="Actions")
    action_frame.pack(fill=tk.X, padx=5, pady=5)
    button_frame = ttk.Frame(action_frame); button_frame.pack(fill=tk.X, padx=5, pady=5)
    analyze_btn = ttk.Button(button_frame, text="Analyze Double Peaks", command=callbacks['analyze_double_peaks'], style="Primary.TButton")
    analyze_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    show_grid_btn = ttk.Button(button_frame, text="Show Grid View", command=callbacks['show_double_peaks_grid'])
    show_grid_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    export_btn = ttk.Button(button_frame, text="Export Double Peak Data", command=callbacks['save_double_peak_info'])
    export_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    nav_frame = ttk.Frame(action_frame); nav_frame.pack(fill=tk.X, padx=5, pady=5)
    prev_btn = ttk.Button(nav_frame, text="Previous Page", command=callbacks['show_prev_double_peaks_page'])
    prev_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    next_btn = ttk.Button(nav_frame, text="Next Page", command=callbacks['show_next_double_peaks_page'])
    next_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    if add_tooltip_callback:
        add_tooltip_callback(analyze_btn, "Update visualization and detect double peaks based on current parameters")
        add_tooltip_callback(show_grid_btn, "Show grid view of detected double peaks")
        add_tooltip_callback(export_btn, "Save double peak information (distances, ratios, etc.) to a file")
        add_tooltip_callback(prev_btn, "Show previous page of double peaks in the grid view")
        add_tooltip_callback(next_btn, "Show next page of double peaks in the grid view")

    # Return widgets needed by Application (for theme updates)
    return {
        "amp_hist_canvas": amp_hist_canvas,
        "amp_hist_ax": amp_hist_ax,
        "width_hist_canvas": width_hist_canvas,
        "width_hist_ax": width_hist_ax
    }

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