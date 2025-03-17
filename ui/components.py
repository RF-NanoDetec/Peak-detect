"""
UI Components for Peak Analysis Tool

This module contains functions to create the main UI components used in the application.
"""

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config.settings import Config

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

    # Signal Filtering Frame
    filtering_frame = ttk.LabelFrame(preprocessing_tab, text="Signal Filtering")
    filtering_frame.pack(fill=tk.X, padx=5, pady=5)

    # Cutoff Frequency frame with auto-calculate button
    cutoff_frame = ttk.Frame(filtering_frame)
    cutoff_frame.pack(fill=tk.X, padx=5, pady=2)

    ttk.Label(cutoff_frame, text="Cutoff Frequency (Hz)").pack(side=tk.LEFT)
    cutoff_entry = ttk.Entry(cutoff_frame, textvariable=app.cutoff_value, width=10)
    cutoff_entry.pack(side=tk.LEFT, padx=5)

    auto_cutoff_button = ttk.Button(
        cutoff_frame, 
        text="Auto Calculate", 
        command=app.calculate_auto_cutoff_frequency
    )
    auto_cutoff_button.pack(side=tk.LEFT, padx=5)

    # Parameters for auto calculation
    auto_params_frame = ttk.LabelFrame(filtering_frame, text="Auto Calculation Parameters")
    auto_params_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(auto_params_frame, text="Biggest Peaks").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    big_counts_entry = ttk.Entry(auto_params_frame, textvariable=app.big_counts)
    big_counts_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

    ttk.Label(auto_params_frame, text="Normalization Factor").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    norm_entry = ttk.Entry(
        auto_params_frame, 
        textvariable=app.normalization_factor,
        validate='key',
        validatecommand=(app.register(app.validate_float), '%P')
    )
    norm_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

    # Action Buttons - Reordered
    action_frame = ttk.Frame(preprocessing_tab)
    action_frame.pack(fill=tk.X, padx=5, pady=5)

    ttk.Button(
        action_frame,
        text="View Raw Data",  # First button
        command=app.plot_raw_data
    ).pack(side=tk.LEFT, padx=5)

    ttk.Button(
        action_frame,
        text="Filter",  # Second button
        command=app.start_analysis
    ).pack(side=tk.LEFT, padx=5)

    # Add tooltips
    app.add_tooltip(
        cutoff_entry,
        "Frequency cutoff for the Butterworth low-pass filter (Hz)\nSet to 0 for automatic calculation"
    )

    app.add_tooltip(
        auto_cutoff_button,
        "Calculate optimal cutoff frequency based on peak widths"
    )

    app.add_tooltip(
        big_counts_entry,
        "Threshold for identifying largest peaks\nUsed for automatic cutoff calculation"
    )

    app.add_tooltip(
        norm_entry,
        "Factor for normalizing signal amplitude\nTypically between 0.1 and 10"
    )

    # Configure grid weights
    auto_params_frame.columnconfigure(1, weight=1)

def create_peak_detection_tab(app, tab_control):
    """Create the peak detection tab"""
    peak_detection_tab = ttk.Frame(tab_control)
    tab_control.add(peak_detection_tab, text="Peak Detection")

    # Peak Parameters Frame
    peak_params_frame = ttk.LabelFrame(peak_detection_tab, text="Peak Parameters")
    peak_params_frame.pack(fill=tk.X, padx=5, pady=5)

    # Threshold frame with auto-calculate button
    row = 0
    threshold_frame = ttk.Frame(peak_params_frame)
    threshold_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=2)

    ttk.Label(threshold_frame, text="Counts Threshold").pack(side=tk.LEFT)
    threshold_entry = ttk.Entry(threshold_frame, textvariable=app.height_lim, width=10)
    threshold_entry.pack(side=tk.LEFT, padx=5)

    auto_calc_button = ttk.Button(
        threshold_frame, 
        text="Auto Calculate", 
        command=app.calculate_auto_threshold
    )
    auto_calc_button.pack(side=tk.LEFT, padx=5)

    # Add tooltip for Auto Calculate button
    app.add_tooltip(
        auto_calc_button,
        "Automatically calculate optimal threshold based on 5σ (sigma) of the filtered signal"
    )

    # Other peak parameters
    row += 1
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
    peak_detection_frame = ttk.Frame(peak_detection_tab)
    peak_detection_frame.pack(fill=tk.X, padx=5, pady=10)

    ttk.Button(
        peak_detection_frame, 
        text="Detect Peaks",
        command=app.run_peak_detection
    ).pack(side=tk.LEFT, padx=5)

    ttk.Button(
        peak_detection_frame, 
        text="View Peaks",
        command=app.plot_filtered_peaks
    ).pack(side=tk.LEFT, padx=5)

    ttk.Button(
        peak_detection_frame,
        text="Next Peaks",
        command=app.show_next_peaks
    ).pack(side=tk.LEFT, padx=5)

    ttk.Button(
        peak_detection_frame, 
        text="Quick Save Results",
        command=app.save_peak_information_to_csv
    ).pack(side=tk.LEFT, padx=5)

    # Add tooltips
    app.add_tooltip(
        peak_detection_frame.winfo_children()[-2],  # Next Peaks button
        "Show next set of individual peaks"
    )

    # Configure grid weights
    peak_params_frame.columnconfigure(1, weight=1)

    # Add tooltips for better user guidance
    app.add_tooltip(
        threshold_entry,
        "Minimum height threshold for peak detection"
    )
    app.add_tooltip(
        distance_entry,
        "Minimum number of points between peaks"
    )
    app.add_tooltip(
        rel_height_entry,
        "Relative height from peak maximum for width calculation (0-1)"
    )
    app.add_tooltip(
        width_entry,
        "Expected peak width range in milliseconds (min,max)"
    )
    app.add_tooltip(
        peak_detection_frame.winfo_children()[0],  # Detect Peaks button
        "Run peak detection algorithm with current parameters"
    )
    app.add_tooltip(
        peak_detection_frame.winfo_children()[1],  # View Peaks button
        "Display detailed view of selected individual peaks"
    )
    app.add_tooltip(
        peak_detection_frame.winfo_children()[2],  # Quick Save Results button
        "Save current peak detection results to CSV file"
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