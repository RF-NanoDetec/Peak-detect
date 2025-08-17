"""
UI Components for Peak Analysis Tool

This module contains functions to create the main UI components used in the application.
"""

import os
import sys
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
    file_menu.add_command(label="Open File", command=app.browse_file, accelerator="Ctrl+O")
    file_menu.add_command(label="Export Results", command=app.save_peak_information_to_csv)
    file_menu.add_separator()
    file_menu.add_command(label="Export Current Plot", command=app.export_plot, accelerator="Ctrl+E")
    file_menu.add_command(label="Take Screenshot", command=app.take_screenshot)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=app.quit, accelerator="Ctrl+Q")
    
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
    # Add reset layout option
    view_menu.add_command(label="Reset Layout", command=app.reset_layout)
    
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
    
    # Set a fixed width for the control panel
    control_frame.grid_propagate(False)  # Prevent the frame from resizing to its children
    control_frame.configure(width=400)   # Set a fixed width
    
    # Add status indicator at the top
    status_frame = ttk.Frame(control_frame, style='Card.TFrame')
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
    
    # Configure main_frame to give more weight to the preview column
    main_frame.columnconfigure(1, weight=1)  # Preview frame gets all extra space
    
    # Tab Control for Multiple Plots on the right
    app.plot_tab_control = ttk.Notebook(preview_frame)
    app.plot_tab_control.grid(row=0, column=0, sticky="nsew")

    # Create an empty frame with fixed size instead of blank image
    app.blank_tab = ttk.Frame(app.plot_tab_control, width=800, height=600, style='Card.TFrame')
    app.plot_tab_control.add(app.blank_tab, text="Welcome")
    
    # Create enhanced welcome screen with modern layout
    create_welcome_screen(app, app.blank_tab)
    
    # Prevent the blank tab from shrinking
    app.blank_tab.pack_propagate(False)

    # Functional Bar under plot tabs
    functional_bar = ttk.Frame(preview_frame, style='Toolbar.TFrame')
    functional_bar.grid(row=1, column=0, sticky="ew", pady=10)

    ttk.Button(functional_bar, 
              text="Export Plot", 
              command=app.export_plot,
              style='Tool.TButton'
    ).grid(row=0, column=0, padx=5, pady=5)
    
    # Add scale toggle button
    scale_toggle_btn = ttk.Button(
        functional_bar,
        text="Toggle Scale (Log/Linear)",
        command=app.toggle_scale_mode,
        style='Tool.TButton'
    )
    scale_toggle_btn.grid(row=0, column=1, padx=5, pady=5)
    
    # Add tooltip for scale toggle button
    app.add_tooltip(
        scale_toggle_btn,
        "Toggle between logarithmic and linear scales for peak analysis plots"
    )
    
    return preview_frame

def create_welcome_screen(app, parent):
    """
    Create an enhanced welcome screen with modern design elements.
    
    Features:
    - App logo and banner
    - Welcome message with app version
    - Quick start buttons
    - Visual peak detection illustration
    """
    from PIL import Image, ImageTk
    import os
    
    # Main container for welcome screen with theme-aware background
    welcome_container = ttk.Frame(parent)
    welcome_container.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.9, relheight=0.9)
    
    # Configure grid for welcome container
    welcome_container.columnconfigure(0, weight=1)
    welcome_container.rowconfigure(0, weight=0)  # Header
    welcome_container.rowconfigure(1, weight=0)  # Title
    welcome_container.rowconfigure(2, weight=1)  # Content
    welcome_container.rowconfigure(3, weight=0)  # Footer
    
    # ===== Header with Logo =====
    header_frame = ttk.Frame(welcome_container)
    header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    
    # Try to load the app icon
    try:
        # Determine the correct path
        icon_path = os.path.join("resources", "images", "icon.ico")
        # For PyInstaller bundle support
        if not os.path.exists(icon_path) and hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, "resources", "images", "icon.ico")
            
        if os.path.exists(icon_path):
            icon_img = Image.open(icon_path)
            icon_img = icon_img.resize((64, 64), Image.Resampling.LANCZOS)
            icon_photo = ImageTk.PhotoImage(icon_img)
            
            # Store reference to prevent garbage collection
            app.welcome_icon = icon_photo
            
            # Display the icon
            icon_label = ttk.Label(header_frame, image=icon_photo, background=app.theme_manager.get_color('background'))
            icon_label.pack(side=tk.LEFT, padx=10)
    except Exception as e:
        print(f"Could not load icon image: {e}")
    
    # App name and version
    from config import APP_VERSION
    title_label = ttk.Label(
        header_frame,
        text="Peak Analysis Tool",
        style="Display.TLabel"
    )
    title_label.pack(side=tk.LEFT, padx=10)
    
    version_label = ttk.Label(
        header_frame,
        text=f"v{APP_VERSION}",
        style="Small.TLabel"
    )
    version_label.pack(side=tk.LEFT)
    
    # ===== Welcome Message =====
    welcome_text = ttk.Label(
        welcome_container,
        text="Welcome to your scientific peak analysis workbench",
        style="Heading.TLabel",
        anchor="center"
    )
    welcome_text.grid(row=1, column=0, sticky="ew", pady=(0, 20))
    
    # ===== Main Content Area =====
    content_frame = ttk.Frame(welcome_container)
    content_frame.grid(row=2, column=0, sticky="nsew")
    content_frame.columnconfigure(0, weight=1)
    content_frame.columnconfigure(1, weight=1)
    
    # Left side: Quick start buttons
    quick_start_frame = ttk.LabelFrame(content_frame, text="Quick Start")
    quick_start_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    # Add quick start buttons with icons (if available)
    load_button = ttk.Button(
        quick_start_frame,
        text="   Load Data File",
        command=app.browse_file
    )
    load_button.pack(fill=tk.X, padx=20, pady=10)
    
    analyze_button = ttk.Button(
        quick_start_frame,
        text="   Start Analysis",
        command=app.start_analysis,
        state="disabled"  # Initially disabled until data is loaded
    )
    analyze_button.pack(fill=tk.X, padx=20, pady=10)
    
    docs_button = ttk.Button(
        quick_start_frame,
        text="   View Documentation",
        command=app.show_documentation
    )
    docs_button.pack(fill=tk.X, padx=20, pady=10)
    
    # Right side: Features overview with image
    features_frame = ttk.LabelFrame(content_frame, text="Key Features")
    features_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
    
    # Create a container frame to hold both image and text
    features_content = ttk.Frame(features_frame)
    features_content.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Try to load the example image
    image_loaded = False
    try:
        # Determine the correct path for an example image
        img_path = os.path.join("resources", "images", "startim.png")
        # For PyInstaller bundle support
        if not os.path.exists(img_path) and hasattr(sys, '_MEIPASS'):
            img_path = os.path.join(sys._MEIPASS, "resources", "images", "startim.png")
            
        if os.path.exists(img_path):
            # Load and resize image to fit the frame
            example_img = Image.open(img_path)
            # Resize to fit the frame (adjust dimensions as needed)
            max_width = 400
            max_height = 250
            img_ratio = min(max_width/example_img.width, max_height/example_img.height)
            new_size = (int(example_img.width * img_ratio), int(example_img.height * img_ratio))
            example_img = example_img.resize(new_size, Image.Resampling.LANCZOS)
            
            example_photo = ImageTk.PhotoImage(example_img)
            
            # Store reference to prevent garbage collection
            app.welcome_example = example_photo
            
            # Display the image
            img_label = ttk.Label(
                features_content, 
                image=example_photo,
                background=app.theme_manager.get_color('background')
            )
            img_label.pack(padx=10, pady=10)
            image_loaded = True
    except Exception as e:
        print(f"Could not load example image: {e}")
    
    # Always display features text, either with or without image
    features_text = """
• Precise Peak Detection with advanced filtering algorithms
• Time-resolved analysis & peak pattern identification
• Multi-file batch processing with automatic sequencing
• Customizable detection parameters for different signal types
• Statistical analysis (distribution, correlation, properties)
• Double peak detection for complex signal patterns
• Interactive visualization with dark/light theme support
• CSV export with complete peak metadata
    """
    
    features_label = ttk.Label(
        features_content,
        text=features_text,
        style="Body.TLabel",
        justify=tk.LEFT
    )
    features_label.pack(padx=20, pady=10, anchor="w")
    
    # ===== Footer with Tip =====
    tip_frame = ttk.Frame(welcome_container, style="Card.TFrame")
    tip_frame.grid(row=3, column=0, sticky="ew", pady=(20, 0))
    
    tip_icon_label = ttk.Label(
        tip_frame,
        text="💡",
        style="Heading.TLabel"
    )
    tip_icon_label.pack(side=tk.LEFT, padx=(10, 5), pady=10)
    
    tip_text = ttk.Label(
        tip_frame,
        text="Tip: Use the control panel on the left to configure parameters for each analysis step.",
        style="Body.TLabel",
        wraplength=600
    )
    tip_text.pack(side=tk.LEFT, padx=5, pady=10)
    
    # Create function to update analyze button state
    def update_analyze_button_state(*args):
        if hasattr(app, 'data') and app.data is not None:
            analyze_button.config(state="normal")
        else:
            analyze_button.config(state="disabled")
    
    # Store the function for later use
    app.update_welcome_analyze_button = update_analyze_button_state

def create_data_loading_tab(app, tab_control):
    """Create the data loading tab"""
    data_loading_tab = ttk.Frame(tab_control)
    tab_control.add(data_loading_tab, text="Load Data")

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
    peak_mode_container.pack(padx=5, pady=5, anchor=tk.W)
    
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
    
    time_res_label = ttk.Label(app.highlight_frame, text="Dwell Time (ms):", style="Heading.TLabel")
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
    
    units_label = ttk.Label(app.highlight_frame, text="milliseconds", style="Body.TLabel")
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
        ("Sample Number:", app.protocol_sample_number),
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

    # Add some padding to the main tab content area
    content_frame = ttk.Frame(preprocessing_tab, padding="10 10 10 10")
    content_frame.pack(expand=True, fill=tk.BOTH)
    
    # Make content_frame expand with window
    content_frame.columnconfigure(0, weight=1) 

    mode_frame = ttk.LabelFrame(content_frame, text="Signal Processing Mode")
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
    
    # Create a more compact and modern processing mode selector
    mode_selector = ttk.Frame(mode_frame)
    mode_selector.pack(pady=5, anchor=tk.W)
    
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
        background=app.theme_manager.get_color('primary'),
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
        background=app.theme_manager.get_color('text'),
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
        bg=app.theme_manager.get_color('background'),
        highlightthickness=0
    )
    app.preprocessing_comparison_canvas.pack(pady=5, anchor=tk.W)
    
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
        app.filter_enabled.set(is_filtered) # Ensure the underlying variable is set
        
        if hasattr(app, 'process_btn') and app.process_btn:
            if is_filtered:
                app.process_btn.configure(text="Apply Filtering")
            else:
                app.process_btn.configure(text="Process Raw Data")

        interactive_widgets_to_manage = [
            app.filter_type_dropdown,
            app.cutoff_freq_entry, 
            # auto_cutoff_button is not stored on app, handle via parent frame iteration
            app.filter_order_entry,
            app.savgol_window_entry,
            app.savgol_polyorder_entry
        ]
        # Also find auto_cutoff_button if it exists in butterworth_controls_frame
        # This assumes auto_cutoff_button was defined and is a child of butterworth_controls_frame

        if is_filtered:
            filtering_frame.pack(fill=tk.X, padx=5, pady=5)
            app.filter_type_dropdown.config(state="readonly") # Combobox always readonly when active
            toggle_filter_controls() # This will show/hide Butterworth or SavGol frames and set states of their children
            
            # toggle_filter_controls now handles state of children in active frame.
            # We just need to ensure that if filtering_frame itself is shown,
            # the initially selected filter type's controls are correctly enabled.
            # This is already handled by toggle_filter_controls being called.

        else:
            filtering_frame.pack_forget() # Hide the main filtering frame
            app.filter_type_dropdown.config(state=tk.DISABLED)
            # Disable all specific filter controls since the main filtering is off
            for widget in butterworth_controls_frame.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Combobox, tk.Scale)):
                    widget.config(state=tk.DISABLED)
            for widget in savgol_controls_frame.winfo_children():
                if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Combobox, tk.Scale)):
                    widget.config(state=tk.DISABLED)
    
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
    filtering_frame = ttk.LabelFrame(content_frame, text="Filtering Parameters")
    filtering_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
    filtering_frame.columnconfigure(1, weight=1) # Allow entry fields to expand

    # Initialize filter type variable (load from prefs if available)
    if not hasattr(app, 'filter_type_var'):
        initial_filter = app.prefs.get('filter_type', Config.DEFAULT_FILTER_TYPE) if hasattr(app, 'prefs') else Config.DEFAULT_FILTER_TYPE
        app.filter_type_var = tk.StringVar(value=initial_filter)
    
    # Initialize Savitzky-Golay specific variables
    if not hasattr(app, 'savgol_window_var'):
        initial_win = str(app.prefs.get('savgol_window', Config.DEFAULT_SAVGOL_WINDOW_LENGTH)) if hasattr(app, 'prefs') else str(Config.DEFAULT_SAVGOL_WINDOW_LENGTH)
        app.savgol_window_var = tk.StringVar(value=initial_win) 
    if not hasattr(app, 'savgol_polyorder_var'):
        initial_poly = str(app.prefs.get('savgol_polyorder', Config.DEFAULT_SAVGOL_POLYORDER)) if hasattr(app, 'prefs') else str(Config.DEFAULT_SAVGOL_POLYORDER)
        app.savgol_polyorder_var = tk.StringVar(value=initial_poly) 
    if not hasattr(app, 'butter_order_var'):
        initial_order = str(app.prefs.get('butter_order', Config.DEFAULT_BUTTER_FILTER_ORDER)) if hasattr(app, 'prefs') else str(Config.DEFAULT_BUTTER_FILTER_ORDER)
        app.butter_order_var = tk.StringVar(value=initial_order)


    # --- Filter Type Selection ---
    filter_type_label = ttk.Label(filtering_frame, text="Filter Type:")
    filter_type_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    
    filter_options = ["butterworth", "savgol"]
    app.filter_type_dropdown = ttk.Combobox(
        filtering_frame, 
        textvariable=app.filter_type_var, 
        values=filter_options, 
        state="readonly",
        width=18 # Adjusted width
    )
    app.filter_type_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    app.add_tooltip(app.filter_type_dropdown, "Select the type of filter to apply.\nButterworth: Good general-purpose low-pass filter.\nSavitzky-Golay: Good for smoothing while preserving peak shapes.")

    # Frame for Butterworth specific controls
    butterworth_controls_frame = ttk.Frame(filtering_frame)
    butterworth_controls_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=(0,5))
    butterworth_controls_frame.columnconfigure(1, weight=1)

    # Frame for Savitzky-Golay specific controls
    savgol_controls_frame = ttk.Frame(filtering_frame)
    savgol_controls_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=(0,5))
    savgol_controls_frame.columnconfigure(1, weight=1)


    # --- Butterworth Specific Controls ---
    # Ensure app.filter_cutoff_freq exists (should be initialized in main.py or similar)
    if not hasattr(app, 'filter_cutoff_freq'):
        app.filter_cutoff_freq = tk.StringVar(value=str(Config.DEFAULT_FILTER_CUTOFF_FREQ))
    
    cutoff_label = ttk.Label(butterworth_controls_frame, text="Cutoff Freq. (Hz):")
    cutoff_label.grid(row=0, column=0, padx=0, pady=5, sticky=tk.W)
    app.cutoff_freq_entry = ttk.Entry(butterworth_controls_frame, textvariable=app.filter_cutoff_freq, width=20)
    app.cutoff_freq_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    app.add_tooltip(app.cutoff_freq_entry, "Cutoff frequency for the Butterworth low-pass filter in Hz.\nLeave empty or set to 0 for auto-calculation based on peak widths.")

    # Auto-calculate cutoff button (Butterworth)
    # Ensure app.calculate_auto_cutoff_frequency is defined in main app class
    auto_cutoff_button = ttk.Button(butterworth_controls_frame, text="Auto", command=app.calculate_auto_cutoff_frequency if hasattr(app, 'calculate_auto_cutoff_frequency') else lambda: print("Auto-cutoff function not set"), width=5)
    auto_cutoff_button.grid(row=0, column=2, padx=5, pady=5, sticky=tk.E)
    app.add_tooltip(auto_cutoff_button, "Automatically estimate a suitable cutoff frequency based on average peak width.")

    order_label = ttk.Label(butterworth_controls_frame, text="Filter Order:")
    order_label.grid(row=1, column=0, padx=0, pady=5, sticky=tk.W)
    app.filter_order_entry = ttk.Entry(butterworth_controls_frame, textvariable=app.butter_order_var, width=20)
    app.filter_order_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
    app.add_tooltip(app.filter_order_entry, "Order of the Butterworth filter (e.g., 2, 3, 4). Higher orders have a steeper rolloff.")
    

    # --- Savitzky-Golay Specific Controls ---
    savgol_window_label = ttk.Label(savgol_controls_frame, text="Window Length:")
    savgol_window_label.grid(row=0, column=0, padx=0, pady=5, sticky=tk.W)
    app.savgol_window_entry = ttk.Entry(savgol_controls_frame, textvariable=app.savgol_window_var, width=20)
    app.savgol_window_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
    app.add_tooltip(app.savgol_window_entry, "Window length for Savitzky-Golay filter (odd integer, e.g., 5, 11, 21).\nLeave empty for auto-estimation based on peak widths.")

    savgol_polyorder_label = ttk.Label(savgol_controls_frame, text="Polynomial Order:")
    savgol_polyorder_label.grid(row=1, column=0, padx=0, pady=5, sticky=tk.W)
    app.savgol_polyorder_entry = ttk.Entry(savgol_controls_frame, textvariable=app.savgol_polyorder_var, width=20)
    app.savgol_polyorder_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
    app.add_tooltip(app.savgol_polyorder_entry, "Polynomial order for Savitzky-Golay filter (integer, less than window length).\nLeave empty for default (e.g., 2 or 3).")

    # Function to toggle filter controls visibility
    def toggle_filter_controls(*args): # Already defined by previous edit, ensure it's robust
        selected_filter = app.filter_type_var.get()
        
        # Determine which controls to show/enable vs hide/disable
        if selected_filter == 'butterworth':
            butterworth_controls_frame.grid()
            for child in butterworth_controls_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox, tk.Scale)):
                    child.config(state=tk.NORMAL)
            savgol_controls_frame.grid_remove()
            for child in savgol_controls_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox, tk.Scale)):
                    child.config(state=tk.DISABLED)

        elif selected_filter == 'savgol':
            butterworth_controls_frame.grid_remove()
            for child in butterworth_controls_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox, tk.Scale)):
                    child.config(state=tk.DISABLED)
            savgol_controls_frame.grid()
            for child in savgol_controls_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox, tk.Scale)):
                    child.config(state=tk.NORMAL)
        else: 
            butterworth_controls_frame.grid_remove()
            for child in butterworth_controls_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox, tk.Scale)):
                    child.config(state=tk.DISABLED)
            savgol_controls_frame.grid_remove()
            for child in savgol_controls_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox, tk.Scale)):
                    child.config(state=tk.DISABLED)

    app.filter_type_var.trace_add("write", toggle_filter_controls)

    # Help text for filtering (already added and gridded by previous commit)
    # filtering_help_text = (...)
    # filtering_help_label = ttk.Label(filtering_frame, text=filtering_help_text, style="Tooltip.TLabel")
    # filtering_help_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(10,5))

    # Initial state update is handled by the radio button's command calling update_filter_state.
    # When the tab is created, app.filter_enabled (BooleanVar) should have its default value (e.g., True).
    # So, update_filter_state(app.filter_enabled.get()) should be called once after all related widgets are defined.
    # Or, even better, the radio buttons default state will trigger its command.
    # Let's ensure the initial call happens correctly.
    # The `variable=app.filter_enabled` in Radiobutton and its default value in main app
    # should trigger an initial call to update_filter_state if its value changes from None to True/False.
    # To be certain, an explicit call after everything is defined is safest:
    update_filter_state(app.filter_enabled.get()) 

    # Separator and Preview
    ttk.Separator(content_frame, orient="horizontal").pack(fill=tk.X, padx=10, pady=10)

    signal_processing_frame = ttk.LabelFrame(content_frame, text="Signal Processing")
    signal_processing_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

    # Main help text for signal processing
    signal_main_help_text = (
        "Signal processing filters help remove noise and enhance signal quality for better peak detection. "
        "Choose the appropriate filter type based on your data characteristics and analysis requirements."
    )
    signal_main_help_label = ttk.Label(signal_processing_frame, text=signal_main_help_text, style="Tooltip.TLabel", wraplength=380, justify=tk.LEFT)
    signal_main_help_label.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(5,5))

    # --- Collapsible detailed descriptions ---
    # Frame to hold the details and the toggle button
    details_frame = ttk.Frame(signal_processing_frame)
    details_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(0,5))
    details_frame.columnconfigure(0, weight=1)

    # Button to toggle details
    app.signal_details_visible = tk.BooleanVar(value=False)
    
    def toggle_signal_details():
        visible = app.signal_details_visible.get()
        if visible:
            butterworth_details_label.grid()
            savgol_details_label.grid()
            toggle_button.configure(text="Hide Details")
        else:
            butterworth_details_label.grid_remove()
            savgol_details_label.grid_remove()
            toggle_button.configure(text="Show Details")

    toggle_button = ttk.Button(details_frame, text="Show Details", command=lambda: [app.signal_details_visible.set(not app.signal_details_visible.get()), toggle_signal_details()])
    toggle_button.grid(row=0, column=0, sticky=tk.W, pady=(5,2))

    # Detailed text for Butterworth Filter
    butterworth_details_text = (
        "• Butterworth Filter:\n"
        "  A low-pass filter with maximally flat frequency response in the passband. It effectively removes "
        "high-frequency noise while preserving the shape of your peaks.\n"
        "  ✓ Pros: Excellent frequency response, no ripple in passband, good for general noise reduction\n"
        "  ✗ Cons: Can introduce phase distortion, may smooth sharp features slightly\n"
        "  Best for: General-purpose noise removal, signals with consistent peak shapes\n"
        "  Key Parameters: Cutoff Frequency (Hz) - frequencies above this are attenuated, Filter Order - higher orders provide steeper rolloff"
    )
    butterworth_details_label = ttk.Label(details_frame, text=butterworth_details_text, style="Tooltip.TLabel", wraplength=370, justify=tk.LEFT)
    butterworth_details_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5)
    butterworth_details_label.grid_remove()

    # Detailed text for Savitzky-Golay Filter
    savgol_details_text = (
        "• Savitzky-Golay Filter:\n"
        "  A polynomial smoothing filter that preserves peak shapes and heights better than moving averages. "
        "It fits polynomials to local data windows and uses the fitted values as filtered output.\n"
        "  ✓ Pros: Preserves peak shapes excellently, maintains peak heights, good for derivative calculations\n"
        "  ✗ Cons: Can be sensitive to outliers, requires careful parameter selection\n"
        "  Best for: Peak-rich signals where maintaining peak characteristics is crucial\n"
        "  Standard Values: Window Length=11, Polynomial Order=2 (good starting point for most data)\n"
        "  Key Parameters: Window Length (odd number) - larger windows = more smoothing, Polynomial Order - higher orders fit more complex local patterns"
    )
    savgol_details_label = ttk.Label(details_frame, text=savgol_details_text, style="Tooltip.TLabel", wraplength=370, justify=tk.LEFT)
    savgol_details_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5)
    savgol_details_label.grid_remove()
    
    # Ensure initial state of button text is correct
    toggle_signal_details()


    # Action Buttons with improved layout
    action_frame = ttk.LabelFrame(content_frame, text="Processing Actions")
    action_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

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
    app.process_btn = process_btn # Assign to app for access in update_filter_state

def create_peak_detection_tab(app, tab_control):
    """Create the peak detection tab"""
    peak_detection_tab = ttk.Frame(tab_control)
    tab_control.add(peak_detection_tab, text="Detection")  # Changed from "Peak Detection"

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
        bg=app.theme_manager.get_color('background'),
        highlightthickness=0
    )
    app.threshold_diagram_canvas.pack()
    
    # Draw a sine-like signal to represent data
    signal_color = app.theme_manager.get_color('primary')  # Use blue directly instead of theme's primary color
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
        style="Heading.TLabel"
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
        bg=app.theme_manager.get_color('background'),
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
        bg=app.theme_manager.get_color('background'),
        highlightthickness=0
    )
    app.manual_diagram_canvas.pack(fill=tk.X, padx=5, pady=5)
    
    # Draw a realistic signal with two clear peaks
    signal_color = app.theme_manager.get_color('primary')  # Use blue directly instead of theme's primary color
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
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('panel_bg')
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
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('panel_bg')
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

    # Reset to defaults button for detection params
    def reset_detection_defaults():
        try:
            app.height_lim.set(20)
            app.distance.set(5)
            app.rel_height.set(0.8)
            app.width_p.set("0.1,50")
            # Clear validation message if present
            if hasattr(app, 'validation_label'):
                app.validation_label.config(text="")
            # Sync sliders
            if hasattr(app, 'distance_slider'):
                app.distance_slider.set(app.distance.get())
            if hasattr(app, 'rel_height_slider'):
                app.rel_height_slider.set(app.rel_height.get())
        except Exception:
            pass

    reset_btn = ttk.Button(params_frame, text="Reset to Defaults", command=reset_detection_defaults)
    reset_btn.pack(anchor=tk.E, padx=5, pady=6)
    
    # Prominence ratio threshold slider
    prominence_ratio_frame = ttk.Frame(manual_params_container)
    prominence_ratio_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(prominence_ratio_frame, text="Prominence Ratio Threshold:").pack(side=tk.LEFT, padx=5)
    
    # Create variable if it doesn't exist, otherwise use the existing one
    if not hasattr(app, 'prominence_ratio'):
        app.prominence_ratio = tk.DoubleVar(value=0.8)  # Default 0.8 (80%)
    prominence_ratio_slider = tk.Scale(
        prominence_ratio_frame,
        from_=0.0,
        to=1.0,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        variable=app.prominence_ratio,
        length=140,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        troughcolor=app.theme_manager.get_color('panel_bg')
    )
    prominence_ratio_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Create diagram to illustrate the concept
    prominence_diagram_canvas = tk.Canvas(
        manual_params_container, 
        width=380, 
        height=120, 
        bg=app.theme_manager.get_color('background'),
        highlightthickness=0
    )
    prominence_diagram_canvas.pack(pady=5)
    
    # Store canvas as app attribute so it can be updated when theme changes
    app.prominence_diagram_canvas = prominence_diagram_canvas
    
    # Define the draw function as an app method so it can be called when theme changes
    def draw_prominence_diagram():
        app._draw_prominence_diagram()
    
    # Schedule drawing when the canvas becomes visible
    prominence_diagram_canvas.after(100, draw_prominence_diagram)
    
    # Add tooltip for prominence ratio slider
    app.add_tooltip(
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
    tab_control.add(peak_analysis_tab, text="Analysis")  # Changed from "Peak Analysis"

    # Create a main container for all controls
    main_container = ttk.Frame(peak_analysis_tab)
    main_container.pack(fill=tk.X, padx=5, pady=5, anchor="w")
    
    # Analysis Options Frame - placed at the top
    analysis_options_frame = ttk.LabelFrame(main_container, text="Analysis Options")
    analysis_options_frame.pack(fill=tk.X, padx=5, pady=5, anchor="w")

    # Button container for analysis buttons
    button_container = ttk.Frame(analysis_options_frame)
    button_container.pack(fill=tk.X, padx=5, pady=5, anchor="w")

    # Time-resolved analysis button (first)
    ttk.Button(
        button_container,
        text="Time-Resolved Analysis",
        command=app.plot_data
    ).pack(side=tk.LEFT, padx=5, pady=5)

    # Peak properties correlation button (second)
    ttk.Button(
        button_container,
        text="Peak Property Correlations",
        command=app.plot_scatter
    ).pack(side=tk.LEFT, padx=5, pady=5)
    
    # Filter Controls Frame - placed below the analysis options
    filter_frame = ttk.LabelFrame(main_container, text="Peak Filtering")
    filter_frame.pack(fill=tk.X, padx=5, pady=5, anchor="w")
    
    # Create top row container for filter controls
    filter_controls_row = ttk.Frame(filter_frame)
    filter_controls_row.pack(fill=tk.X, padx=5, pady=5, anchor="w")
    
    # Create variable for tracking filtered peaks visibility if it doesn't exist
    if not hasattr(app, 'show_filtered_peaks'):
        app.show_filtered_peaks = tk.BooleanVar(value=False)
    
    # Peak display options (left side of filter controls)
    peaks_display_frame = ttk.Frame(filter_controls_row)
    peaks_display_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
    
    ttk.Label(peaks_display_frame, text="Display:").pack(anchor="w", padx=5, pady=2)
    
    # Radio buttons for peak display options
    ttk.Radiobutton(
        peaks_display_frame,
        text="All Peaks",
        variable=app.show_filtered_peaks,
        value=False,
        command=app.toggle_filtered_peaks_visibility
    ).pack(anchor="w", padx=5, pady=2)
    
    filtered_peaks_radio = ttk.Radiobutton(
        peaks_display_frame,
        text="Show Filtered Peaks",
        variable=app.show_filtered_peaks,
        value=True,
        command=app.toggle_filtered_peaks_visibility
    )
    filtered_peaks_radio.pack(anchor="w", padx=5, pady=2)
    
    # Visual indicator for filtered peaks
    indicator_frame = ttk.Frame(peaks_display_frame)
    indicator_frame.pack(fill=tk.X, padx=5, pady=2)
    
    # Prominence ratio controls (right side of filter controls)
    prominence_frame = ttk.Frame(filter_controls_row) 
    prominence_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=2)
    
    # Row for label, info button, and controls
    label_row = ttk.Frame(prominence_frame)
    label_row.pack(fill=tk.X, padx=0, pady=0)
    
    # Prominence Ratio label
    ttk.Label(label_row, text="Prominence Ratio:").pack(side=tk.LEFT, padx=(0,2), pady=0)
    # Info button right next to label
    info_button = ttk.Button(
        label_row, 
        text="?", 
        width=2,
        command=lambda: app.show_tooltip_popup(
            "Prominence Ratio",
            "Controls the filtering of subpeaks using the prominence-to-height ratio:\n"
            "• High filtering (0.9-1.0): Very strict, keeps only the most prominent peaks\n"
            "• Medium filtering (0.8-0.9): Balanced filtering (recommended)\n"
            "• Low filtering (0.0-0.8): More permissive, keeps more peaks\n\n"
            "The default value of 0.8 works well for most measurements.\n\n"
            "Peaks with ratio < threshold are filtered out."
        )
    )
    info_button.pack(side=tk.LEFT, padx=(0,5), pady=0)
    
    # Controls row: slider and entry only
    controls_row = ttk.Frame(prominence_frame)
    controls_row.pack(fill=tk.X, padx=0, pady=0)
    
    prominence_ratio_slider = tk.Scale(
        controls_row,
        from_=0.0,
        to=1.0,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        variable=app.prominence_ratio,
        length=140,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        troughcolor=app.theme_manager.get_color('panel_bg')
    )
    prominence_ratio_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5), pady=0)
    
    ratio_entry = ttk.Entry(controls_row, textvariable=app.prominence_ratio, width=4)
    ratio_entry.pack(side=tk.LEFT, padx=(0,5), pady=0)
    
    # New frame for Apply button and feedback, aligned bottom row
    apply_feedback_row = ttk.Frame(prominence_frame)
    apply_feedback_row.pack(fill=tk.X, padx=0, pady=(5,0), anchor="s")
    
    apply_button = ttk.Button(
        apply_feedback_row,
        text="Apply",
        command=app.on_apply_prominence_ratio
    )
    apply_button.pack(side=tk.LEFT, padx=(5,0), pady=0)
    app.add_tooltip(apply_button, "Apply the current prominence ratio threshold to update the analysis and feedback.")
    
    # Feedback label for filtered peaks, right of Apply (bottom right)
    app.filtered_peaks_feedback = ttk.Label(apply_feedback_row, text="", foreground=app.theme_manager.get_color('primary'))
    app.filtered_peaks_feedback.pack(side=tk.RIGHT, padx=(0,10), pady=0)
    
    # Add tooltips
    app.add_tooltip(
        filtered_peaks_radio,
        "When enabled, peaks that would be filtered out are shown in light red.\n"
        "This helps visualize which peaks are being excluded by the current threshold setting."
    )
    
    app.add_tooltip(
        prominence_ratio_slider,
        "Controls the filtering of subpeaks using the prominence-to-height ratio:\n"
        "• High filtering (0.9-1.0): Very strict, keeps only the most prominent peaks\n"
        "• Medium filtering (0.8-0.9): Balanced filtering (recommended)\n"
        "• Low filtering (0.0-0.8): More permissive, keeps more peaks\n\n"
        "The default value of 0.8 works well for most measurements."
    )

    # Add tooltips for analysis buttons
    app.add_tooltip(
        button_container.winfo_children()[0],  # Time-Resolved Analysis button
        "Display peak properties changes over time and throughput analysis.\n"
        "Also applies any changes to the throughput interval setting."
    )
    app.add_tooltip(
        button_container.winfo_children()[1],  # Peak Property Correlations button
        "Display correlation plots between peak width, height, and area"
    )

    # --- Throughput Interval Control ---
    interval_frame = ttk.LabelFrame(main_container, text="Throughput Interval (s)")
    interval_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
    
    # Entry for precise value
    interval_entry = ttk.Entry(interval_frame, textvariable=app.throughput_interval, width=6)
    interval_entry.pack(side=tk.LEFT, padx=(0,5))
    
    # Slider for interval (1-100s)
    interval_slider = tk.Scale(
        interval_frame,
        from_=1,
        to=100,
        orient=tk.HORIZONTAL,
        resolution=1,
        variable=app.throughput_interval,
        length=180,
        showvalue=False,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        troughcolor=app.theme_manager.get_color('panel_bg')
    )
    interval_slider.pack(side=tk.LEFT, padx=(0,5))
    
    # Tooltip for interval
    app.add_tooltip(
        interval_entry,
        "Set the time window (in seconds) for throughput calculation.\n"
        "This controls the bin size for the 'peaks per X seconds' bar plot.\n"
        "Default is 10 seconds.\n"
        "Click 'Time Resolved Analysis' to apply changes."
    )
    app.add_tooltip(
        interval_slider,
        "Set the time window (in seconds) for throughput calculation.\n"
        "This controls the bin size for the 'peaks per X seconds' bar plot.\n"
        "Default is 10 seconds.\n"
        "Click 'Time Resolved Analysis' to apply changes."
    )
    
    # Note: Throughput interval changes are applied when clicking the Time Resolved Analysis button
    
    # Add tooltips for analysis buttons
    app.add_tooltip(
        button_container.winfo_children()[0],  # Time-Resolved Analysis button
        "Display peak properties changes over time and throughput analysis.\n"
        "Also applies any changes to the throughput interval setting."
    )

def create_double_peak_analysis_tab(app, tab_control):
    """Create the double peak analysis tab"""
    double_peak_tab = ttk.Frame(tab_control)
    tab_control.add(double_peak_tab, text="Double Peak")  # Changed from "Double Peak Analysis"
    
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
                app.min_slider.set(value)
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
                app.max_slider.set(value)
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
    app.min_slider = tk.Scale(
        min_slider_frame, 
        from_=0.1, 
        to=25.0,
        variable=min_distance_ms, 
        orient=tk.HORIZONTAL,
        command=update_min_entry,
        length=250,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background'),
        showvalue=False  # Hide the default value display
    )
    app.min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
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
    app.max_slider = tk.Scale(
        max_slider_frame, 
        from_=1.0, 
        to=50.0,
        variable=max_distance_ms, 
        orient=tk.HORIZONTAL,
        command=update_max_entry,
        length=250,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background'),
        showvalue=False  # Hide the default value display
    )
    app.max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
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
    
    # Create variables to store the slider values
    min_amp_ratio = tk.DoubleVar(value=app.double_peak_min_amp_ratio.get())
    max_amp_ratio = tk.DoubleVar(value=app.double_peak_max_amp_ratio.get())
    
    # Functions to update app variables and sync entries
    def update_min_amp_entry(val):
        val = float(val)
        min_amp_entry.delete(0, tk.END)
        min_amp_entry.insert(0, f"{val:.2f}")
        min_amp_ratio.set(val)
        app.double_peak_min_amp_ratio.set(val)
    
    def update_max_amp_entry(val):
        val = float(val)
        max_amp_entry.delete(0, tk.END)
        max_amp_entry.insert(0, f"{val:.2f}")
        max_amp_ratio.set(val)
        app.double_peak_max_amp_ratio.set(val)
    
    def sync_min_amp_slider_to_entry(*args):
        try:
            value = float(min_amp_entry.get())
            if value >= 0.0 and value <= max_amp_ratio.get():
                app.min_amp_slider.set(value)
                min_amp_ratio.set(value)
                app.double_peak_min_amp_ratio.set(value)
            else:
                min_amp_entry.delete(0, tk.END)
                min_amp_entry.insert(0, f"{min_amp_ratio.get():.2f}")
        except ValueError:
            min_amp_entry.delete(0, tk.END)
            min_amp_entry.insert(0, f"{min_amp_ratio.get():.2f}")
    
    def sync_max_amp_slider_to_entry(*args):
        try:
            value = float(max_amp_entry.get())
            if value >= min_amp_ratio.get() and value <= 5.0:
                app.max_amp_slider.set(value)
                max_amp_ratio.set(value)
                app.double_peak_max_amp_ratio.set(value)
            else:
                max_amp_entry.delete(0, tk.END)
                max_amp_entry.insert(0, f"{max_amp_ratio.get():.2f}")
        except ValueError:
            max_amp_entry.delete(0, tk.END)
            max_amp_entry.insert(0, f"{max_amp_ratio.get():.2f}")
    
    # Min amplitude ratio slider and entry
    min_amp_slider_frame = ttk.Frame(amp_frame)
    min_amp_slider_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(min_amp_slider_frame, text="Min:").pack(side=tk.LEFT, padx=5)
    app.min_amp_slider = tk.Scale(
        min_amp_slider_frame, 
        from_=0.0, 
        to=2.0,
        variable=min_amp_ratio, 
        orient=tk.HORIZONTAL,
        command=update_min_amp_entry,
        length=250,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background'),
        showvalue=False,  # Hide the default value display
        resolution=0.01
    )
    app.min_amp_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    min_amp_entry = ttk.Entry(min_amp_slider_frame, width=6)
    min_amp_entry.pack(side=tk.LEFT, padx=5)
    min_amp_entry.insert(0, f"{min_amp_ratio.get():.2f}")
    min_amp_entry.bind("<Return>", sync_min_amp_slider_to_entry)
    min_amp_entry.bind("<FocusOut>", sync_min_amp_slider_to_entry)
    
    # Max amplitude ratio slider and entry
    max_amp_slider_frame = ttk.Frame(amp_frame)
    max_amp_slider_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(max_amp_slider_frame, text="Max:").pack(side=tk.LEFT, padx=5)
    app.max_amp_slider = tk.Scale(
        max_amp_slider_frame, 
        from_=0.1, 
        to=5.0,
        variable=max_amp_ratio, 
        orient=tk.HORIZONTAL,
        command=update_max_amp_entry,
        length=250,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background'),
        showvalue=False,  # Hide the default value display
        resolution=0.01
    )
    app.max_amp_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    max_amp_entry = ttk.Entry(max_amp_slider_frame, width=6)
    max_amp_entry.pack(side=tk.LEFT, padx=5)
    max_amp_entry.insert(0, f"{max_amp_ratio.get():.2f}")
    max_amp_entry.bind("<Return>", sync_max_amp_slider_to_entry)
    max_amp_entry.bind("<FocusOut>", sync_max_amp_slider_to_entry)
    
    # Histogram frame
    amp_hist_frame = ttk.Frame(amp_frame)
    amp_hist_frame.pack(fill=tk.X, padx=5, pady=5)
    
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
    
    # Width ratio parameters
    width_frame = ttk.LabelFrame(param_frame, text="Width Ratio")
    width_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create variables to store the slider values
    min_width_ratio = tk.DoubleVar(value=app.double_peak_min_width_ratio.get())
    max_width_ratio = tk.DoubleVar(value=app.double_peak_max_width_ratio.get())
    
    # Functions to update app variables and sync entries
    def update_min_width_entry(val):
        val = float(val)
        min_width_entry.delete(0, tk.END)
        min_width_entry.insert(0, f"{val:.2f}")
        min_width_ratio.set(val)
        app.double_peak_min_width_ratio.set(val)
    
    def update_max_width_entry(val):
        val = float(val)
        max_width_entry.delete(0, tk.END)
        max_width_entry.insert(0, f"{val:.2f}")
        max_width_ratio.set(val)
        app.double_peak_max_width_ratio.set(val)
    
    def sync_min_width_slider_to_entry(*args):
        try:
            value = float(min_width_entry.get())
            if value >= 0.0 and value <= max_width_ratio.get():
                app.min_width_slider.set(value)
                min_width_ratio.set(value)
                app.double_peak_min_width_ratio.set(value)
            else:
                min_width_entry.delete(0, tk.END)
                min_width_entry.insert(0, f"{min_width_ratio.get():.2f}")
        except ValueError:
            min_width_entry.delete(0, tk.END)
            min_width_entry.insert(0, f"{min_width_ratio.get():.2f}")
    
    def sync_max_width_slider_to_entry(*args):
        try:
            value = float(max_width_entry.get())
            if value >= min_width_ratio.get() and value <= 5.0:
                app.max_width_slider.set(value)
                max_width_ratio.set(value)
                app.double_peak_max_width_ratio.set(value)
            else:
                max_width_entry.delete(0, tk.END)
                max_width_entry.insert(0, f"{max_width_ratio.get():.2f}")
        except ValueError:
            max_width_entry.delete(0, tk.END)
            max_width_entry.insert(0, f"{max_width_ratio.get():.2f}")
    
    # Min width ratio slider and entry
    min_width_slider_frame = ttk.Frame(width_frame)
    min_width_slider_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(min_width_slider_frame, text="Min:").pack(side=tk.LEFT, padx=5)
    app.min_width_slider = tk.Scale(
        min_width_slider_frame, 
        from_=0.0, 
        to=2.0,
        variable=min_width_ratio, 
        orient=tk.HORIZONTAL,
        command=update_min_width_entry,
        length=250,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background'),
        showvalue=False,  # Hide the default value display
        resolution=0.01
    )
    app.min_width_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    min_width_entry = ttk.Entry(min_width_slider_frame, width=6)
    min_width_entry.pack(side=tk.LEFT, padx=5)
    min_width_entry.insert(0, f"{min_width_ratio.get():.2f}")
    min_width_entry.bind("<Return>", sync_min_width_slider_to_entry)
    min_width_entry.bind("<FocusOut>", sync_min_width_slider_to_entry)
    
    # Max width ratio slider and entry
    max_width_slider_frame = ttk.Frame(width_frame)
    max_width_slider_frame.pack(fill=tk.X, padx=5, pady=2)
    
    ttk.Label(max_width_slider_frame, text="Max:").pack(side=tk.LEFT, padx=5)
    app.max_width_slider = tk.Scale(
        max_width_slider_frame, 
        from_=0.1, 
        to=5.0,
        variable=max_width_ratio, 
        orient=tk.HORIZONTAL,
        command=update_max_width_entry,
        length=250,
        bg=app.theme_manager.get_color('background'),
        fg=app.theme_manager.get_color('text'),
        highlightthickness=0,
        troughcolor=app.theme_manager.get_color('background'),
        showvalue=False,  # Hide the default value display
        resolution=0.01
    )
    app.max_width_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    max_width_entry = ttk.Entry(max_width_slider_frame, width=6)
    max_width_entry.pack(side=tk.LEFT, padx=5)
    max_width_entry.insert(0, f"{max_width_ratio.get():.2f}")
    max_width_entry.bind("<Return>", sync_max_width_slider_to_entry)
    max_width_entry.bind("<FocusOut>", sync_max_width_slider_to_entry)
    
    # Histogram frame
    width_hist_frame = ttk.Frame(width_frame)
    width_hist_frame.pack(fill=tk.X, padx=5, pady=5)
    
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