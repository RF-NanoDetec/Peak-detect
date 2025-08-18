"""
UI utility functions for the Peak Analysis Tool.

This module contains utility functions for managing the application's
user interface elements, including updating, validating, and displaying information.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import traceback
import logging
from PIL import ImageGrab
import os
from config.settings import Config
import matplotlib.pyplot as plt
from functools import wraps
from PIL import Image
from config import resource_path, APP_VERSION
from core.peak_analysis_utils import (
    compute_baseline_mask,
    compute_noise_stats,
    compute_snr_values,
)
import numpy as np
import webbrowser

# Configure logging
logger = logging.getLogger(__name__)

def update_results_summary(app, events=None, max_amp=None, peak_areas=None, 
                         peak_intervals=None, preview_text=None):
    """
    Update the results summary text box.
    
    Parameters
    ----------
    app : Application
        The main application instance
    events : int, optional
        Number of detected peaks
    max_amp : float, optional
        Maximum peak amplitude
    peak_areas : list, optional
        List of peak areas
    peak_intervals : list, optional
        List of peak intervals
    preview_text : str, optional
        Text to display in the preview section
    """
    try:
        app.results_summary.config(state=tk.NORMAL)
        app.results_summary.delete(1.0, tk.END)  # Clear existing content
        
        summary_text = "=== PEAK DETECTION SUMMARY ===\n"
        
        # Most critical information first - peak counts and throughput
        if events is not None:
            summary_text += f"Number of Peaks Detected: {events}\n"
            
            # Calculate and add throughput information if we have time values
            if hasattr(app, 't_value') and app.t_value is not None and len(app.t_value) > 1:
                total_time_minutes = (app.t_value[-1] - app.t_value[0]) / 60  # Convert to minutes
                if total_time_minutes > 0:
                    avg_throughput = events / total_time_minutes
                    summary_text += f"Average Throughput: {avg_throughput:.2f} events/minute\n"
                    
                    # If we have peak intervals, we can calculate more detailed throughput stats
                    if peak_intervals is not None and len(peak_intervals) > 1:
                        valid_intervals = [i for i in peak_intervals[1:] if i > 0]  # Skip first and zero intervals
                        if len(valid_intervals) > 0:
                            min_interval = np.min(valid_intervals)
                            max_throughput = 60.0 / min_interval if min_interval > 0 else 0  # Max possible throughput
                            summary_text += f"Maximum Throughput: {max_throughput:.2f} events/minute\n"
                            
                            # Calculate throughput in different time windows
                            windows = [60, 30, 10]  # seconds
                            for window in windows:
                                bins = np.arange(app.t_value[0], app.t_value[-1], window)
                                if len(bins) > 1:
                                    if hasattr(app, 'peaks') and app.peaks is not None:
                                        counts, _ = np.histogram(app.t_value[app.peaks], bins=bins)
                                    else:
                                        counts, _ = np.histogram([], bins=bins)
                                    max_window_throughput = np.max(counts) * (60 / window)  # Convert to per minute
                                    summary_text += f"Max {window}s Window Throughput: {max_window_throughput:.2f} events/minute\n"
            
            summary_text += "\n"  # Add spacing after throughput section
        
        # Add peak height statistics if available
        if hasattr(app, 'peak_heights') and app.peak_heights is not None and len(app.peak_heights) > 0:
            mean_height = np.mean(app.peak_heights)
            median_height = np.median(app.peak_heights)
            std_height = np.std(app.peak_heights)
            summary_text += f"Peak Heights (Mean ± SD): {mean_height:.2f} ± {std_height:.2f}\n"
            summary_text += f"Peak Heights (Median): {median_height:.2f}\n"
            
        # Add max amplitude if available
        if max_amp is not None:
            summary_text += f"Maximum Peak Amplitude: {max_amp:.2f}\n"
            
        # Add peak width statistics if available
        if hasattr(app, 'peak_widths') and app.peak_widths is not None and len(app.peak_widths) > 0:
            # Get time resolution to convert from samples to ms
            time_res = app.time_resolution.get() if hasattr(app, 'time_resolution') else 1e-4
            
            mean_width_samples = np.mean(app.peak_widths)
            median_width_samples = np.median(app.peak_widths)
            std_width_samples = np.std(app.peak_widths)
            
            # Convert from samples to milliseconds
            mean_width_ms = mean_width_samples * time_res * 1000
            median_width_ms = median_width_samples * time_res * 1000
            std_width_ms = std_width_samples * time_res * 1000
            
            summary_text += f"Peak Widths (Mean ± SD): {mean_width_ms:.2f} ± {std_width_ms:.2f} ms\n"
            summary_text += f"Peak Widths (Median): {median_width_ms:.2f} ms\n"
            
        # Add statistics section
        if peak_areas is not None or peak_intervals is not None:
            summary_text += "\n=== PEAK STATISTICS ===\n"
            
            if peak_areas is not None:
                # Calculate statistics for peak areas
                mean_area = np.mean(peak_areas) if len(peak_areas) > 0 else 0
                median_area = np.median(peak_areas) if len(peak_areas) > 0 else 0
                std_area = np.std(peak_areas) if len(peak_areas) > 0 else 0
                
                summary_text += f"Peak Areas (Mean ± SD): {mean_area:.2f} ± {std_area:.2f}\n"
                summary_text += f"Peak Areas (Median): {median_area:.2f}\n"
                summary_text += f"First 5 Areas: {[round(a, 2) for a in peak_areas[:5]]}\n"
            
            if peak_intervals is not None:
                # Calculate statistics for peak intervals
                valid_intervals = [i for i in peak_intervals[1:] if i > 0]  # Skip first and zero intervals
                
                if len(valid_intervals) > 0:
                    mean_interval = np.mean(valid_intervals)
                    median_interval = np.median(valid_intervals)
                    min_interval = np.min(valid_intervals)
                    max_interval = np.max(valid_intervals)
                    
                    summary_text += f"Peak Intervals (Mean): {mean_interval:.2f} seconds\n"
                    summary_text += f"Peak Intervals (Median): {median_interval:.2f} seconds\n"
                    summary_text += f"Peak Intervals (Range): {min_interval:.2f} - {max_interval:.2f} seconds\n"
                    summary_text += f"First 5 Intervals: {[round(i, 2) for i in peak_intervals[1:6]]}\n"
                    
                    # Estimate throughput
                    if mean_interval > 0:
                        mean_throughput = 60.0 / mean_interval  # Events per minute based on mean
                        summary_text += f"Throughput (Mean): {mean_throughput:.2f} events/minute\n"
                    
                    if median_interval > 0:
                        median_throughput = 60.0 / median_interval  # Events per minute based on median
                        summary_text += f"Throughput (Median): {median_throughput:.2f} events/minute\n"
                    
                    if min_interval > 0:
                        max_throughput = 60.0 / min_interval  # Max possible throughput
                        summary_text += f"Max Throughput: {max_throughput:.2f} events/minute\n"
        
        # Add filter information and metadata as provided
        if preview_text is not None:
            summary_text += f"\n{preview_text}\n"
        
        app.results_summary.insert(tk.END, summary_text)
        app.results_summary.config(state=tk.DISABLED)
        
    except Exception as e:
        show_error(app, "Error updating results", e)
        app.results_summary.config(state=tk.DISABLED)

def validate_float(value):
    """
    Validate that the input is a valid float.
    
    Parameters
    ----------
    value : str
        The value to validate
        
    Returns
    -------
    bool
        True if the value is a valid float, False otherwise
    """
    if value == "":
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False

def update_progress_bar(app, value=0, maximum=None):
    """
    Update progress bar with optional maximum value.
    
    Parameters
    ----------
    app : Application
        The main application instance
    value : int, optional
        Current progress value
    maximum : int, optional
        Maximum progress value
    """
    try:
        if maximum is not None:
            app.progress['maximum'] = maximum
        app.progress['value'] = value
        app.update_idletasks()
        
        # Reset to zero if completed
        if value >= app.progress['maximum']:
            app.after(Config.PROGRESS_RESET_DELAY, 
                      lambda: update_progress_bar(app, 0))
    except Exception as e:
        logger.error(f"Error updating progress bar: {e}")

def take_screenshot(app):
    """
    Take a screenshot of the entire application window
    
    Parameters
    ----------
    app : Application
        The main application instance
    """
    try:
        # Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Taking screenshot...")
        
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            title="Save Screenshot"
        )
        
        if file_path:
            # Wait for dialog to close
            app.update_idletasks()
            app.after(200)  # Short delay
            
            # Get window geometry
            x = app.winfo_rootx()
            y = app.winfo_rooty()
            width = app.winfo_width()
            height = app.winfo_height()
            
            # Take screenshot
            screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
            screenshot.save(file_path)
            
            # Update status
            app.status_indicator.set_state('success')
            app.status_indicator.set_text(f"Screenshot saved to {os.path.basename(file_path)}")
            
            app.preview_label.config(
                text=f"Screenshot saved to {file_path}", 
                foreground=app.theme_manager.get_color('success')
            )
        else:
            # Screenshot cancelled
            app.status_indicator.set_state('idle')
            app.status_indicator.set_text("Screenshot cancelled")
            
    except Exception as e:
        # Update status
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error taking screenshot")
        show_error(app, "Error taking screenshot", e)

def show_error(app, title, error):
    """
    Display error message and log error details
    
    Parameters
    ----------
    app : Application
        The main application instance
    title : str
        Error title
    error : Exception
        Exception object
    """
    # Log error
    logger.error(f"{title}: {str(error)}")
    logger.error(traceback.format_exc())
    
    # Update status label
    app.preview_label.config(
        text=f"{title}: {str(error)}", 
        foreground=app.theme_manager.get_color('error')
    )
    
    # Show error message box for serious errors
    tk.messagebox.showerror(title, str(error))

def add_tooltip(widget, text):
    """
    Add a tooltip to a widget.
    
    Parameters
    ----------
    widget : tk.Widget
        The widget to add the tooltip to
    text : str
        The tooltip text
    """
    # Use Toplevel instead of Label to ensure tooltip stays on top
    tooltip = tk.Toplevel(widget.master)
    tooltip.withdraw()  # Initially hidden
    tooltip.overrideredirect(True)  # No window decorations
    tooltip.attributes("-topmost", True)  # Stay on top of all windows
    
    # Create a frame with a label inside the Toplevel
    # Use theme colors if available via widget's app reference
    try:
        app = widget.winfo_toplevel()
        from ui import ThemeManager  # for isinstance check if needed
        # Pick a subtle tooltip background derived from theme panel/card
        bg_color = getattr(app, 'theme_manager', None).get_color('card_bg') if getattr(app, 'theme_manager', None) else "#FFFFEA"
        border_color = getattr(app, 'theme_manager', None).get_color('border') if getattr(app, 'theme_manager', None) else "#CCCCAA"
    except Exception:
        bg_color = "#FFFFEA"
        border_color = "#CCCCAA"
    frame = tk.Frame(tooltip, bg=bg_color, relief="solid", borderwidth=1)
    frame.pack(fill="both", expand=True)
    
    # High-contrast text based on theme
    try:
        fg_color = app.theme_manager.get_color('text') if app.theme_manager.current_theme == 'light' else '#ffffff'
    except Exception:
        fg_color = 'black'
    label = tk.Label(
        frame, 
        text=text, 
        bg=bg_color, 
        fg=fg_color,
        padx=5,
        pady=3,
        wraplength=250,
        justify="left"
    )
    label.pack()
    
    def enter(event):
        # Get widget position
        x = widget.winfo_rootx() + widget.winfo_width() + 2
        y = widget.winfo_rooty() + widget.winfo_height() // 2
        
        # Check if tooltip would go off the right edge of the screen
        tooltip.update_idletasks()  # Update to calculate tooltip size
        tooltip_width = tooltip.winfo_reqwidth()
        screen_width = widget.winfo_screenwidth()
        
        if x + tooltip_width > screen_width:
            # Position tooltip to the left of the widget instead
            x = widget.winfo_rootx() - tooltip_width - 2
        
        # Position and show tooltip
        tooltip.geometry(f"+{x}+{y}")
        tooltip.deiconify()
        
    def leave(event):
        tooltip.withdraw()
        
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)
    
    # Hide tooltip when widget is destroyed
    def on_widget_destroy(event):
        tooltip.destroy()
    
    widget.bind("<Destroy>", on_widget_destroy)
    
    return tooltip

def show_documentation(app):
    """
    Show application documentation.
    
    Parameters
    ----------
    app : Application
        The main application instance
    """
    # In a real application, this might open a browser to online docs
    # or display a help window
    documentation_text = """
Peak Analysis Tool Documentation
===============================

This tool helps analyze signal data to detect and characterize peaks.

Main Features:
1. Load single or batch files for analysis
2. Apply filtering to reduce noise
3. Detect peaks with customizable parameters
4. Analyze peak properties (width, height, area)
5. Generate visualizations and reports

For detailed usage instructions, please refer to the full documentation.
"""
    help_window = tk.Toplevel(app)
    help_window.title("Documentation")
    help_window.geometry("600x400")
    
    text_widget = ScrolledText(help_window, wrap=tk.WORD)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    text_widget.insert(tk.END, documentation_text)
    text_widget.config(state=tk.DISABLED)

def show_about_dialog(app):
    """
    Show about dialog with application information.
    
    Parameters
    ----------
    app : Application
        The main application instance
    """
    version = "1.0"
    about_text = f"""
Peak Analysis Tool v{version}

Developed by Lucjan & Silas

This tool provides advanced peak detection and analysis capabilities 
for signal processing and scientific data analysis.

© 2024 All rights reserved.
"""
    tk.messagebox.showinfo("About Peak Analysis Tool", about_text)

def on_file_mode_change(app):
    """
    Handle file mode changes between single and batch modes.
    
    Parameters
    ----------
    app : Application
        The main application instance
    """
    if app.file_mode.get() == "batch":
        app.timestamps_label.pack(side=tk.LEFT, padx=5, pady=5)
        app.timestamps_entry.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)
        app.browse_button.config(text="Load Files with Timestamps")
    else:
        app.timestamps_label.pack_forget()
        app.timestamps_entry.pack_forget()
        app.browse_button.config(text="Load Files")
    
    # Force update of the GUI
    app.update_idletasks() 

def with_ui_error_handling(app, processing_message, success_message, error_message, function, *args, **kwargs):
    """
    Execute a function with standard UI status updates and error handling.
    
    This utility function updates the UI before and after executing a function,
    and handles errors in a consistent way. It reduces code duplication in
    UI wrapper methods.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements
    processing_message : str
        Status message during processing
    success_message : str
        Status message on success
    error_message : str
        Status message on error
    function : callable
        The function to execute
    *args, **kwargs
        Arguments to pass to the function
        
    Returns
    -------
    Any
        The return value from the function, or None if an error occurred
    """
    try:
        # Update status for processing
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text(processing_message)
        app.update_idletasks()
        
        # Call the function
        result = function(*args, **kwargs)
        
        # Update status for success
        app.status_indicator.set_state('success')
        app.status_indicator.set_text(success_message)
        
        # Update preview label if appropriate
        if hasattr(app, 'preview_label'):
            app.preview_label.config(
                text=success_message,
                foreground=app.theme_manager.get_color('success')
            )
            
        return result
        
    except Exception as e:
        # Update status for error
        app.status_indicator.set_state('error')
        app.status_indicator.set_text(error_message)
        
        # Show error dialog
        show_error(app, error_message, e)
        
        # Update preview label if appropriate
        if hasattr(app, 'preview_label'):
            app.preview_label.config(
                text=f"{error_message}: {str(e)}",
                foreground=app.theme_manager.get_color('error')
            )
            
        return None 

def ui_action(processing_message, success_message, error_message):
    """
    Decorator for Application class methods that handle UI updates and error handling.
    
    This decorator wraps Application class methods with standard UI status updates
    and error handling logic, reducing code duplication.
    
    Parameters
    ----------
    processing_message : str
        Status message during processing
    success_message : str
        Status message on success
    error_message : str
        Status message on error
        
    Returns
    -------
    function
        Decorated function with UI updates and error handling
        
    Example
    -------
    @ui_action("Processing data...", "Data processed successfully", "Error processing data")
    def process_data(self, param1, param2):
        # Method code that focuses on business logic only
        result = some_core_function(param1, param2)
        return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                # Update status for processing
                self.status_indicator.set_state('processing')
                self.status_indicator.set_text(processing_message)
                self.update_idletasks()
                
                # Call the method
                result = func(self, *args, **kwargs)
                
                # Update status for success
                self.status_indicator.set_state('success')
                self.status_indicator.set_text(success_message)
                
                # Update preview label if appropriate
                if hasattr(self, 'preview_label'):
                    self.preview_label.config(
                        text=success_message,
                        foreground=self.theme_manager.get_color('success')
                    )
                    
                return result
                
            except Exception as e:
                # Update status for error
                self.status_indicator.set_state('error')
                self.status_indicator.set_text(error_message)
                
                # Show error dialog
                show_error(self, error_message, e)
                
                # Update preview label if appropriate
                if hasattr(self, 'preview_label'):
                    self.preview_label.config(
                        text=f"{error_message}: {str(e)}",
                        foreground=self.theme_manager.get_color('error')
                    )
                    
                return None
        return wrapper
    return decorator 

def ui_operation(processing_message, success_message, error_message):
    """
    Decorator for Application class methods that perform UI operations.
    
    Similar to ui_action, but with a more neutral name for operations that
    aren't specifically user-initiated actions but still need UI feedback.
    
    Parameters
    ----------
    processing_message : str
        Status message during processing
    success_message : str
        Status message on success
    error_message : str
        Status message on error
        
    Returns
    -------
    function
        Decorated function with UI updates and error handling
        
    Example
    -------
    @ui_operation("Adding tooltip...", "Tooltip added", "Error adding tooltip")
    def add_tooltip(self, widget, text):
        # Method code for adding tooltip
        return tooltip_object
    """
    # Implementation is identical to ui_action but kept separate for semantic clarity
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                # Update status for processing
                self.status_indicator.set_state('processing')
                self.status_indicator.set_text(processing_message)
                self.update_idletasks()
                
                # Call the method
                result = func(self, *args, **kwargs)
                
                # Update status for success
                self.status_indicator.set_state('success')
                self.status_indicator.set_text(success_message)
                
                # Update preview label if appropriate
                if hasattr(self, 'preview_label'):
                    self.preview_label.config(
                        text=success_message,
                        foreground=self.theme_manager.get_color('success')
                    )
                    
                return result
                
            except Exception as e:
                # Update status for error
                self.status_indicator.set_state('error')
                self.status_indicator.set_text(error_message)
                
                # Show error dialog
                show_error(self, error_message, e)
                
                # Update preview label if appropriate
                if hasattr(self, 'preview_label'):
                    self.preview_label.config(
                        text=f"{error_message}: {str(e)}",
                        foreground=self.theme_manager.get_color('error')
                    )
                    
                return None
        return wrapper
    return decorator

def show_documentation_with_ui(app):
    """
    Display comprehensive application documentation in a separate window.
    
    This function creates a scrollable text window with markdown-formatted
    documentation that explains all aspects of the application in detail.
    
    Parameters
    ----------
    app : Application
        The main application instance
        
    Returns
    -------
    Toplevel
        The documentation window object
    """
    # --- Preferred: Open pre-rendered PDF documentation ---
    pdf_path = resource_path('docs/user_manual.pdf')
    if os.path.exists(pdf_path):
        try:
            webbrowser.open_new(pdf_path)  # Open with system's default PDF viewer
            # Update status indicator & preview label for success
            app.status_indicator.set_state('success')
            app.status_indicator.set_text("Documentation opened (PDF)")
            if hasattr(app, 'preview_label'):
                app.preview_label.config(
                    text="Opened documentation PDF",
                    foreground=app.theme_manager.get_color('success')
                )
            return None  # Skip creating the Tk window entirely
        except Exception as e:
            # Fall back to internal viewer if PDF cannot be opened
            logger.error(f"Failed to open PDF docs: {e}\n{traceback.format_exc()}")

    # --- Fallback: internal markdown viewer as before ---
    # Create a new window
    doc_window = tk.Toplevel(app)
    doc_window.title("Peak Analysis Tool - Comprehensive Documentation")
    doc_window.geometry("900x700")
    doc_window.minsize(800, 600)
    
    # Add a custom icon
    try:
        doc_window.iconbitmap(app.get_icon_path())
    except:
        pass  # Ignore if icon can't be set
    
    # Create main frame with proper styling
    main_frame = ttk.Frame(doc_window, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Add styled header
    header_frame = ttk.Frame(main_frame)
    header_frame.pack(fill=tk.X, pady=(0, 10))
    
    header_label = ttk.Label(
        header_frame,
        text="Peak Analysis Tool Documentation",
        style="Heading.TLabel"
    )
    header_label.pack(pady=5)
    
    version_label = ttk.Label(
        header_frame,
        text=f"Version {APP_VERSION}",
        style="Small.TLabel"
    )
    version_label.pack()
    
    # Add separator
    separator = ttk.Separator(main_frame, orient='horizontal')
    separator.pack(fill=tk.X, pady=10)
    
    # Create notebook for tabbed documentation
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Read documentation from markdown file
    try:
        with open(resource_path('docs/user_manual.md'), 'r') as f:
            doc_content = f.read()
    except:
        doc_content = "Documentation file not found. Please reinstall the application."
    
    # Split the documentation into sections based on ## headers
    sections = []
    current_section = {"title": "Overview", "content": ""}
    
    for line in doc_content.split('\n'):
        if line.startswith('## '):
            # Save the previous section
            if current_section["title"] != "Overview" or current_section["content"]:
                sections.append(current_section)
            
            # Start a new section
            current_section = {
                "title": line.replace('## ', ''),
                "content": ""
            }
        elif line.startswith('# '):
            # This is the main title, skip it
            continue
        else:
            # Add to current section content
            current_section["content"] += line + "\n"
    
    # Add the last section
    if current_section:
        sections.append(current_section)
    
    # Create a tab for each major section
    for section in sections:
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=section["title"])
        
        # Create scrollable text widget for the section
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create text widget with scrollbar
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Helvetica', 11),
            padx=10,
            pady=10,
            background=app.theme_manager.get_color('background'),
            foreground=app.theme_manager.get_color('text'),
            highlightthickness=0,
            relief=tk.FLAT
        )
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Pack the text widget and scrollbar
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Insert the section content with markdown-like formatting
        text_widget.insert(tk.END, section["content"])
        
        # Apply simple markdown formatting
        apply_markdown_formatting(text_widget)
        
        # Make the text read-only
        text_widget.configure(state=tk.DISABLED)
    
    # Add a search feature
    search_frame = ttk.Frame(main_frame)
    search_frame.pack(fill=tk.X, pady=10)
    
    search_label = ttk.Label(search_frame, text="Search: ")
    search_label.pack(side=tk.LEFT, padx=5)
    
    search_var = tk.StringVar()
    search_entry = ttk.Entry(search_frame, textvariable=search_var, width=30)
    search_entry.pack(side=tk.LEFT, padx=5)
    
    def search_documentation():
        """Search through all tabs for the search term"""
        search_term = search_var.get().lower()
        if not search_term:
            return
        
        # Search through all tabs
        for tab_idx in range(notebook.index('end')):
            tab = notebook.winfo_children()[tab_idx]
            for child in tab.winfo_children():
                if isinstance(child, ttk.Frame):
                    for widget in child.winfo_children():
                        if isinstance(widget, tk.Text):
                            # Reset any previous tags
                            widget.tag_remove('search', '1.0', tk.END)
                            
                            # Search for the term
                            start_pos = '1.0'
                            while True:
                                start_pos = widget.search(search_term, start_pos, tk.END, nocase=True)
                                if not start_pos:
                                    break
                                
                                end_pos = f"{start_pos}+{len(search_term)}c"
                                widget.tag_add('search', start_pos, end_pos)
                                widget.tag_config('search', background='yellow', foreground='black')
                                
                                # Move to the next position
                                start_pos = end_pos
                            
                            # If found, switch to this tab
                            if widget.tag_ranges('search'):
                                notebook.select(tab_idx)
                                # Scroll to the first occurrence
                                widget.see(widget.tag_nextrange('search', '1.0')[0])
                                return
    
    search_button = ttk.Button(search_frame, text="Find", command=search_documentation)
    search_button.pack(side=tk.LEFT, padx=5)
    
    # Bind Enter key to search
    search_entry.bind('<Return>', lambda e: search_documentation())
    
    # Add a close button
    close_button = ttk.Button(main_frame, text="Close", command=doc_window.destroy)
    close_button.pack(pady=10)
    
    # Make the window modal
    doc_window.transient(app)
    doc_window.grab_set()
    
    return doc_window

def apply_markdown_formatting(text_widget):
    """
    Apply basic markdown-like formatting to a text widget.
    
    Parameters
    ----------
    text_widget : tk.Text
        The text widget to format
    """
    # Configure tags for different markdown elements (bigger font for h1/h2)
    text_widget.tag_configure('h1', font=('Helvetica', 18, 'bold'))
    text_widget.tag_configure('h2', font=('Helvetica', 16, 'bold'))
    text_widget.tag_configure('h3', font=('Helvetica', 14, 'bold'))
    text_widget.tag_configure('h4', font=('Helvetica', 12, 'bold'))
    # Other styles
    text_widget.tag_configure('code', font=('Courier', 10), background='#f0f0f0')
    text_widget.tag_configure('bold', font=('Helvetica', 11, 'bold'))
    text_widget.tag_configure('italic', font=('Helvetica', 11, 'italic'))
    text_widget.tag_configure('bullet', lmargin1=20, lmargin2=30)
    text_widget.tag_configure('link', foreground='blue', underline=True)
    
    # Process the content line by line
    content = text_widget.get('1.0', tk.END)
    text_widget.delete('1.0', tk.END)
    
    # Flags for multi-line code blocks
    in_code_block = False
    for line in content.split('\n'):
        # Check for code block delimiters
        if line.strip().startswith('```'):
            if in_code_block:
                # Closing code block
                in_code_block = False
                text_widget.insert(tk.END, '\n')
            else:
                # Opening code block (language info ignored)
                in_code_block = True
            continue  # Skip the delimiter line itself

        if in_code_block:
            # Everything inside code block uses 'code' tag
            text_widget.insert(tk.END, line + '\n', 'code')
            continue

        # ===== Headers =====
        if line.startswith('# '):
            text_widget.insert(tk.END, line[2:] + '\n', 'h1')
            continue
        elif line.startswith('## '):
            text_widget.insert(tk.END, line[3:] + '\n', 'h2')
            continue
        elif line.startswith('### '):
            text_widget.insert(tk.END, line[4:] + '\n', 'h3')
            continue
        elif line.startswith('#### '):
            text_widget.insert(tk.END, line[5:] + '\n', 'h4')
            continue

        # ===== Horizontal rule =====
        if line.strip() in {'---', '***', '___'}:
            text_widget.insert(tk.END, '\n')
            continue

        # ===== Bullet & numbered lists =====
        stripped = line.lstrip()
        if stripped.startswith('- '):
            text_widget.insert(tk.END, line + '\n', 'bullet')
            continue
        if stripped[:1].isdigit() and stripped[1:].startswith('. '):
            text_widget.insert(tk.END, line + '\n', 'bullet')
            continue

        # ===== Process inline formatting =====
        processed_line = ""
        i = 0
        while i < len(line):
            # Bold text
            if i+1 < len(line) and line[i:i+2] == '**' and '**' in line[i+2:]:
                end = line.find('**', i+2)
                text_widget.insert(tk.END, processed_line)
                processed_line = ""
                text_widget.insert(tk.END, line[i+2:end], 'bold')
                i = end + 2
            # Italic text
            elif line[i] == '*' and '*' in line[i+1:]:
                end = line.find('*', i+1)
                text_widget.insert(tk.END, processed_line)
                processed_line = ""
                text_widget.insert(tk.END, line[i+1:end], 'italic')
                i = end + 1
            # Inline code
            elif line[i] == '`' and '`' in line[i+1:]:
                end = line.find('`', i+1)
                text_widget.insert(tk.END, processed_line)
                processed_line = ""
                text_widget.insert(tk.END, line[i+1:end], 'code')
                i = end + 1
            else:
                processed_line += line[i]
                i += 1

        text_widget.insert(tk.END, processed_line + '\n')

def show_about_dialog_with_ui(app):
    """
    Show about dialog with integrated UI handling.
    
    This function directly integrates UI updates, eliminating the need
    for a separate wrapper method in the Application class.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
        
    Returns
    -------
    None
    """
    try:
        # UI pre-processing: Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Loading about dialog...")
        app.update_idletasks()
        
        # Show the about dialog
        show_about_dialog(app)
        
        # UI post-processing: Update status with success
        app.status_indicator.set_state('success')
        app.status_indicator.set_text("About dialog displayed")
        
        # Update preview label
        app.preview_label.config(
            text="About dialog displayed",
            foreground=app.theme_manager.get_color('success')
        )
        
        return None
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error showing about dialog: {str(e)}\n{traceback.format_exc()}")
        
        # Update UI with error info
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error showing about dialog")
        
        # Show error dialog
        show_error(app, "Error showing about dialog", e)
        
        # Update preview label
        app.preview_label.config(
            text=f"Error showing about dialog: {str(e)}",
            foreground=app.theme_manager.get_color('error')
        )
        
        return None 

def take_screenshot_with_ui(app):
    """
    Take a screenshot of the application window with integrated UI handling.
    
    This function directly integrates UI updates, eliminating the need
    for a separate wrapper method in the Application class.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
        
    Returns
    -------
    str or None
        Path to the saved screenshot file, or None if canceled or an error occurred
    """
    try:
        # UI pre-processing: Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Taking screenshot...")
        app.update_idletasks()
        
        # Open file save dialog
        file_path = tk.filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            title="Save Screenshot"
        )
        
        if not file_path:
            # Screenshot cancelled
            app.status_indicator.set_state('idle')
            app.status_indicator.set_text("Screenshot cancelled")
            return None
            
        # Wait for dialog to close
        app.update_idletasks()
        app.after(200)  # Short delay
        
        # Get window geometry
        x = app.winfo_rootx()
        y = app.winfo_rooty()
        width = app.winfo_width()
        height = app.winfo_height()
        
        # Take screenshot
        screenshot = ImageGrab.grab(bbox=(x, y, x+width, y+height))
        screenshot.save(file_path)
        
        # UI post-processing: Update status with success
        app.status_indicator.set_state('success')
        app.status_indicator.set_text(f"Screenshot saved to {os.path.basename(file_path)}")
        
        # Update preview label
        app.preview_label.config(
            text=f"Screenshot saved to {file_path}",
            foreground=app.theme_manager.get_color('success')
        )
        
        return file_path
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error taking screenshot: {str(e)}\n{traceback.format_exc()}")
        
        # Update UI with error info
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error taking screenshot")
        
        # Show error dialog
        show_error(app, "Error taking screenshot", e)
        
        # Update preview label
        app.preview_label.config(
            text=f"Error taking screenshot: {str(e)}",
            foreground=app.theme_manager.get_color('error')
        )
        
        return None 

def update_results_summary_with_ui(app, events=None, max_amp=None, peak_areas=None, peak_intervals=None, preview_text=None, context="peak_analysis"):
    """
    Update the results summary text widget with analysis results.
    Now includes sections for different processing stages based on context or available data.

    Args:
        app: The main application instance.
        events (int, optional): Number of events (e.g., peaks kept).
        max_amp (float, optional): Maximum amplitude detected.
        peak_areas (list/np.ndarray, optional): Calculated peak areas.
        peak_intervals (float, optional): Total peak intervals.
        preview_text (str, optional): Text for the preview label.
        context (str, optional): Hint about which process triggered the update (e.g., 'data_loading', 'preprocessing', 'peak_detection', 'peak_analysis', 'double_peaks'). Defaults to 'peak_analysis'.
    """
    try:
        # Only update preview label if it exists
        if hasattr(app, 'preview_label'):
            if preview_text:
                app.preview_label.config(text=preview_text, foreground=app.theme_manager.get_color('success'))
            elif events is not None:
                app.preview_label.config(
                    text=f"Found {events} peaks",
                    foreground=app.theme_manager.get_color('success')
                )
            elif peak_areas is not None:
                app.preview_label.config(
                    text=f"Calculated peak areas",
                    foreground=app.theme_manager.get_color('success')
                )

        # Update the core results_summary text widget if it exists
        if hasattr(app, 'results_summary'):
            summary_content = ""
            show_all_sections = context == "all" # A way to show everything if needed

            # Clear existing content
            app.results_summary.config(state=tk.NORMAL)
            app.results_summary.delete("1.0", tk.END)

            # === DATA LOADING ===
            if hasattr(app, 'data_handler') and hasattr(app.data_handler, 'file_path') and app.data_handler.file_path:
                data_info = "=== LOAD DATA ===\n"
                data_info += f"File: {os.path.basename(app.data_handler.file_path)}\n"
                if hasattr(app.data_handler, 'raw_data') and app.data_handler.raw_data is not None:
                    data_info += f"Data Points: {len(app.data_handler.raw_data)}\n"
                if hasattr(app, 't_value') and app.t_value is not None and len(app.t_value) > 1:
                    duration_seconds = app.t_value[-1] - app.t_value[0]
                    data_info += f"Duration: {duration_seconds:.2f} s ({duration_seconds/60:.2f} min)\n"
                summary_content += data_info + "\n"
            elif show_all_sections :
                summary_content += "=== LOAD DATA ===\nNo data loaded.\n\n"


            # === PREPROCESSING & FILTER SETTINGS ===
            preprocessing_info = ""
            if hasattr(app, 'baseline_correction_method') and app.baseline_correction_method.get() != 'None':
                 if not preprocessing_info: preprocessing_info = "=== PREPROCESSING ===\n"
                 preprocessing_info += f"Baseline Correction: {app.baseline_correction_method.get()}\n"
                 if app.baseline_correction_method.get() == 'Rolling Ball' and hasattr(app, 'baseline_window_size'):
                     preprocessing_info += f"  Window Size: {app.baseline_window_size.get()}\n"
                 elif app.baseline_correction_method.get() == 'ALS' and hasattr(app, 'als_lambda') and hasattr(app, 'als_p'):
                     preprocessing_info += f"  Lambda: {app.als_lambda.get()}, P: {app.als_p.get()}\n"

            # --- Applied Filter Settings ---
            # This section now reads from app.applied_filter_settings
            if not preprocessing_info: preprocessing_info = "=== PREPROCESSING ===\n" # Start section if not already
            
            if hasattr(app, 'applied_filter_settings') and app.applied_filter_settings:
                filter_params = app.applied_filter_settings
                filter_type = filter_params.get('type', 'N/A')
                
                preprocessing_info += f"Signal Filtering: {filter_type.capitalize() if filter_type not in ['none', 'N/A'] else 'Disabled'}\n"

                if filter_type == 'butterworth':
                    cutoff = filter_params.get('cutoff_hz', 'N/A')
                    order = filter_params.get('order', 'N/A')
                    estimation = filter_params.get('estimation_method', 'N/A')
                    preprocessing_info += f"  Type: Butterworth\n"
                    preprocessing_info += f"  Cutoff: {cutoff:.2f} Hz (Order: {order})\n" if isinstance(cutoff, float) else f"  Cutoff: {cutoff} (Order: {order})\n"
                    preprocessing_info += f"  Cutoff Estimation: {estimation}\n"
                elif filter_type == 'savgol':
                    window = filter_params.get('window_length', 'N/A')
                    polyorder = filter_params.get('polyorder', 'N/A')
                    preprocessing_info += f"  Type: Savitzky-Golay\n"
                    preprocessing_info += f"  Window Length: {window}\n"
                    preprocessing_info += f"  Polynomial Order: {polyorder}\n"
                    if 'estimated_window_source' in filter_params:
                        preprocessing_info += f"  Window Estimation: {filter_params['estimated_window_source']}\n"
                elif filter_type == 'none':
                    pass # Already handled by "Signal Filtering: Disabled"
                elif 'error' in filter_params:
                    preprocessing_info += f"  Error applying filter: {filter_params['error']}\n"
                elif filter_type != 'N/A': # For unknown filter types if they somehow occur
                    preprocessing_info += f"  Details: {filter_params}\n"
            elif hasattr(app, 'filter_enabled') and not app.filter_enabled.get(): # Fallback if applied_filter_settings is missing
                 preprocessing_info += f"Signal Filtering: Disabled (Legacy Check)\n"
            else:
                preprocessing_info += "Signal Filtering: Settings N/A (Analysis likely not run)\n"


            if preprocessing_info or show_all_sections: # Ensure section header prints if forcing all
                if "=== PREPROCESSING ===" not in preprocessing_info and show_all_sections:
                     summary_content += "=== PREPROCESSING ===\nNo specific preprocessing steps recorded or N/A.\n\n"
                else:
                     summary_content += preprocessing_info + "\n"


            # === DETECTION PARAMETERS ===
            if hasattr(app, 'height_lim') or hasattr(app, 'distance') or hasattr(app, 'rel_height') or hasattr(app, 'width_p') or show_all_sections:
                parameters_info = "=== DETECTION PARAMETERS ===\n"
                if hasattr(app, 'height_lim'):
                    parameters_info += f"Height Threshold: {app.height_lim.get()}\n"
                else: parameters_info += "Height Threshold: N/A\n"
                if hasattr(app, 'distance'):
                    parameters_info += f"Minimum Distance: {app.distance.get()} samples\n"
                if hasattr(app, 'rel_height'):
                    parameters_info += f"Relative Height: {app.rel_height.get()}\n"
                if hasattr(app, 'width_p'):
                    width_range = app.width_p.get()
                    time_res_ms = app.time_resolution.get() * 1000 if hasattr(app, 'time_resolution') else None
                    try:
                        width_samples = [int(w.strip()) for w in width_range.split(',')]
                        if time_res_ms:
                             parameters_info += f"Width Range: {width_samples[0] * time_res_ms:.2f} - {width_samples[1] * time_res_ms:.2f} ms ({width_samples[0]}-{width_samples[1]} samples)\n"
                        else:
                             parameters_info += f"Width Range: {width_samples[0]} - {width_samples[1]} samples\n"
                    except:
                        parameters_info += f"Width Range: {width_range} (raw samples)\n" 
                if hasattr(app, 'time_resolution'):
                    time_res_ms = app.time_resolution.get() * 1000
                    parameters_info += f"Time Resolution: {time_res_ms:.4f} ms/sample\n"
                summary_content += parameters_info + "\n"


            # === PEAK ANALYSIS (SUB-PEAK FILTERING) ===
            # Renamed for clarity
            filtering_analysis_info = ""
            if hasattr(app, 'prominence_ratio') or show_all_sections:
                 if not filtering_analysis_info: filtering_analysis_info = "=== PEAK ANALYSIS (SUB-PEAK FILTERING) ===\n"
                 if hasattr(app, 'prominence_ratio'):
                     prominence_ratio = app.prominence_ratio.get()
                     filtering_analysis_info += f"Sub-peak Prominence Ratio Threshold: {prominence_ratio:.2f}\n"
                 else: filtering_analysis_info += "Sub-peak Prominence Ratio Threshold: N/A\n"


            if hasattr(app, 'peak_detector') and hasattr(app.peak_detector, 'all_peaks_count') and app.peak_detector.all_peaks_count is not None:
                total_peaks = app.peak_detector.all_peaks_count
                if total_peaks >= 0: 
                    if not filtering_analysis_info: filtering_analysis_info = "=== PEAK ANALYSIS (SUB-PEAK FILTERING) ===\n"
                    filtered_kept = events if events is not None else total_peaks 
                    filtered_out = total_peaks - filtered_kept
                    filtered_percentage = (filtered_out / total_peaks * 100) if total_peaks > 0 else 0
                    kept_percentage = 100 - filtered_percentage

                    filtering_analysis_info += f"Total Peaks Detected (Pre-SubFilter): {total_peaks}\n"
                    filtering_analysis_info += f"Sub-Peaks Filtered Out (Prominence): {filtered_out} ({filtered_percentage:.1f}%)\n"
                    filtering_analysis_info += f"Peaks Retained (Post-SubFilter): {filtered_kept} ({kept_percentage:.1f}%)\n"
            elif show_all_sections and not filtering_analysis_info:
                 filtering_analysis_info = "=== PEAK ANALYSIS (SUB-PEAK FILTERING) ===\nSub-peak filtering stats N/A.\n"


            if filtering_analysis_info:
                 summary_content += filtering_analysis_info + "\n"
            
            # === DOUBLE PEAK ANALYSIS ===
            if hasattr(app, 'double_peak_results') or show_all_sections: 
                 double_peak_info = "=== DOUBLE PEAK ANALYSIS ===\n"
                 if hasattr(app, 'double_peak_results'):
                     results = app.double_peak_results 
                     analyzed = results.get('analyzed', 'N/A')
                     double_peaks_found = results.get('double_peaks', 'N/A') # Renamed for clarity
                     min_dist = results.get('min_dist_ms', 'N/A')
                     max_dist = results.get('max_dist_ms', 'N/A')
                     percentage = (double_peaks_found / analyzed * 100) if isinstance(analyzed, int) and isinstance(double_peaks_found, int) and analyzed > 0 else 0

                     double_peak_info += f"Distance Range Analyzed: {min_dist} - {max_dist} ms\n"
                     double_peak_info += f"Peak Pairs Analyzed: {analyzed}\n"
                     double_peak_info += f"Double Peaks Found: {double_peaks_found} ({percentage:.1f}%)\n"
                 else: double_peak_info += "Double peak analysis not performed or results N/A.\n"
                 summary_content += double_peak_info + "\n"


            # === SIGNAL INFORMATION (POST-PROCESSING) ===
            # Clarified section title
            if hasattr(app, 'filtered_signal') and app.filtered_signal is not None or show_all_sections:
                signal_info = "=== SIGNAL INFORMATION (POST-PROCESSING) ===\n"
                if hasattr(app, 'filtered_signal') and app.filtered_signal is not None:
                    signal = app.filtered_signal
                    signal_mean = np.mean(signal)
                    signal_median = np.median(signal)
                    signal_std = np.std(signal)
                    signal_min = np.min(signal)
                    signal_max = np.max(signal)

                    signal_info += f"Signal Mean: {signal_mean:.2f}\n"
                    signal_info += f"Signal Median: {signal_median:.2f}\n"
                    signal_info += f"Signal Standard Deviation: {signal_std:.2f}\n"
                    signal_info += f"Signal Range: {signal_min:.2f} - {signal_max:.2f}\n"

                    if signal_std > 0:
                        snr_est = (signal_max - signal_mean) / signal_std
                        signal_info += f"Estimated Signal-to-Noise Ratio (heuristic): {snr_est:.2f}\n"

                    signal_info += f"Signal Length: {len(signal)} samples\n"
                    if not ('Duration:' in summary_content) and hasattr(app, 't_value') and app.t_value is not None and len(app.t_value) > 1:
                         duration_seconds = app.t_value[-1] - app.t_value[0]
                         signal_info += f"Signal Duration: {duration_seconds:.2f} s ({duration_seconds/60:.2f} min)\n"
                else: signal_info += "Signal information N/A (data not processed or loaded).\n"
                summary_content += signal_info + "\n"


            # === PEAK STATISTICS (POST-SUBFILTER) ===
            # Clarified section title
            peak_stats_info = ""
            retained_peaks_count = events if events is not None else (app.peak_detector.all_peaks_count if hasattr(app, 'peak_detector') and hasattr(app.peak_detector, 'all_peaks_count') and app.peak_detector.all_peaks_count is not None else 0)

            if retained_peaks_count > 0 or show_all_sections:
                 peak_stats_info = f"=== PEAK STATISTICS (Retained: {retained_peaks_count if retained_peaks_count > 0 else 'N/A'}) ===\n"
                 if retained_peaks_count > 0:
                     if hasattr(app, 'peak_heights') and app.peak_heights is not None and len(app.peak_heights) == retained_peaks_count:
                         heights = app.peak_heights
                         peak_stats_info += f"Peak Height (Mean ± SD): {np.mean(heights):.2f} ± {np.std(heights):.2f}\n"
                         peak_stats_info += f"Peak Height (Median): {np.median(heights):.2f}\n"
                         peak_stats_info += f"Peak Height (Min/Max): {np.min(heights):.2f} / {np.max(heights):.2f}\n"
                     else: peak_stats_info += "Peak Height Stats: N/A (data mismatch or not available)\n"

                     if hasattr(app, 'peak_widths') and app.peak_widths is not None and len(app.peak_widths) == retained_peaks_count:
                         widths_samples = app.peak_widths
                         time_res = app.time_resolution.get() if hasattr(app, 'time_resolution') else None
                         if time_res:
                             widths_time = widths_samples * time_res
                             peak_stats_info += f"Peak Width (Mean ± SD): {np.mean(widths_time)*1000:.2f} ± {np.std(widths_time)*1000:.2f} ms\n"
                             peak_stats_info += f"Peak Width (Median): {np.median(widths_time)*1000:.2f} ms\n"
                             peak_stats_info += f"Peak Width (Min/Max): {np.min(widths_time)*1000:.2f} / {np.max(widths_time)*1000:.2f} ms\n"
                         else: # Fallback to samples if time_res not available
                             peak_stats_info += f"Peak Width (Mean ± SD): {np.mean(widths_samples):.2f} ± {np.std(widths_samples):.2f} samples\n"
                     else: peak_stats_info += "Peak Width Stats: N/A (data mismatch or not available)\n"

                     if peak_areas is not None and len(peak_areas) == retained_peaks_count:
                         peak_stats_info += f"Peak Area (Mean ± SD): {np.mean(peak_areas):.2f} ± {np.std(peak_areas):.2f}\n"
                         peak_stats_info += f"Peak Area (Median): {np.median(peak_areas):.2f}\n"
                         peak_stats_info += f"Peak Area (Min/Max): {np.min(peak_areas):.2f} / {np.max(peak_areas):.2f}\n"
                     else: peak_stats_info += "Peak Area Stats: N/A (data mismatch or not available)\n"

                     if peak_intervals is not None and len(peak_intervals) > 0: # peak_intervals usually has one less element than peaks
                        intervals_ms = peak_intervals * 1000
                        peak_stats_info += f"Inter-Peak Interval (Mean ± SD): {np.mean(intervals_ms):.2f} ± {np.std(intervals_ms):.2f} ms\n"
                        peak_stats_info += f"Inter-Peak Interval (Median): {np.median(intervals_ms):.2f} ms\n"
                        peak_stats_info += f"Inter-Peak Interval (Min/Max): {np.min(intervals_ms):.2f} / {np.max(intervals_ms):.2f} ms\n"
                        if np.mean(intervals_ms) > 0:
                             throughput_epm = 60 * 1000 / np.mean(intervals_ms) # Events per minute
                             peak_stats_info += f"Throughput (Mean): {throughput_epm:.2f} events/minute\n"
                     else: peak_stats_info += "Inter-Peak Interval Stats: N/A (data mismatch or not available)\n"
                 elif show_all_sections : peak_stats_info += "No retained peaks to analyze or stats N/A.\n"


            # === SIGNAL-TO-NOISE (BASELINE METHOD) ===
            snr_section = ""
            try:
                if (
                    hasattr(app, 'filtered_signal') and app.filtered_signal is not None and
                    hasattr(app, 'peaks') and app.peaks is not None and len(app.peaks) > 0 and
                    hasattr(app, 'peak_widths') and app.peak_widths is not None and len(app.peak_widths) == len(app.peaks)
                ):
                    baseline_mask = compute_baseline_mask(
                        signal_length=len(app.filtered_signal),
                        peak_indices=app.peaks,
                        widths_in_samples=app.peak_widths,
                        multiplier=2.0,
                        left_indices=(app.peak_left_ips if hasattr(app, 'peak_left_ips') else None),
                        right_indices=(app.peak_right_ips if hasattr(app, 'peak_right_ips') else None),
                    )
                    baseline_signal = app.filtered_signal[baseline_mask]
                    noise_std, noise_mad_std, baseline_mean = compute_noise_stats(baseline_signal)

                    # Persist for reuse in plotting/export
                    app.baseline_points = int(np.sum(baseline_mask))
                    app.noise_std = noise_std
                    app.noise_mad_std = noise_mad_std
                    app.baseline_mean = baseline_mean

                    # Compute per-peak SNR using peak heights (prominences)
                    if hasattr(app, 'peak_heights') and app.peak_heights is not None and len(app.peak_heights) == len(app.peaks):
                        snr_values = compute_snr_values(app.peak_heights, noise_std)
                        app.snr_values = snr_values
                    else:
                        snr_values = np.array([])

                    snr_section = "=== SIGNAL-TO-NOISE (BASELINE METHOD) ===\n"
                    total_points = len(app.filtered_signal)
                    baseline_pct = (app.baseline_points / total_points * 100.0) if total_points > 0 else 0.0
                    snr_section += f"Baseline Points: {app.baseline_points} / {total_points} ({baseline_pct:.1f}%)\n"
                    snr_section += f"Noise Std (baseline): {noise_std:.4g}\n"
                    snr_section += f"Noise Robust (MAD*1.4826): {noise_mad_std:.4g}\n"

                    if snr_values.size > 0:
                        snr_section += f"SNR (Heights/Noise Std) Mean ± SD: {np.mean(snr_values):.2f} ± {np.std(snr_values):.2f}\n"
                        snr_section += f"SNR Median: {np.median(snr_values):.2f}\n"
                        snr_section += f"SNR Min/Max: {np.min(snr_values):.2f} / {np.max(snr_values):.2f}\n"
                    else:
                        if noise_std <= 0:
                            snr_section += "SNR: Undefined (noise std is zero).\n"
                        else:
                            snr_section += "SNR: N/A (peak heights unavailable).\n"
                    snr_section += "Exclusion rule: ±2×peak width around each peak excluded from baseline.\n\n"
            except Exception as _e:
                # Keep summary robust; omit SNR section on error
                snr_section = ""

            if snr_section:
                summary_content += snr_section


            if peak_stats_info: # Only add if there's content
                summary_content += peak_stats_info + "\n"

            app.results_summary.insert(tk.END, summary_content)
            app.results_summary.config(state=tk.DISABLED)
            
            # Scroll to top
            app.results_summary.yview_moveto(0)

    except Exception as e:
        # Use show_error_with_ui for consistent error display
        show_error_with_ui(app, "Error updating results summary", e)
        if hasattr(app, 'results_summary'): # Ensure it exists before trying to config
            app.results_summary.config(state=tk.DISABLED)

def validate_float_with_ui(app, value):
    """
    Validate that an input is a valid float with integrated UI handling.
    
    This is a simple pass-through function since validate_float is already
    a pure utility function without UI dependencies. However, we create this
    for consistency with our refactoring approach.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
    value : str
        The value to validate
        
    Returns
    -------
    bool
        True if the value is a valid float, False otherwise
    """
    return validate_float(value) 

def update_progress_bar_with_ui(app, value=0, maximum=None):
    """
    Update the progress bar with integrated UI handling.
    
    This function is a simple wrapper around update_progress_bar since
    it's already a UI utility function, but we create this for consistency
    with our refactoring approach.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
    value : int, optional
        Current progress value
    maximum : int, optional
        Maximum progress value
        
    Returns
    -------
    None
    """
    try:
        # Call the core function to update progress bar
        update_progress_bar(app, value, maximum)
        return None
        
    except Exception as e:
        # Handle errors silently (progress bar errors should not interrupt workflow)
        logger.error(f"Error updating progress bar: {str(e)}\n{traceback.format_exc()}")
        return None 

def on_file_mode_change_with_ui(app):
    """
    Handle file mode changes with integrated UI handling.
    
    This function directly integrates UI updates, eliminating the need
    for a separate wrapper method in the Application class.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
        
    Returns
    -------
    None
    """
    try:
        # UI pre-processing: Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Updating file mode...")
        app.update_idletasks()
        
        # Call the core function to update file mode
        on_file_mode_change(app)
        
        # UI post-processing: Update status with success
        app.status_indicator.set_state('success')
        
        # Set appropriate status text based on the mode
        mode_name = "standard" if app.file_mode.get() == "single" else "timestamp"
        app.status_indicator.set_text(f"Switched to {mode_name} mode")
        
        # Update preview label
        app.preview_label.config(
            text=f"Switched to {mode_name} mode",
            foreground=app.theme_manager.get_color('success')
        )
        
        return None
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error updating file mode: {str(e)}\n{traceback.format_exc()}")
        
        # Update UI with error info
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error updating file mode")
        
        # Show error dialog
        show_error(app, "Error updating file mode", e)
        
        # Update preview label
        app.preview_label.config(
            text=f"Error updating file mode: {str(e)}",
            foreground=app.theme_manager.get_color('error')
        )
        
        return None 

def show_error_with_ui(app, title, error):
    """
    Display error message with integrated UI handling.
    
    This is a simple wrapper around show_error since it's already a UI utility function,
    but we create this for consistency with our refactoring approach.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
    title : str
        Error title
    error : Exception
        Exception object
        
    Returns
    -------
    None
    """
    # The show_error function already updates UI elements, so we just pass through
    show_error(app, title, error)
    return None 