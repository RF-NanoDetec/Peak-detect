"""
UI utility functions for the Peak Analysis Tool.

This module contains utility functions for managing the application's
user interface elements, including updating, validating, and displaying information.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import traceback
import logging
from PIL import ImageGrab
import os
from config.settings import Config
import matplotlib.pyplot as plt
from functools import wraps
from PIL import Image

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
    tooltip = tk.Label(
        widget.master, 
        text=text, 
        bg="#FFFFEA", 
        relief="solid", 
        borderwidth=1
    )
    tooltip.place_forget()
    
    def enter(event):
        x, y, _, _ = widget.bbox("insert")
        x += widget.winfo_rootx() + 25
        y += widget.winfo_rooty() + 25
        tooltip.place(x=x, y=y)
        
    def leave(event):
        tooltip.place_forget()
        
    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

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
        app.browse_button.config(text="Select Folder")
    else:
        app.timestamps_label.pack_forget()
        app.timestamps_entry.pack_forget()
        app.browse_button.config(text="Load File")
    
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

def show_documentation_with_ui(app):
    """
    Show application documentation with integrated UI handling.
    
    This function directly integrates UI updates, eliminating the need
    for a separate wrapper method in the Application class.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
        
    Returns
    -------
    object
        The documentation window object
    """
    try:
        # UI pre-processing: Update status
        app.status_indicator.set_state('processing')
        app.status_indicator.set_text("Loading documentation...")
        app.update_idletasks()
        
        # Create the documentation window
        help_window = show_documentation(app)
        
        # UI post-processing: Update status with success
        app.status_indicator.set_state('success')
        app.status_indicator.set_text("Documentation displayed")
        
        # Update preview label
        app.preview_label.config(
            text="Documentation displayed",
            foreground=app.theme_manager.get_color('success')
        )
        
        return help_window
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error showing documentation: {str(e)}\n{traceback.format_exc()}")
        
        # Update UI with error info
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error showing documentation")
        
        # Show error dialog
        show_error(app, "Error showing documentation", e)
        
        # Update preview label
        app.preview_label.config(
            text=f"Error showing documentation: {str(e)}",
            foreground=app.theme_manager.get_color('error')
        )
        
        return None 

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

def update_results_summary_with_ui(app, events=None, max_amp=None, peak_areas=None, peak_intervals=None, preview_text=None):
    """
    Update the results summary with integrated UI handling.
    
    This function directly integrates UI updates, eliminating the need
    for a separate wrapper method in the Application class.
    
    Parameters
    ----------
    app : Application
        The main application instance with UI elements and state
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
        
    Returns
    -------
    bool
        True if the update was successful, False otherwise
    """
    try:
        # UI pre-processing: Update status (only if doing substantial updates)
        if events is not None or max_amp is not None or peak_areas is not None:
            app.status_indicator.set_state('processing')
            app.status_indicator.set_text("Updating results summary...")
            app.update_idletasks()
        
        # Call the core function to update results
        update_results_summary(app, events, max_amp, peak_areas, peak_intervals, preview_text)
        
        # UI post-processing: Update status with success (only for substantial updates)
        if events is not None or max_amp is not None or peak_areas is not None:
            app.status_indicator.set_state('success')
            app.status_indicator.set_text("Results summary updated")
            
            # Update preview label
            app.preview_label.config(
                text="Results summary updated",
                foreground=app.theme_manager.get_color('success')
            )
        
        return True
        
    except Exception as e:
        # Handle errors
        logger.error(f"Error updating results summary: {str(e)}\n{traceback.format_exc()}")
        
        # Update UI with error info
        app.status_indicator.set_state('error')
        app.status_indicator.set_text("Error updating results summary")
        
        # Show error dialog
        show_error(app, "Error updating results summary", e)
        
        # Update preview label
        app.preview_label.config(
            text=f"Error updating results summary: {str(e)}",
            foreground=app.theme_manager.get_color('error')
        )
        
        return False 

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
        mode_name = "single file" if app.file_mode.get() == "single" else "batch"
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