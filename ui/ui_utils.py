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
    # Use Toplevel instead of Label to ensure tooltip stays on top
    tooltip = tk.Toplevel(widget.master)
    tooltip.withdraw()  # Initially hidden
    tooltip.overrideredirect(True)  # No window decorations
    tooltip.attributes("-topmost", True)  # Stay on top of all windows
    
    # Create a frame with a label inside the Toplevel
    frame = tk.Frame(tooltip, bg="#FFFFEA", relief="solid", borderwidth=1)
    frame.pack(fill="both", expand=True)
    
    label = tk.Label(
        frame, 
        text=text, 
        bg="#FFFFEA", 
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
        font=('Helvetica', 16, 'bold'),
        foreground=app.theme_manager.get_color('primary')
    )
    header_label.pack(pady=5)
    
    version_label = ttk.Label(
        header_frame,
        text=f"Version {APP_VERSION}",
        font=('Helvetica', 10),
        foreground=app.theme_manager.get_color('text_secondary')
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
            background=app.theme_manager.get_color('card_bg'),
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
    # Configure tags for different markdown elements
    text_widget.tag_configure('h1', font=('Helvetica', 16, 'bold'))
    text_widget.tag_configure('h2', font=('Helvetica', 14, 'bold'))
    text_widget.tag_configure('h3', font=('Helvetica', 12, 'bold'))
    text_widget.tag_configure('h4', font=('Helvetica', 11, 'bold'))
    text_widget.tag_configure('code', font=('Courier', 10), background='#f0f0f0')
    text_widget.tag_configure('bold', font=('Helvetica', 11, 'bold'))
    text_widget.tag_configure('italic', font=('Helvetica', 11, 'italic'))
    text_widget.tag_configure('bullet', lmargin1=20, lmargin2=30)
    text_widget.tag_configure('link', foreground='blue', underline=True)
    
    # Process the content line by line
    content = text_widget.get('1.0', tk.END)
    text_widget.delete('1.0', tk.END)
    
    for line in content.split('\n'):
        # Handle headers
        if line.startswith('### '):
            text_widget.insert(tk.END, line[4:] + '\n', 'h3')
        elif line.startswith('#### '):
            text_widget.insert(tk.END, line[5:] + '\n', 'h4')
        # Handle bullet points
        elif line.strip().startswith('- '):
            text_widget.insert(tk.END, line + '\n', 'bullet')
        # Handle numbered lists
        elif line.strip() and line.strip()[0].isdigit() and line.strip()[1:].startswith('. '):
            text_widget.insert(tk.END, line + '\n', 'bullet')
        # Handle code blocks
        elif line.strip().startswith('```') and line.strip().endswith('```'):
            code_content = line.strip()[3:-3]
            text_widget.insert(tk.END, code_content + '\n', 'code')
        else:
            # Process inline formatting
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
                elif i < len(line) and line[i] == '*' and '*' in line[i+1:]:
                    end = line.find('*', i+1)
                    text_widget.insert(tk.END, processed_line)
                    processed_line = ""
                    text_widget.insert(tk.END, line[i+1:end], 'italic')
                    i = end + 1
                # Inline code
                elif i < len(line) and line[i] == '`' and '`' in line[i+1:]:
                    end = line.find('`', i+1)
                    text_widget.insert(tk.END, processed_line)
                    processed_line = ""
                    text_widget.insert(tk.END, line[i+1:end], 'code')
                    i = end + 1
                else:
                    processed_line += line[i]
                    i += 1
            
            if processed_line:
                text_widget.insert(tk.END, processed_line + '\n')
    
    # Handle table formatting
    content = text_widget.get('1.0', tk.END)
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '|' in line:
            # Check if this is a table header separator
            if line.strip().startswith('|') and all(c in '|-:' for c in line.strip()):
                continue
                
            # Format table rows
            text_widget.delete(f"{i+1}.0", f"{i+1}.end")
            cells = [cell.strip() for cell in line.split('|')]
            formatted_line = "  ".join(cells)
            text_widget.insert(f"{i+1}.0", formatted_line)

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
    Update the results summary text widget with analysis results.
    This is a UI wrapper around the core update_results_summary function.
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
        
        return True
    except Exception as e:
        app.show_error("Error updating results summary", str(e))
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