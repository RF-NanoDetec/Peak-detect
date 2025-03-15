"""
Peak Analysis Tool - Main Application
==================================

Created: 2024
Authors: Lucjan & Silas

This is the main entry point for the Peak Analysis Tool application.
It initializes the application, shows a splash screen, and handles high-level
error management.

The application is designed for scientific peak analysis with a focus on:
- Data loading and visualization
- Peak detection algorithms
- Statistical analysis of peak characteristics
- Export of results and visualizations

For configuration options, see the config module.
For detailed usage instructions, refer to the README.md file.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Import configuration and utilities from reorganized structure
from config import (
    APP_VERSION, 
    DEBUG_MODE, 
    resource_path, 
    setup_logging, 
    logger, 
    get_app_info
)
from config.settings import Config
from ui import ThemeManager, StatusIndicator
from core.peak_detection import PeakDetector
from utils.performance import get_memory_usage
from main import Application  # Import the Application class from main.py

def splash_screen():
    """
    Display a splash screen while the application is loading.
    
    This function creates and displays a splash screen with the application logo, 
    version information, and a loading bar. The splash screen is shown for 3 seconds
    before automatically closing.
    
    The splash screen is centered on the user's display and provides visual feedback
    that the application is starting up.
    
    Raises:
        Exception: If there's an error loading or displaying the splash screen image,
                  the error is logged but the application will continue to start.
    """
    splash = tk.Tk()
    splash.overrideredirect(True)  # Remove window decorations
    splash.title(f"Peak Analysis Tool v{APP_VERSION}")

    # Load and display the image
    img_path = resource_path("resources/images/startim.png")
    try:
        img = Image.open(img_path)
        img = img.resize((400, 400), Image.LANCZOS)  # Resize to a larger square size
        photo = ImageTk.PhotoImage(img)

        label = tk.Label(splash, image=photo)
        label.pack()

        # Add version text
        version_label = tk.Label(
            splash, 
            text=f"Peak Analysis Tool v{APP_VERSION}", 
            font=("Helvetica", 12, "bold")
        )
        version_label.pack(pady=(0, 10))

        # Create a loading bar
        loading_bar = ttk.Progressbar(splash, orient="horizontal", length=300, mode="indeterminate")
        loading_bar.pack(pady=10)  # Add some vertical padding
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
        splash.after(3000, lambda: [loading_bar.stop(), splash.destroy()])
        splash.mainloop()
    except Exception as e:
        logger.error(f"Error showing splash screen: {e}")

def main():
    """
    Main entry point for the application.
    
    This function serves as the primary entry point for the Peak Analysis Tool.
    It performs the following operations in sequence:
    1. Initializes logging and records application startup
    2. Displays the splash screen
    3. Reports initial memory usage
    4. Creates and runs the main application
    5. Catches and handles any unhandled exceptions
    
    The function implements global error handling to prevent crashes and
    provide user-friendly error messages when problems occur.
    
    Returns:
        None
        
    Raises:
        SystemExit: If the application exits normally through the main event loop
    """
    # Log application startup and environment info
    logger.info(f"Starting Peak Analysis Tool v{APP_VERSION}")
    logger.info(f"Environment: {get_app_info()}")
    
    # Show splash screen
    splash_screen()
    
    # Log initial memory usage
    logger.info(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    # Create and run the application
    try:
        app = Application()
        app.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        # Show error dialog to user
        messagebox.showerror(
            "Application Error",
            f"An error occurred while running the application:\n{e}\n\nPlease check the log files."
        )

if __name__ == "__main__":
    main() 