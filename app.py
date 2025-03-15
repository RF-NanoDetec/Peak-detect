"""
Peak Analysis Tool - Main Application

Created: 2024
Authors: Lucjan & Silas

This is the main entry point for the Peak Analysis Tool application.
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
    Show a splash screen while the application is loading
    
    This function displays a splash screen with a logo and loading bar
    during the application startup process.
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
    Main entry point for the application
    
    This function initializes the application components,
    shows the splash screen, and starts the main event loop.
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