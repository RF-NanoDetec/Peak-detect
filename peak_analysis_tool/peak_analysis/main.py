"""
Main entry point for the Peak Analysis Tool.
Initializes and runs the application.
"""

import sys
import os
import logging
from pathlib import Path
import tkinter as tk
from typing import Optional

from .gui.application import Application
from .utils import setup_logging, ensure_directory
from .config import config

def init_directories() -> None:
    """Initialize necessary directories."""
    dirs = [
        config.file.CACHE_DIR,
        config.file.LOG_DIR,
        config.file.DATA_DIR,
        config.file.EXPORT_DIR
    ]
    
    for directory in dirs:
        ensure_directory(directory)

def init_logging() -> None:
    """Initialize logging configuration."""
    log_file = Path(config.file.LOG_DIR) / "peak_analysis.log"
    setup_logging(
        log_file=str(log_file),
        level=config.system.LOG_LEVEL
    )

def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are available
    """
    required_packages = [
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn',
        'numba'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:", ", ".join(missing_packages))
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_application(debug: bool = False) -> Optional[Application]:
    """
    Initialize and run the main application.
    
    Args:
        debug: Whether to run in debug mode
        
    Returns:
        Application instance if successful, None otherwise
    """
    try:
        # Initialize directories and logging
        init_directories()
        init_logging()
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Peak Analysis Tool")
        
        # Check dependencies
        if not check_dependencies():
            logger.error("Missing dependencies")
            return None
        
        # Create and configure root window
        root = tk.Tk()
        root.withdraw()  # Hide root window temporarily
        
        # Set application style
        style = tk.ttk.Style()
        style.theme_use(config.gui.THEME)
        
        # Create application instance
        app = Application()
        
        # Configure main window
        app.title("Peak Analysis Tool")
        app.geometry(f"{config.gui.WINDOW_SIZE[0]}x{config.gui.WINDOW_SIZE[1]}")
        app.minsize(
            config.gui.MIN_WINDOW_SIZE[0],
            config.gui.MIN_WINDOW_SIZE[1]
        )
        
        if debug:
            # Add debug information
            app.bind('<Control-d>', lambda e: show_debug_info(app))
            logger.info("Debug mode enabled")
        
        # Center window on screen
        screen_width = app.winfo_screenwidth()
        screen_height = app.winfo_screenheight()
        x = (screen_width - config.gui.WINDOW_SIZE[0]) // 2
        y = (screen_height - config.gui.WINDOW_SIZE[1]) // 2
        app.geometry(f"+{x}+{y}")
        
        # Show application window
        app.deiconify()
        
        # Start main loop
        app.mainloop()
        
        return app
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        if debug:
            raise
        return None

def show_debug_info(app: Application) -> None:
    """
    Show debug information window.
    
    Args:
        app: Application instance
    """
    debug_window = tk.Toplevel(app)
    debug_window.title("Debug Information")
    debug_window.geometry("400x300")
    
    text = tk.Text(debug_window, wrap=tk.WORD)
    text.pack(fill=tk.BOTH, expand=True)
    
    # Add system information
    import platform
    text.insert(tk.END, "System Information:\n")
    text.insert(tk.END, f"Python: {sys.version}\n")
    text.insert(tk.END, f"Platform: {platform.platform()}\n")
    text.insert(tk.END, f"Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB\n")
    
    # Add configuration
    text.insert(tk.END, "\nConfiguration:\n")
    for section, values in config.to_dict().items():
        text.insert(tk.END, f"\n{section}:\n")
        for key, value in values.items():
            text.insert(tk.END, f"  {key}: {value}\n")
    
    text.configure(state=tk.DISABLED)

def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Peak Analysis Tool")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    if args.config:
        try:
            config.load_config(args.config)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return 1
    
    # Run application
    app = run_application(debug=args.debug)
    
    # Return status code
    return 0 if app is not None else 1

if __name__ == "__main__":
    sys.exit(main())
