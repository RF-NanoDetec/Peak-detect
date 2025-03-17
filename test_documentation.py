"""
Test script for the documentation viewer.

This script creates a simple window with a button to launch the documentation
viewer, allowing for easy testing of the documentation feature.
"""

import os
import tkinter as tk
from tkinter import ttk
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the documentation function and theme manager
from ui.ui_utils import show_documentation_with_ui
from ui.theme import ThemeManager

class MockApp(tk.Tk):
    """A mock application class for testing the documentation viewer"""
    
    def __init__(self):
        super().__init__()
        self.title("Documentation Test")
        self.geometry("400x200")
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        
        # Create a simple UI
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        header = ttk.Label(
            main_frame, 
            text="Documentation Viewer Test",
            font=('Helvetica', 14, 'bold')
        )
        header.pack(pady=10)
        
        description = ttk.Label(
            main_frame,
            text="Click the button below to test the documentation viewer",
            wraplength=350
        )
        description.pack(pady=10)
        
        # Button to launch documentation
        doc_button = ttk.Button(
            main_frame,
            text="Show Documentation",
            command=self.show_docs
        )
        doc_button.pack(pady=10)
        
    def show_docs(self):
        """Show the documentation viewer"""
        show_documentation_with_ui(self)
        
    def get_icon_path(self):
        """Mock method to provide an icon path"""
        return ""
        
if __name__ == "__main__":
    app = MockApp()
    app.mainloop() 