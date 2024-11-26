"""
GUI components for the Peak Analysis Tool.
-----------------------------------------

This module contains all GUI-related components including:
- Main application window
- Custom widgets
- Plot management
- User interface utilities
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging

logger = logging.getLogger(__name__)

# GUI Constants
class GuiConfig:
    """GUI-specific configuration settings."""
    PADDING = 5
    FONTS = {
        'default': ('Arial', 10),
        'title': ('Arial', 12, 'bold'),
        'small': ('Arial', 8),
    }
    COLORS = {
        'bg': '#f0f0f0',
        'fg': '#333333',
        'highlight': '#0078d7',
        'error': '#ff0000',
        'success': '#00ff00',
        'warning': '#ffa500',
    }

class ToolTip:
    """Create tooltips for widgets."""
    
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind('<Enter>', self.show)
        self.widget.bind('<Leave>', self.hide)

    def show(self, event=None):
        """Display the tooltip."""
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            self.tooltip,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padding=(5, 2)
        )
        label.pack()

    def hide(self, event=None):
        """Hide the tooltip."""
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class ProgressBar(ttk.Frame):
    """Custom progress bar with label."""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.setup_widgets()

    def setup_widgets(self):
        """Create and configure progress bar widgets."""
        self.progress = ttk.Progressbar(
            self,
            mode='determinate',
            length=200
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.label = ttk.Label(self, text="0%")
        self.label.pack(side=tk.LEFT, padx=5)

    def update(self, value, maximum=None):
        """Update progress bar value and label."""
        if maximum is not None:
            self.progress['maximum'] = maximum
        self.progress['value'] = value
        percentage = (value / self.progress['maximum']) * 100
        self.label['text'] = f"{percentage:.1f}%"
        self.update_idletasks()

class PlotManager:
    """Manage matplotlib plots in tkinter."""
    
    def __init__(self, master):
        self.master = master
        self.figures = {}
        self.canvases = {}

    def create_figure(self, name, figsize=(8, 6), dpi=100):
        """Create a new figure and canvas."""
        fig = Figure(figsize=figsize, dpi=dpi)
        canvas = FigureCanvasTkAgg(fig, self.master)
        self.figures[name] = fig
        self.canvases[name] = canvas
        return fig, canvas

    def get_figure(self, name):
        """Get existing figure or create new one."""
        return self.figures.get(name)

    def update_plot(self, name):
        """Update specific plot."""
        if name in self.canvases:
            self.canvases[name].draw_idle()

    def clear_figure(self, name):
        """Clear specific figure."""
        if name in self.figures:
            self.figures[name].clear()

class StatusBar(ttk.Frame):
    """Status bar with message and progress indicator."""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.setup_widgets()

    def setup_widgets(self):
        """Create status bar widgets."""
        self.message_label = ttk.Label(
            self,
            text="Ready",
            padding=(5, 2)
        )
        self.message_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress = ProgressBar(self)
        self.progress.pack(side=tk.RIGHT, padx=5)

    def set_message(self, message, message_type="info"):
        """Update status message with type-specific formatting."""
        colors = {
            "info": GuiConfig.COLORS['fg'],
            "error": GuiConfig.COLORS['error'],
            "success": GuiConfig.COLORS['success'],
            "warning": GuiConfig.COLORS['warning']
        }
        self.message_label['foreground'] = colors.get(message_type, GuiConfig.COLORS['fg'])
        self.message_label['text'] = message

def create_tooltip(widget, text):
    """Convenience function to create tooltips."""
    return ToolTip(widget, text)

def show_error(title, message):
    """Show error message box."""
    messagebox.showerror(title, str(message))
    logger.error(f"{title}: {message}")

def show_info(title, message):
    """Show information message box."""
    messagebox.showinfo(title, message)
    logger.info(f"{title}: {message}")

def show_warning(title, message):
    """Show warning message box."""
    messagebox.showwarning(title, message)
    logger.warning(f"{title}: {message}")

# Export public interface
__all__ = [
    'GuiConfig',
    'ToolTip',
    'ProgressBar',
    'PlotManager',
    'StatusBar',
    'create_tooltip',
    'show_error',
    'show_info',
    'show_warning'
]

# Setup matplotlib style
plt.style.use('seaborn')
