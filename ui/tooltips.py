"""
Enhanced tooltip functionality for the Peak Analysis Tool.

This module provides improved tooltips with animations, styling,
and better positioning than the standard tkinter tooltips.
"""

import tkinter as tk
from tkinter import ttk

class EnhancedTooltip:
    """
    Enhanced tooltips with animations and styling.
    
    This class provides a more professional tooltip implementation
    with fade-in/fade-out effects, auto-positioning, and themed styling.
    """
    
    def __init__(self, widget, text, delay=300, duration=5000, 
                 bg="#505a64", fg="white", font=None, padding=5):
        """
        Initialize a new enhanced tooltip.
        
        Parameters
        ----------
        widget : tk.Widget
            The widget to attach the tooltip to
        text : str
            The tooltip text
        delay : int, optional
            Delay in milliseconds before showing the tooltip (reduced for responsiveness)
        duration : int, optional
            Duration in milliseconds to show the tooltip (0 for indefinite)
        bg : str, optional
            Background color
        fg : str, optional
            Foreground color
        font : tuple, optional
            Font configuration
        padding : int, optional
            Padding around tooltip text
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.duration = duration
        self.bg = bg
        self.fg = fg
        self.font = font or ("Segoe UI", 9)
        self.padding = padding
        
        self.tooltip_window = None
        self.scheduled_show = None
        self.scheduled_hide = None
        self.alpha = 0.0
        
        # Bind events to widget
        widget.bind("<Enter>", self.on_enter)
        widget.bind("<Leave>", self.on_leave)
        widget.bind("<ButtonPress>", self.on_leave)
    
    def on_enter(self, event=None):
        """Handle mouse entering the widget."""
        # Cancel any scheduled hide event
        if self.scheduled_hide:
            self.widget.after_cancel(self.scheduled_hide)
            self.scheduled_hide = None
        
        # Schedule showing the tooltip
        if not self.tooltip_window:
            self.scheduled_show = self.widget.after(self.delay, self.show_tooltip)
    
    def on_leave(self, event=None):
        """Handle mouse leaving the widget."""
        # Cancel any scheduled show event
        if self.scheduled_show:
            self.widget.after_cancel(self.scheduled_show)
            self.scheduled_show = None
        
        # Schedule hiding the tooltip immediately
        if self.tooltip_window:
            self.hide_tooltip()  # No fade for faster response
    
    def show_tooltip(self):
        """Show the tooltip with minimal animation for better responsiveness."""
        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations
        
        # Make it float above other windows
        self.tooltip_window.attributes("-topmost", True)
        
        # Create tooltip content
        frame = ttk.Frame(self.tooltip_window)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a label for the tooltip text
        label = ttk.Label(
            frame, 
            text=self.text, 
            background=self.bg,
            foreground=self.fg,
            font=self.font,
            wraplength=400,
            justify=tk.LEFT,
            padding=self.padding
        )
        label.pack(fill=tk.BOTH, expand=True)
        
        # Position tooltip near the widget
        self.position_tooltip()
        
        # Appear immediately with minimal fade-in
        self.alpha = 0.8  # Start at higher transparency
        self.tooltip_window.attributes("-alpha", self.alpha)
        self.tooltip_window.update_idletasks()
        
        # Complete fade-in quickly
        self.fade_in()
        
        # Set auto-hide if duration is specified
        if self.duration > 0:
            self.widget.after(self.duration, self.hide_tooltip)
    
    def position_tooltip(self):
        """Position the tooltip relative to the widget and screen."""
        # Get widget position and dimensions
        x, y, width, height = self.get_widget_geometry()
        
        # Get tooltip size
        self.tooltip_window.update_idletasks()
        tooltip_width = self.tooltip_window.winfo_reqwidth()
        tooltip_height = self.tooltip_window.winfo_reqheight()
        
        # Get screen dimensions
        screen_width = self.widget.winfo_screenwidth()
        screen_height = self.widget.winfo_screenheight()
        
        # Default position below the widget
        tooltip_x = x + width // 2 - tooltip_width // 2
        tooltip_y = y + height + 5
        
        # Check if tooltip would go off-screen to the right
        if tooltip_x + tooltip_width > screen_width:
            tooltip_x = screen_width - tooltip_width - 10
        
        # Check if tooltip would go off-screen to the left
        if tooltip_x < 0:
            tooltip_x = 10
        
        # Check if tooltip would go off-screen at the bottom
        if tooltip_y + tooltip_height > screen_height:
            # Position above the widget instead
            tooltip_y = y - tooltip_height - 5
        
        # Set tooltip position
        self.tooltip_window.wm_geometry(f"+{tooltip_x}+{tooltip_y}")
    
    def get_widget_geometry(self):
        """Get widget position relative to the screen."""
        # Get widget position relative to its parent
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty()
        width = self.widget.winfo_width()
        height = self.widget.winfo_height()
        
        return x, y, width, height
    
    def fade_in(self):
        """Quickly increase opacity for a minimal fade-in effect."""
        if self.tooltip_window:
            if self.alpha < 1.0:
                self.alpha += 0.2  # Faster increment
                self.tooltip_window.attributes("-alpha", min(self.alpha, 1.0))
                self.widget.after(10, self.fade_in)  # Faster timer
    
    def start_fade_out(self):
        """Start the fade-out effect."""
        if self.tooltip_window:
            self.hide_tooltip()  # Skip fade-out for faster response
    
    def fade_out(self):
        """Quickly decrease opacity for a minimal fade-out effect."""
        if self.tooltip_window:
            if self.alpha > 0.1:
                self.alpha -= 0.2  # Faster decrement
                self.tooltip_window.attributes("-alpha", max(self.alpha, 0.0))
                self.widget.after(10, self.fade_out)  # Faster timer
            else:
                self.hide_tooltip()
    
    def hide_tooltip(self):
        """Hide and destroy the tooltip window immediately."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
    
    def update_text(self, text):
        """Update the tooltip text."""
        self.text = text
        # If tooltip is currently shown, update its text
        if self.tooltip_window:
            for child in self.tooltip_window.winfo_children():
                if isinstance(child, ttk.Frame):
                    for label in child.winfo_children():
                        if isinstance(label, ttk.Label):
                            label.configure(text=text)

def create_tooltip(widget, text, **kwargs):
    """
    Create an enhanced tooltip for a widget.
    
    Parameters
    ----------
    widget : tk.Widget
        The widget to attach the tooltip to
    text : str
        The tooltip text
    **kwargs : dict
        Additional options for the tooltip
        
    Returns
    -------
    EnhancedTooltip
        The created tooltip instance
    """
    return EnhancedTooltip(widget, text, **kwargs) 