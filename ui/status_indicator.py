"""
Status indicator module for the Peak Analysis Tool.

This module provides status indicators for visual feedback
to show processing status, success, error, and other states.
"""

import tkinter as tk
from tkinter import ttk

class StatusIndicator(ttk.Frame):
    """
    Status indicator for visual feedback.
    
    This class provides a customizable status indicator that can show
    different states such as idle, processing, success, error, etc.
    """
    
    # Default colors when theme manager is not available
    STATES = {
        'idle': {'bg': '#e0e0e0', 'fg': '#505a64', 'text': 'Idle'},
        'loading': {'bg': '#6c8eb3', 'fg': 'white', 'text': 'Loading...'},
        'processing': {'bg': '#c89f5d', 'fg': 'white', 'text': 'Processing...'},
        'success': {'bg': '#7b9d6f', 'fg': 'white', 'text': 'Success'},
        'error': {'bg': '#ba6d6d', 'fg': 'white', 'text': 'Error'},
        'warning': {'bg': '#d4b157', 'fg': 'black', 'text': 'Warning'}
    }
    
    def __init__(self, parent, state='idle', theme_manager=None, **kwargs):
        """
        Initialize a new status indicator.
        
        Parameters
        ----------
        parent : tk.Widget
            Parent widget
        state : str, optional
            Initial state (idle, loading, processing, success, error, warning)
        theme_manager : ThemeManager, optional
            Theme manager for consistent styling
        **kwargs : dict
            Additional options for ttk.Frame
        """
        super().__init__(parent, **kwargs)
        
        self.theme_manager = theme_manager
        self._animation_running = False
        
        # Create indicator elements with proper padding for better visual appearance
        self.frame = ttk.Frame(self, padding=(8, 6))
        self.frame.pack(fill=tk.X, expand=True)
        
        # Indicator circle - slightly larger for better visibility
        self.indicator = tk.Canvas(self.frame, width=14, height=14, bd=0, highlightthickness=0)
        self.indicator.pack(side=tk.LEFT, padx=(0, 8))
        
        # Create initial indicator circle with default color
        # Initial indicator uses theme colors when available
        if self.theme_manager:
            fill = self.theme_manager.get_color('panel_bg')
            outline = self.theme_manager.get_color('border')
        else:
            fill = "#e0e0e0"
            outline = "#d0d0d0"
        self.indicator.create_oval(2, 2, 12, 12, fill=fill, outline=outline, tags="indicator")
        
        # Status label with appropriate font
        if self.theme_manager:
            font = self.theme_manager.get_font('default')
        else:
            font = ('Segoe UI', 10)
            
        self.label = ttk.Label(
            self.frame, 
            text=self.STATES['idle']['text'],
            font=font
        )
        self.label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Background frame color
        if self.theme_manager:
            self.configure(style='StatusFrame.TFrame')
            
        # Draw initial state
        self.set_state(state)
    
    def set_state(self, state):
        """
        Set the indicator state.
        
        Parameters
        ----------
        state : str
            State to set (idle, loading, processing, success, error, warning)
        """
        if state not in self.STATES:
            state = 'idle'
        
        try:
            # Use theme colors if available
            if self.theme_manager:
                # Get status text color for better contrast
                text_color = self.theme_manager.get_color('status_text')
                
                if state == 'idle':
                    bg_color = self.theme_manager.get_color('panel_bg')
                    text_color = self.theme_manager.get_color('text')
                elif state == 'loading':
                    bg_color = self.theme_manager.get_color('info')
                    # Keep text_color as status_text
                elif state == 'processing':
                    bg_color = self.theme_manager.get_color('warning')
                    # Keep text_color as status_text
                elif state == 'success':
                    bg_color = self.theme_manager.get_color('success')
                    # Keep text_color as status_text
                elif state == 'error':
                    bg_color = self.theme_manager.get_color('error')
                    # Keep text_color as status_text
                elif state == 'warning':
                    bg_color = self.theme_manager.get_color('warning')
                    # For warning in dark theme, we need better contrast
                    if self.theme_manager.current_theme == 'dark':
                        text_color = 'black'
            else:
                # Use default colors from STATES
                bg_color = self.STATES[state]['bg']
                text_color = self.STATES[state]['fg']
        except Exception as e:
            # Fallback to default colors if there's any issue with theme colors
            print(f"Error getting color for state {state}: {e}")
            bg_color = self.STATES[state]['bg']
            text_color = self.STATES[state]['fg']
        
        # Update indicator circle using tag instead of item ID
        self.indicator.delete("all")
        # Add a light outline for better definition
        outline_color = self._adjust_color(bg_color, -20)
        self.indicator.create_oval(2, 2, 12, 12, fill=bg_color, outline=outline_color, tags="indicator", width=1)
        
        # Update text with the appropriate color for contrast
        self.label.config(text=self.STATES[state]['text'], foreground=text_color)
        
        # Start animation if processing or loading
        if state in ['loading', 'processing']:
            self._start_animation()
        else:
            self._stop_animation()
    
    def _adjust_color(self, color, amount):
        """
        Adjust a color by the given amount to make it lighter or darker.
        
        Parameters
        ----------
        color : str
            Hex color code
        amount : int
            Amount to adjust (-255 to 255)
            
        Returns
        -------
        str
            Adjusted hex color code
        """
        try:
            # Convert hex to RGB
            if color.startswith('#'):
                color = color[1:]
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            
            # Adjust colors
            r = max(0, min(255, r + amount))
            g = max(0, min(255, g + amount))
            b = max(0, min(255, b + amount))
            
            # Convert back to hex
            return f'#{r:02x}{g:02x}{b:02x}'
        except Exception:
            # Return original color if there's an error
            return f'#{color}' if not color.startswith('#') else color
    
    def _start_animation(self):
        """Start the indicator animation."""
        self._animation_running = True
        self._animate()
    
    def _stop_animation(self):
        """Stop the indicator animation."""
        self._animation_running = False
    
    def _animate(self):
        """Animate the indicator."""
        if not self._animation_running:
            return
        
        try:
            # Check if the canvas still exists and has the oval
            if not self.indicator.winfo_exists():
                self._animation_running = False
                return
                
            # Get the indicator item by tag
            indicator_items = self.indicator.find_withtag("indicator")
            if not indicator_items:
                self._animation_running = False
                return
            
            indicator_item = indicator_items[0]
            
            # Get current color (safely)
            try:
                current_color = self.indicator.itemcget(indicator_item, 'fill')
                if not current_color:  # If empty or None
                    current_color = "#6c8eb3"  # Default to a safe blue color
            except:
                current_color = "#6c8eb3"  # Default to a safe blue color
            
            # Safely get RGB values
            try:
                # Simple blink/pulse effect
                current_stipple = self.indicator.itemcget(indicator_item, 'stipple')
                new_stipple = '' if current_stipple else 'gray50'
                self.indicator.itemconfig(indicator_item, stipple=new_stipple)
            except Exception as e:
                print(f"Error in animation effect: {e}")
            
            # Schedule next animation frame
            self.after(500, self._animate)
        except Exception as e:
            print(f"Animation error: {e}")
            self._animation_running = False
    
    def set_text(self, text):
        """
        Set custom text for the status indicator.
        
        Parameters
        ----------
        text : str
            Text to display
        """
        self.label.config(text=text) 