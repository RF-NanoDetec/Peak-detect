"""
Custom widgets for the Peak Analysis Tool GUI.
Provides specialized widgets and controls for the application.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Any, Dict
import logging
from dataclasses import dataclass

from ..config import config

logger = logging.getLogger(__name__)

@dataclass
class WidgetConfig:
    """Configuration for widget appearance."""
    padding: int = 5
    entry_width: int = 15
    button_width: int = 20
    label_width: int = 20
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'bg': '#f0f0f0',
                'fg': '#333333',
                'error': '#ff0000',
                'success': '#00ff00',
                'warning': '#ffa500'
            }

class ToolTip:
    """Create tooltips for widgets."""
    
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        
        self.widget.bind('<Enter>', self.show_tooltip)
        self.widget.bind('<Leave>', self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        """Display the tooltip."""
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(
            self.tooltip_window,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            padding=(5, 2)
        )
        label.pack()
    
    def hide_tooltip(self, event=None):
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

class LabeledEntry(ttk.Frame):
    """Entry widget with label and optional validation."""
    
    def __init__(
        self,
        master,
        label: str,
        variable: tk.Variable,
        validator: Optional[Callable[[str], bool]] = None,
        tooltip: Optional[str] = None,
        width: int = 15,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self.variable = variable
        self.validator = validator
        
        # Create label
        self.label = ttk.Label(self, text=label)
        self.label.pack(side=tk.LEFT, padx=5)
        
        # Create entry
        validate_command = self.register(self._validate) if validator else None
        self.entry = ttk.Entry(
            self,
            textvariable=variable,
            width=width,
            validate='focusout',
            validatecommand=(validate_command, '%P') if validate_command else None
        )
        self.entry.pack(side=tk.LEFT, padx=5)
        
        # Add tooltip if provided
        if tooltip:
            ToolTip(self.entry, tooltip)
    
    def _validate(self, value: str) -> bool:
        """Validate entry value."""
        try:
            is_valid = self.validator(value)
            self.entry.state(['!invalid'] if is_valid else ['invalid'])
            return is_valid
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            self.entry.state(['invalid'])
            return False

class ProgressBar(ttk.Frame):
    """Custom progress bar with label."""
    
    def __init__(
        self,
        master,
        label: str = "Progress:",
        mode: str = 'determinate',
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self.label = ttk.Label(self, text=label)
        self.label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(
            self,
            mode=mode,
            length=200
        )
        self.progress.pack(side=tk.LEFT, padx=5)
        
        self.percentage = ttk.Label(self, text="0%")
        self.percentage.pack(side=tk.LEFT, padx=5)
    
    def set_progress(self, value: float):
        """Update progress bar value."""
        self.progress['value'] = value
        self.percentage['text'] = f"{int(value)}%"
        self.update_idletasks()

class ScrolledFrame(ttk.Frame):
    """Frame with scrollbars."""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Create canvas
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw"
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack widgets
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

class StatusBar(ttk.Frame):
    """Status bar with message and progress."""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.message = ttk.Label(
            self,
            text="Ready",
            padding=(5, 2)
        )
        self.message.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress = ProgressBar(self)
        self.progress.pack(side=tk.RIGHT, padx=5)
    
    def set_message(self, text: str, message_type: str = 'info'):
        """Update status message."""
        colors = {
            'info': 'black',
            'error': 'red',
            'success': 'green',
            'warning': 'orange'
        }
        self.message['foreground'] = colors.get(message_type, 'black')
        self.message['text'] = text

class ParameterFrame(ttk.LabelFrame):
    """Frame for analysis parameters."""
    
    def __init__(
        self,
        master,
        title: str = "Parameters",
        **kwargs
    ):
        super().__init__(master, text=title, **kwargs)
        
        self.entries: Dict[str, LabeledEntry] = {}
    
    def add_parameter(
        self,
        name: str,
        variable: tk.Variable,
        validator: Optional[Callable[[str], bool]] = None,
        tooltip: Optional[str] = None
    ):
        """Add a parameter entry."""
        entry = LabeledEntry(
            self,
            label=name.replace('_', ' ').title(),
            variable=variable,
            validator=validator,
            tooltip=tooltip
        )
        entry.pack(fill=tk.X, padx=5, pady=2)
        self.entries[name] = entry
    
    def get_values(self) -> Dict[str, Any]:
        """Get all parameter values."""
        return {
            name: entry.variable.get()
            for name, entry in self.entries.items()
        }

class FileSelector(ttk.Frame):
    """File selection widget with browse button."""
    
    def __init__(
        self,
        master,
        label: str = "File:",
        file_types: Optional[list] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self.file_types = file_types or [("All files", "*.*")]
        
        # Create widgets
        ttk.Label(self, text=label).pack(side=tk.LEFT, padx=5)
        
        self.path_var = tk.StringVar()
        self.entry = ttk.Entry(
            self,
            textvariable=self.path_var,
            width=40
        )
        self.entry.pack(side=tk.LEFT, padx=5)
        
        self.browse_button = ttk.Button(
            self,
            text="Browse...",
            command=self.browse
        )
        self.browse_button.pack(side=tk.LEFT, padx=5)
    
    def browse(self):
        """Open file browser dialog."""
        from tkinter import filedialog
        
        filename = filedialog.askopenfilename(
            filetypes=self.file_types
        )
        if filename:
            self.path_var.set(filename)
    
    def get_path(self) -> str:
        """Get selected file path."""
        return self.path_var.get()

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Widget Demo")
    
    # Create parameter frame
    params = ParameterFrame(root, "Analysis Parameters")
    params.pack(fill=tk.X, padx=10, pady=5)
    
    # Add parameters
    params.add_parameter(
        "threshold",
        tk.DoubleVar(value=1.0),
        lambda x: float(x) > 0,
        "Minimum peak height"
    )
    
    # Create file selector
    file_select = FileSelector(
        root,
        "Data File:",
        [("Text files", "*.txt"), ("All files", "*.*")]
    )
    file_select.pack(fill=tk.X, padx=10, pady=5)
    
    # Create status bar
    status = StatusBar(root)
    status.pack(fill=tk.X, side=tk.BOTTOM)
    
    root.mainloop()
