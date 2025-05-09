"""
Theme manager for the Peak Analysis Tool.

This module handles all styling, colors, and theme configuration
for consistent UI appearance across the application.
"""

import tkinter as tk
from tkinter import ttk
import platform

class ThemeManager:
    """
    Manages the application's visual theme and style.
    
    This class provides methods for applying consistent styling
    to various UI elements across the application.
    """
    
    # Light theme color palette - Updated with more neutral, understated colors
    LIGHT_COLORS = {
        'primary': '#505a64',      # Muted slate gray for main elements
        'secondary': '#6e7278',    # More neutral gray (less blue tint)
        'success': '#6c7c67',      # More desaturated green for success messages
        'warning': '#8a8070',      # More neutral tan for warnings
        'error': '#8a7272',        # More desaturated red for errors
        'info': '#6c7580',         # More desaturated blue-gray for information
        'background': '#f5f5f5',   # Neutral light gray for backgrounds (removed blue tint)
        'text': '#343a40',         # Dark gray for text
        'text_secondary': '#6c757d', # Secondary text color
        'button': '#505a64',       # Match primary color for buttons
        'button_pressed': '#3f474f', # Darker gray for pressed buttons
        'canvas_bg': '#ffffff',    # White for plot backgrounds
        'panel_bg': '#eeeeee',     # Neutral light gray for panels (removed blue tint)
        'card_bg': '#ffffff',      # White for cards 
        'border': '#d0d0d0',       # Neutral light gray for borders
        'highlight': '#6e7278',    # Neutral highlight color - same as secondary
        'status_text': '#343a40'   # Darker text for status messages
    }
    
    # Dark theme color palette - Updated for less colorful, more subtle look
    DARK_COLORS = {
        'primary': '#3a4653',      # More neutral dark slate (less blue)
        'secondary': '#545c66',    # More neutral mid-gray (less color)
        'success': '#556159',      # More desaturated green for success
        'warning': '#6d6353',      # More desaturated amber for warnings
        'error': '#6c5252',        # More desaturated red for errors
        'info': '#4d5c6d',         # More desaturated blue for information
        'background': '#1e1e1e',   # Dark gray for main background
        'card_bg': '#2d2d2d',      # Slightly lighter gray for cards/panels
        'border': '#3d3d3d',       # Border color
        'text': '#e0e0e0',         # Light gray for text
        'text_secondary': '#9e9e9e', # Secondary text color
        'button': '#3a4653',       # Match primary color for buttons
        'button_pressed': '#2c3740', # Darker slate for pressed buttons
        'canvas_bg': '#ffffff',    # White for plot backgrounds - KEEP PLOTS LIGHT
        'panel_bg': '#2d2d2d',     # Slightly lighter gray for panels
        'highlight': '#545c66',    # Neutral highlight color - same as secondary
        'status_text': '#ffffff'   # White text for status messages
    }
    
    # Store semantic plot colors separately
    LIGHT_SEMANTIC_PLOT_COLORS = {
        'line_raw': '#555555',      # Darker Gray for raw data overlay
        'line_filtered': '#1f77b4', # Default blue for filtered data
        'marker_peak': '#FF0000',   # Red for peak markers
        'marker_width': '#FF0000',  # Red for width lines
        'line_dist_min': '#FF6B6B',  # Light Red
        'line_dist_max': '#4ECDC4',  # Teal
        'span_dist': '#C8E6C9',      # Light Green
        'marker_double': '#1f77b4',  # Blue (match filtered)
        'marker_nondouble': '#AAAAAA', # Light Gray (match raw)
        'moving_average': '#FF8C00', # Dark Orange for moving average
        'scatter_points': 'black', # Default blue for scatter
        'hist_bars': '#1f77b4',      # Default blue for histogram
    }

    DARK_SEMANTIC_PLOT_COLORS = {
        'line_raw': '#666666',      # Mid-Gray for raw data overlay
        'line_filtered': '#81D4FA', # Light Blue for filtered data
        'marker_peak': '#FF6B6B',   # Lighter Red for peak markers
        'marker_width': '#FF6B6B',  # Lighter Red for width lines
        'line_dist_min': '#FF8A80',  # Lighter Red
        'line_dist_max': '#80CBC4',  # Lighter Teal
        'span_dist': '#385739',      # Dark Green
        'marker_double': '#81D4FA',  # Light Blue (match filtered)
        'marker_nondouble': '#666666', # Mid Gray (match raw)
        'moving_average': '#FFA726', # Lighter Orange for moving average
        'scatter_points': '#81D4FA', # Light Blue for scatter
        'hist_bars': '#81D4FA',      # Light Blue for histogram
    }

    # --- Standard Matplotlib rcParams ---
    LIGHT_PLOT_RCParams = {
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#ffffff',
        'savefig.facecolor': '#ffffff',
        'text.color': '#333333',
        'axes.labelcolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'axes.edgecolor': '#666666',
        'grid.color': '#cccccc',
        'grid.alpha': 0.5,
        'lines.color': LIGHT_SEMANTIC_PLOT_COLORS['line_filtered'], # Default line color
        'patch.edgecolor': '#333333'
    }

    DARK_PLOT_RCParams = {
        'figure.facecolor': '#1e1e1e',
        'axes.facecolor': '#2d2d2d',
        'savefig.facecolor': '#1e1e1e',
        'text.color': '#e0e0e0',
        'axes.labelcolor': '#e0e0e0',
        'xtick.color': '#e0e0e0',
        'ytick.color': '#e0e0e0',
        'axes.edgecolor': '#9e9e9e',
        'grid.color': '#545c66',
        'grid.alpha': 0.3,
        'lines.color': DARK_SEMANTIC_PLOT_COLORS['line_filtered'], # Default line color
        'patch.edgecolor': '#e0e0e0'
    }
    # --- End Standard Matplotlib rcParams ---
    
    # Font configurations
    FONTS = {
        'default': ('Segoe UI', 10),
        'heading': ('Segoe UI', 12, 'bold'),
        'subheading': ('Segoe UI', 11, 'bold'),
        'monospace': ('Consolas', 10),
        'small': ('Segoe UI', 9)
    }
    
    def __init__(self, theme_name='light'):
        """
        Initialize the theme manager with default settings.
        
        Parameters
        ----------
        theme_name : str, optional
            Name of the theme to use ('light' or 'dark')
        """
        self.set_theme(theme_name)
        
        # Adjust fonts based on platform
        if platform.system() == 'Darwin':  # macOS
            self.FONTS = {
                'default': ('SF Pro Text', 12),
                'heading': ('SF Pro Display', 14, 'bold'),
                'subheading': ('SF Pro Display', 13, 'bold'),
                'monospace': ('SF Mono', 12),
                'small': ('SF Pro Text', 11)
            }
        elif platform.system() == 'Linux':
            self.FONTS = {
                'default': ('Ubuntu', 10),
                'heading': ('Ubuntu', 12, 'bold'),
                'subheading': ('Ubuntu', 11, 'bold'),
                'monospace': ('Ubuntu Mono', 10),
                'small': ('Ubuntu', 9)
            }
    
    def set_theme(self, theme_name):
        """
        Set the current theme.
        
        Parameters
        ----------
        theme_name : str
            Name of the theme to use ('light' or 'dark')
        """
        self.current_theme = theme_name.lower()
        if self.current_theme == 'dark':
            self.COLORS = self.DARK_COLORS
            self.PLOT_COLORS = self.DARK_SEMANTIC_PLOT_COLORS # Store semantic colors
            self.PLOT_RCParams = self.DARK_PLOT_RCParams      # Store standard rcParams
        else:
            self.COLORS = self.LIGHT_COLORS
            self.PLOT_COLORS = self.LIGHT_SEMANTIC_PLOT_COLORS # Store semantic colors
            self.PLOT_RCParams = self.LIGHT_PLOT_RCParams      # Store standard rcParams
    
    def toggle_theme(self):
        """
        Toggle between light and dark themes.
        
        Returns
        -------
        str
            Name of the new theme ('light' or 'dark')
        """
        if self.current_theme == 'light':
            self.set_theme('dark')
        else:
            self.set_theme('light')
        return self.current_theme
    
    def apply_theme(self, root):
        """
        Apply the selected theme to the application.
        
        Parameters
        ----------
        root : tk.Tk
            The root Tkinter window
        """
        style = ttk.Style(root)
        
        if self.current_theme == 'dark':
            self._apply_dark_theme(style)
        else:
            self._apply_light_theme(style)
            
        # Configure common elements
        root.configure(background=self.COLORS['background'])
        
        # Update matplotlib style if it's installed
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default') # Reset to default first
            # Apply ONLY standard, valid rcParams
            plt.rcParams.update(self.PLOT_RCParams)
        except ImportError:
            pass  # matplotlib not installed
        except Exception as e:
             print(f"Error applying matplotlib rcParams: {e}") # Add error logging
            
        return style
    
    def _apply_light_theme(self, style):
        """Apply the light theme to all ttk widgets."""
        style.theme_use('clam')  # Start with clam as base
        
        # Configure the main colors
        style.configure('.',
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            font=self.FONTS['default']
        )
        
        # Add Accent Frame style for light theme
        style.configure('Accent.TFrame', 
                      background='#e6f2ff',  # Light blue accent
                      borderwidth=1, 
                      relief='solid',
                      bordercolor=self.COLORS['border'])
        
        # Configure TButton
        style.configure('TButton',
            background=self.COLORS['button'],
            foreground='white',
            borderwidth=1,
            bordercolor=self.COLORS['border'],
            padding=6,
            font=self.FONTS['default']
        )
        style.map('TButton',
            background=[('active', self.COLORS['button_pressed']), 
                        ('pressed', self.COLORS['button_pressed'])],
            relief=[('pressed', 'sunken')]
        )
        
        # Configure primary action button
        style.configure('Primary.TButton',
            background=self.COLORS['primary'],
            foreground='white'
        )
        style.map('Primary.TButton',
            background=[('active', '#414a52'), ('pressed', '#414a52')]
        )
        
        # Configure success button
        style.configure('Success.TButton',
            background=self.COLORS['success'],
            foreground='white'
        )
        style.map('Success.TButton',
            background=[('active', '#698a5e'), ('pressed', '#698a5e')]
        )
        
        # Configure TLabel
        style.configure('TLabel', 
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            font=self.FONTS['default']
        )
        
        # Configure heading labels
        style.configure('Heading.TLabel',
            font=self.FONTS['heading'],
            foreground=self.COLORS['primary']
        )
        
        # Configure TFrame
        style.configure('TFrame',
            background=self.COLORS['background'],
            borderwidth=0
        )
        
        # Configure TLabelframe
        style.configure('TLabelframe',
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            borderwidth=1,
            bordercolor=self.COLORS['border'],
            relief='solid'
        )
        style.configure('TLabelframe.Label',
            background=self.COLORS['background'],
            foreground=self.COLORS['primary'],
            font=self.FONTS['subheading']
        )
        
        # Configure TNotebook
        style.configure('TNotebook',
            background=self.COLORS['background'],
            borderwidth=1,
            bordercolor=self.COLORS['border'],
            tabmargins=[2, 5, 2, 0]
        )
        style.configure('TNotebook.Tab',
            background=self.COLORS['panel_bg'],
            foreground=self.COLORS['text'],
            padding=[10, 4],
            font=self.FONTS['default']
        )
        style.map('TNotebook.Tab',
            background=[('selected', self.COLORS['primary'])],
            foreground=[('selected', 'white')],
            expand=[('selected', [1, 1, 1, 0])]
        )
        
        # Configure Progressbar
        style.configure('TProgressbar',
            background=self.COLORS['secondary'],
            troughcolor=self.COLORS['panel_bg'],
            borderwidth=0,
            thickness=8
        )
        
        # Configure Success Progressbar
        style.configure('Success.TProgressbar',
            background=self.COLORS['success']
        )
        
        # Configure Scrollbar
        style.configure('TScrollbar',
            background=self.COLORS['background'],
            troughcolor=self.COLORS['panel_bg'],
            borderwidth=1,
            bordercolor=self.COLORS['border'],
            arrowsize=14
        )
        
        # Configure TEntry
        style.configure('TEntry',
            fieldbackground='white',
            foreground=self.COLORS['text'],
            bordercolor=self.COLORS['border'],
            lightcolor=self.COLORS['card_bg'],
            darkcolor=self.COLORS['border'],
            borderwidth=1,
            padding=5,
            font=self.FONTS['default'],
            insertcolor=self.COLORS['text']
        )
        style.map('TEntry',
            fieldbackground=[('readonly', self.COLORS['background'])],
            bordercolor=[('focus', self.COLORS['highlight'])]
        )
        
        # Configure Treeview
        style.configure('Treeview',
            background='white',
            foreground=self.COLORS['text'],
            fieldbackground='white',
            bordercolor=self.COLORS['border'],
            borderwidth=1,
            rowheight=24,
            font=self.FONTS['default']
        )
        style.configure('Treeview.Heading',
            background=self.COLORS['panel_bg'],
            foreground=self.COLORS['primary'],
            borderwidth=1,
            bordercolor=self.COLORS['border'],
            font=self.FONTS['subheading']
        )
        style.map('Treeview',
            background=[('selected', self.COLORS['secondary'])],
            foreground=[('selected', 'white')]
        )
        
        # Configure Radio Buttons - Fix the dark background issue
        style.configure('TRadiobutton', 
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            font=self.FONTS['default'],
            indicatorcolor=self.COLORS['border'],
            indicatorbackground='white'
        )
        style.map('TRadiobutton',
            background=[('active', self.COLORS['background']), 
                        ('pressed', self.COLORS['background']), 
                        ('selected', self.COLORS['background'])],
            foreground=[('disabled', self.COLORS['text_secondary'])],
            indicatorcolor=[('selected', self.COLORS['secondary'])]
        )
        
        # Configure Checkbuttons
        style.configure('TCheckbutton', 
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            font=self.FONTS['default'],
            indicatorcolor=self.COLORS['border'],
            indicatorbackground='white'
        )
        style.map('TCheckbutton',
            background=[('active', self.COLORS['background']), 
                        ('pressed', self.COLORS['background']), 
                        ('selected', self.COLORS['background'])],
            foreground=[('disabled', self.COLORS['text_secondary'])],
            indicatorcolor=[('selected', self.COLORS['secondary'])]
        )
        
        # Configure Combobox
        style.configure('TCombobox',
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            fieldbackground='white',
            bordercolor=self.COLORS['border'],
            arrowcolor=self.COLORS['text'],
            padding=5
        )
        style.map('TCombobox',
            fieldbackground=[('readonly', 'white'), ('disabled', self.COLORS['panel_bg'])],
            bordercolor=[('focus', self.COLORS['highlight'])]
        )
        
    def _apply_dark_theme(self, style):
        """Apply the dark theme to all ttk widgets."""
        style.theme_use('clam')  # Start with clam as base
        
        # === MAIN ELEMENTS ===
        # Configure the main colors
        style.configure('.',
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            font=self.FONTS['default']
        )
        
        # Add Accent Frame style for dark theme
        style.configure('Accent.TFrame', 
                      background=self.COLORS['panel_bg'],  # Use panel background for dark accent
                      borderwidth=1, 
                      relief='solid',
                      bordercolor=self.COLORS['border'])
        
        # === BUTTONS ===
        # Standard button
        style.configure('TButton',
            background=self.COLORS['button'],
            foreground='white',
            borderwidth=0,
            padding=6,
            font=self.FONTS['default']
        )
        style.map('TButton',
            background=[('active', self.COLORS['button_pressed']), 
                        ('pressed', self.COLORS['button_pressed'])],
            foreground=[('active', 'white'), ('pressed', 'white')],
            relief=[('pressed', 'flat')]
        )
        
        # Primary action button
        style.configure('Primary.TButton',
            background=self.COLORS['primary'],
            foreground='white'
        )
        style.map('Primary.TButton',
            background=[('active', '#0d48ac'), ('pressed', '#0a3880')],
            foreground=[('active', 'white'), ('pressed', 'white')]
        )
        
        # Success button
        style.configure('Success.TButton',
            background=self.COLORS['success'],
            foreground='white'
        )
        style.map('Success.TButton',
            background=[('active', '#3d9140'), ('pressed', '#357a37')],
            foreground=[('active', 'white'), ('pressed', 'white')]
        )
        
        # === LABELS ===
        # Standard label
        style.configure('TLabel', 
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            font=self.FONTS['default']
        )
        
        # Heading labels
        style.configure('Heading.TLabel',
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            font=self.FONTS['heading']
        )
        
        # === FRAMES ===
        # Standard frame
        style.configure('TFrame',
            background=self.COLORS['background'],
            borderwidth=0
        )
        
        # === LABELFRAMES ===
        # Standard labelframe
        style.configure('TLabelframe',
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            bordercolor=self.COLORS['border'],
            borderwidth=1,
            relief='solid'
        )
        style.configure('TLabelframe.Label',
            background=self.COLORS['background'],
            foreground=self.COLORS['secondary'],
            font=self.FONTS['subheading']
        )
        
        # === NOTEBOOK TABS ===
        # Tab control
        style.configure('TNotebook',
            background=self.COLORS['background'],
            bordercolor=self.COLORS['border'],
            tabmargins=[2, 5, 2, 0]
        )
        style.configure('TNotebook.Tab',
            background=self.COLORS['panel_bg'],
            foreground=self.COLORS['text'],
            padding=[10, 4],
            font=self.FONTS['default']
        )
        style.map('TNotebook.Tab',
            background=[('selected', self.COLORS['primary'])],
            foreground=[('selected', 'white')],
            expand=[('selected', [1, 1, 1, 0])]
        )
        
        # === PROGRESSBAR ===
        # Standard progressbar
        style.configure('TProgressbar',
            background=self.COLORS['secondary'],
            troughcolor=self.COLORS['panel_bg'],
            borderwidth=0,
            thickness=8
        )
        
        # Success progressbar
        style.configure('Success.TProgressbar',
            background=self.COLORS['success']
        )
        
        # === SCROLLBAR ===
        style.configure('TScrollbar',
            background=self.COLORS['panel_bg'],
            troughcolor=self.COLORS['background'],
            bordercolor=self.COLORS['border'],
            arrowcolor=self.COLORS['text'],
            borderwidth=1,
            arrowsize=14
        )
        style.map('TScrollbar',
            background=[('active', self.COLORS['primary']), 
                       ('pressed', self.COLORS['primary'])],
            arrowcolor=[('active', 'white'), ('pressed', 'white')]
        )
        
        # === ENTRY FIELDS ===
        style.configure('TEntry',
            fieldbackground=self.COLORS['card_bg'],
            foreground=self.COLORS['text'],
            bordercolor=self.COLORS['border'],
            lightcolor=self.COLORS['card_bg'],
            darkcolor=self.COLORS['card_bg'],
            borderwidth=1,
            padding=5,
            font=self.FONTS['default'],
            insertcolor=self.COLORS['text']
        )
        style.map('TEntry',
            fieldbackground=[('readonly', self.COLORS['background']), 
                            ('disabled', self.COLORS['background'])],
            foreground=[('readonly', self.COLORS['text_secondary']), 
                       ('disabled', self.COLORS['text_secondary'])],
            bordercolor=[('focus', self.COLORS['secondary'])]
        )
        
        # === COMBOBOX ===
        style.configure('TCombobox',
            fieldbackground=self.COLORS['card_bg'],
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],
            arrowcolor=self.COLORS['text'],
            bordercolor=self.COLORS['border'],
            lightcolor=self.COLORS['card_bg'],
            darkcolor=self.COLORS['card_bg']
        )
        style.map('TCombobox',
            fieldbackground=[('readonly', self.COLORS['card_bg']), 
                            ('disabled', self.COLORS['background'])],
            foreground=[('readonly', self.COLORS['text']), 
                       ('disabled', self.COLORS['text_secondary'])],
            bordercolor=[('focus', self.COLORS['secondary'])]
        )
        
        # === RADIO BUTTONS ===
        print(f"--- Applying Dark Theme TRadiobutton (Label Styling Focus) ---")
        style.configure('TRadiobutton',
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],         # Default text color (dim)
            indicatorcolor=self.COLORS['border'],   # Default indicator color (dark)
            indicatorbackground=self.COLORS['background'],
            indicatorrelief='flat'
        )
        style.map('TRadiobutton',
            # Change background slightly when selected
            background=[('active', self.COLORS['background']),
                       ('selected', self.COLORS['panel_bg'])], # <--- Subtle BG change when selected
            # Make FOREGROUND bright white when selected
            foreground=[('disabled', self.COLORS['text_secondary']),
                       ('selected', '#FFFFFF')],             # <--- Bright text when selected
            indicatorcolor=[
                ('selected', self.COLORS['text']),          # Try setting indicator selected bright
                ('active', self.COLORS['panel_bg'])         # Hover state for OFF indicator
                # Default uses configure settings
                ]
        )

        # === CHECKBUTTONS ===
        print(f"--- Applying Dark Theme TCheckbutton (Label Styling Focus) ---")
        style.configure('TCheckbutton',
            background=self.COLORS['background'],
            foreground=self.COLORS['text'],         # Default text color (dim)
            indicatorcolor=self.COLORS['border'],   # Default indicator color (dark)
            indicatorbackground=self.COLORS['background'],
            indicatorrelief='flat'
        )
        style.map('TCheckbutton',
             # Change background slightly when selected
            background=[('active', self.COLORS['background']),
                       ('selected', self.COLORS['panel_bg'])], # <--- Subtle BG change when selected
            # Make FOREGROUND bright white when selected
            foreground=[('disabled', self.COLORS['text_secondary']),
                       ('selected', '#FFFFFF')],             # <--- Bright text when selected
            indicatorcolor=[
                ('selected', self.COLORS['text']),          # Try setting indicator selected bright
                ('active', self.COLORS['panel_bg'])         # Hover state for OFF indicator
                # Default uses configure settings
                ]
        )
        
        # === TREEVIEW (for tables) ===
        style.configure('Treeview',
            background=self.COLORS['card_bg'],
            foreground=self.COLORS['text'],
            fieldbackground=self.COLORS['card_bg'],
            bordercolor=self.COLORS['border'],
            borderwidth=1,
            rowheight=24,
            font=self.FONTS['default']
        )
        style.configure('Treeview.Heading',
            background=self.COLORS['panel_bg'],
            foreground=self.COLORS['text'],
            borderwidth=1,
            bordercolor=self.COLORS['border'],
            relief='flat',
            font=self.FONTS['subheading']
        )
        style.map('Treeview',
            background=[('selected', self.COLORS['primary'])],
            foreground=[('selected', 'white')]
        )
        style.map('Treeview.Heading',
            background=[('active', self.COLORS['primary'])],
            foreground=[('active', 'white')]
        )
        
        # === SCALE/SLIDER ===
        style.configure('TScale',
            background=self.COLORS['background'],
            troughcolor=self.COLORS['panel_bg'],
            troughrelief='flat',
            sliderrelief='flat',
            sliderlength=10,
            sliderthickness=20,
            borderwidth=0
        )
        
        # === SEPARATOR ===
        style.configure('TSeparator',
            background=self.COLORS['border']
        )
        
        # === SIZEGRIP (resize handle) ===
        style.configure('TSizegrip',
            background=self.COLORS['background']
        )
    
    def _apply_default_theme(self, style):
        """Apply the default theme (fallback)."""
        style.theme_use('clam')
    
    def get_color(self, color_name):
        """
        Get a color from the color palette.
        
        Parameters
        ----------
        color_name : str
            Name of the color to retrieve
            
        Returns
        -------
        str
            Hex color code
        """
        if color_name not in self.COLORS:
            print(f"Warning: Requested color '{color_name}' not found in theme palette. Using primary color instead.")
            return self.COLORS.get('primary')
        return self.COLORS.get(color_name)
    
    def get_font(self, font_name):
        """
        Get a font configuration from the font palette.
        
        Parameters
        ----------
        font_name : str
            Name of the font configuration to retrieve
            
        Returns
        -------
        tuple
            Font configuration
        """
        return self.FONTS.get(font_name, self.FONTS['default'])

    def get_plot_color(self, semantic_color_name):
        """
        Get a specific SEMANTIC color value for the current plot theme.

        Parameters
        ----------
        semantic_color_name : str
            The semantic name of the color (e.g., 'line_raw', 'marker_peak')

        Returns
        -------
        str
            The color value for the current theme. Returns a default color if not found.
        """
        # PLOT_COLORS should be set by set_theme
        return self.PLOT_COLORS.get(semantic_color_name, '#000000') # Default to black if missing

    def apply_plot_theme(self, fig, axes_list=None):
        """
        Apply the current theme's standard styles (bg, text, grid) directly
        to a Figure and its Axes. Specific element colors (lines, markers)
        must be set using get_plot_color in the plotting functions.
        """
        # Use the standard rcParams dict for the current theme
        plot_rcParams = self.PLOT_RCParams

        fig_face_color = plot_rcParams.get('figure.facecolor')
        ax_face_color = plot_rcParams.get('axes.facecolor')
        text_color = plot_rcParams.get('text.color')
        edge_color = plot_rcParams.get('axes.edgecolor')
        grid_color = plot_rcParams.get('grid.color')
        grid_alpha = plot_rcParams.get('grid.alpha', 0.5) # Provide default alpha

        if fig_face_color:
            fig.patch.set_facecolor(fig_face_color)

        if axes_list is None:
            axes_list = fig.get_axes()

        for ax in axes_list:
            if ax_face_color:
                ax.set_facecolor(ax_face_color)
            if text_color:
                ax.tick_params(axis='x', colors=text_color)
                ax.tick_params(axis='y', colors=text_color)
                ax.xaxis.label.set_color(text_color)
                ax.yaxis.label.set_color(text_color)
                if ax.get_title():
                    ax.title.set_color(text_color)
                 # Also set colorbar label/tick colors if it's a colorbar axis
                if hasattr(ax, 'yaxis') and hasattr(ax.yaxis, 'label'):
                    ax.yaxis.label.set_color(text_color) # For colorbar label

            if edge_color:
                for spine in ax.spines.values():
                    spine.set_edgecolor(edge_color)
            if grid_color:
                ax.grid(True, color=grid_color, alpha=grid_alpha)
            else:
                ax.grid(False) # Explicitly disable grid if no color defined 