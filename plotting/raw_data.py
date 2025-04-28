"""
Raw data visualization functions for the Peak Analysis Tool.

This module provides functions for visualizing raw data in the application.
It follows the reorganized structure to separate visualization concerns from
data processing and analysis.
"""

import numpy as np
import traceback
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from matplotlib.lines import Line2D

# Import utilities from reorganized structure
from core.performance import profile_function as profile_func

def plot_raw_data(app, profiler=None):
    """
    Optimized plotting of raw data.
    
    This function visualizes the raw data in an optimized way by 
    decimating the points for display while showing statistics
    about the full dataset.
    
    Parameters:
    -----------
    app : Application instance
        The main application instance containing necessary attributes and methods
    profiler : function, optional
        Profiling decorator, defaults to the default profiler from core.performance
        
    Returns:
    --------
    None
        The function updates the application's figure and canvas directly
        
    Notes:
    ------
    This function is part of the plotting package in the reorganized
    structure. It interacts with the application's UI elements to
    display the plot in the appropriate tab.
    """
    # Use the default profiler if none is provided
    if profiler is None:
        profiler = profile_func
        
    if app.data is None:
        app.preview_label.config(text="No data to plot", foreground="red")
        return

    try:
        # Initialize progress
        app.update_progress_bar(0, 3)
        
        # Create new figure if needed
        if app.canvas is None:
            app.canvas = FigureCanvasTkAgg(app.figure, app.plot_tab_control)
        
        # Clear the current figure
        app.figure.clear()
        ax = app.figure.add_subplot(111)
        
        # Update progress
        app.update_progress_bar(1)
        
        # Decimate data for plotting
        t_plot, x_plot = app.decimate_for_plot(
            app.data['Time - Plot 0'].values / 60,  # Convert to minutes (time already in seconds)
            app.data['Amplitude - Plot 0'].values
        )
        
        # Update progress
        app.update_progress_bar(2)
        
        # Get SEMANTIC theme color
        raw_line_color = app.theme_manager.get_plot_color('line_raw')

        # Plot decimated data using SEMANTIC theme color with thin line
        ax.plot(t_plot, x_plot,
                color=raw_line_color,
                linewidth=0.05,
                label=None,  # Remove label from actual plot
                alpha=0.9)
        
        # Create custom legend with thicker line
        legend_line = Line2D([0], [0], color=raw_line_color, linewidth=2.0, 
                            label=f'Raw Data ({len(t_plot):,} points)')
        ax.legend(handles=[legend_line])
        
        # Customize plot (fonts, etc., handled by apply_plot_theme)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Amplitude (counts)')
        ax.set_title('Raw Data (Optimized View)')
        ax.grid(True, linestyle='--') # Grid color/alpha handled by apply_plot_theme
        
        # Add data statistics annotation
        stats_text = (f'Total points: {len(app.data):,}\n'
                     f'Plotted points: {len(t_plot):,}\n'
                     f'Mean: {np.mean(x_plot):.1f}\n'
                     f'Std: {np.std(x_plot):.1f}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8)
        
        # Adjust layout
        app.figure.tight_layout()
        
        # Apply theme standard styles (bg, grid, text)
        app.theme_manager.apply_plot_theme(app.figure, [ax])
        
        # Update or create tab
        tab_exists = False
        raw_data_tab_name = "Raw Data"
        for tab_widget_id in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab_widget_id, "text") == raw_data_tab_name:
                tab_frame = app.plot_tab_control.nametowidget(tab_widget_id)
                app.plot_tab_control.select(tab_frame)
                 # Remove old canvas if exists
                for widget in tab_frame.winfo_children():
                    widget.destroy()
                # Add new canvas
                canvas = FigureCanvasTkAgg(app.figure, master=tab_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                app.canvas = canvas # Update app.canvas reference
                tab_exists = True
                break
        
        if not tab_exists:
            new_tab = ttk.Frame(app.plot_tab_control)
            app.plot_tab_control.add(new_tab, text=raw_data_tab_name)
            app.plot_tab_control.select(new_tab)
            canvas = FigureCanvasTkAgg(app.figure, master=new_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            app.canvas = canvas # Store new canvas reference
        
        # Store the figure associated with the tab
        app.tab_figures[raw_data_tab_name] = app.figure

        # Final progress update
        app.update_progress_bar(3)
        
        # Update status
        app.preview_label.config(
            text=f"Raw data plotted successfully (Decimated from {len(app.data):,} to {len(t_plot):,} points)",
            foreground=app.theme_manager.get_color('success')
        )

    except Exception as e:
        app.preview_label.config(text=f"Error plotting raw data: {str(e)}",
                                  foreground=app.theme_manager.get_color('error'))
        traceback.print_exc() 