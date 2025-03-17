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
            app.data['Time - Plot 0'].values * 1e-4 / 60,  # Convert to minutes
            app.data['Amplitude - Plot 0'].values
        )
        
        # Update progress
        app.update_progress_bar(2)
        
        # Plot decimated data
        ax.plot(t_plot, x_plot,
                color='black',
                linewidth=0.05,
                label=f'Raw Data ({len(t_plot):,} points)',
                alpha=0.9)
        
        # Customize plot
        ax.set_xlabel('Time (min)', fontsize=12)
        ax.set_ylabel('Amplitude (counts)', fontsize=12)
        ax.set_title('Raw Data (Optimized View)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)
        
        # Add data statistics annotation
        stats_text = (f'Total points: {len(app.data):,}\n'
                     f'Plotted points: {len(t_plot):,}\n'
                     f'Mean: {np.mean(x_plot):.1f}\n'
                     f'Std: {np.std(x_plot):.1f}')
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout
        app.figure.tight_layout()
        
        # Update or create tab
        tab_exists = False
        for tab in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab, "text") == "Raw Data":
                app.plot_tab_control.select(tab)
                tab_exists = True
                break
        
        if not tab_exists:
            new_tab = ttk.Frame(app.plot_tab_control)
            app.plot_tab_control.add(new_tab, text="Raw Data")
            app.plot_tab_control.select(new_tab)
            canvas = FigureCanvasTkAgg(app.figure, new_tab)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update the canvas
        app.canvas.draw_idle()
        
        # Final progress update
        app.update_progress_bar(3)
        
        # Update status
        app.preview_label.config(
            text=f"Raw data plotted successfully (Decimated from {len(app.data):,} to {len(t_plot):,} points)",
            foreground="green"
        )

        app.tab_figures["Raw Data"] = app.figure

    except Exception as e:
        app.preview_label.config(text=f"Error plotting raw data: {str(e)}", foreground="red")
        print(f"Detailed error: {str(e)}")
        traceback.print_exc() 