"""
Double peak analysis functions for the Peak Analysis Tool.

This module contains functions for detecting and visualizing double peaks in data.
"""

import numpy as np
import traceback
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from matplotlib.lines import Line2D
from core.peak_analysis_utils import find_peaks_with_window

def find_double_peaks(peaks, properties, parameters, time_resolution):
    """
    Calculate distances between all consecutive peaks and identify which peaks
    form part of a double peak pair based on the criteria.
    
    Parameters
    ----------
    peaks : array
        Array of peak indices
    properties : dict
        Dictionary of peak properties from find_peaks_with_window
    parameters : dict
        Dictionary of parameters for double peak detection:
        - min_distance: minimum time between peaks (seconds)
        - max_distance: maximum time between peaks (seconds)
        - min_amp_ratio: minimum ratio of secondary to primary peak amplitude
        - max_amp_ratio: maximum ratio of secondary to primary peak amplitude
        - min_width_ratio: minimum ratio of secondary to primary peak width
        - max_width_ratio: maximum ratio of secondary to primary peak width
    time_resolution : float
        Time resolution for converting indices to seconds
        
    Returns
    -------
    dict
        Dictionary containing information for all peaks:
        - 'peak_indices': indices of all peaks
        - 'next_peak_indices': indices of next peaks
        - 'distances': distances to next peaks (in seconds)
        - 'amp_ratios': amplitude ratios (next/current)
        - 'width_ratios': width ratios (next/current)
        - 'is_double_peak': boolean array indicating which peaks meet criteria
    """
    print("Calculating distances between all consecutive peaks...")
    
    if len(peaks) < 2:
        print("Not enough peaks to calculate distances")
        return {
            'peak_indices': np.array([]),
            'next_peak_indices': np.array([]),
            'distances': np.array([]),
            'amp_ratios': np.array([]),
            'width_ratios': np.array([]),
            'is_double_peak': np.array([])
        }
    
    # Get parameters
    min_distance = parameters.get('min_distance', 0.001)  # Default: 1ms
    max_distance = parameters.get('max_distance', 0.050)  # Default: 50ms
    min_amp_ratio = parameters.get('min_amp_ratio', 0.1)  # Default: 0.1 (10%)
    max_amp_ratio = parameters.get('max_amp_ratio', 2.0)  # Default: 0.9 (90%)
    min_width_ratio = parameters.get('min_width_ratio', 0.1)  # Default: 0.1 (10%)
    max_width_ratio = parameters.get('max_width_ratio', 2.0)  # Default: 2.0 (200%)
    
    print(f"Double peak detection parameters:")
    print(f"  Distance range: {min_distance*1000:.1f} - {max_distance*1000:.1f} ms")
    print(f"  Amplitude ratio range: {min_amp_ratio:.2f} - {max_amp_ratio:.2f}")
    print(f"  Width ratio range: {min_width_ratio:.2f} - {max_width_ratio:.2f}")
    
    # Arrays to store results for all peaks except the last one (which has no next peak)
    peak_indices = []
    next_peak_indices = []
    distances = []
    amp_ratios = []
    width_ratios = []
    is_double_peak = []
    
    # Calculate information for each peak and its next peak
    for i in range(len(peaks) - 1):
        current_peak = peaks[i]
        next_peak = peaks[i+1]
        
        # Calculate peak distance in seconds
        peak_distance = (next_peak - current_peak) * time_resolution
        
        # Get peak properties
        current_amp = properties['prominences'][i]
        next_amp = properties['prominences'][i+1]
        current_width = properties['widths'][i]
        next_width = properties['widths'][i+1]
        
        # Calculate ratios (next to current)
        amp_ratio = next_amp / current_amp if current_amp > 0 else 0
        width_ratio = next_width / current_width if current_width > 0 else 0
        
        # Store the data
        peak_indices.append(i)
        next_peak_indices.append(i+1)
        distances.append(peak_distance)
        amp_ratios.append(amp_ratio)
        width_ratios.append(width_ratio)
        
        # Check if this peak meets the double peak criteria
        meets_criteria = (
            min_distance <= peak_distance <= max_distance and
            min_amp_ratio <= amp_ratio <= max_amp_ratio and
            min_width_ratio <= width_ratio <= max_width_ratio
        )
        is_double_peak.append(meets_criteria)
    
    # Convert to numpy arrays
    peak_indices = np.array(peak_indices)
    next_peak_indices = np.array(next_peak_indices)
    distances = np.array(distances)
    amp_ratios = np.array(amp_ratios)
    width_ratios = np.array(width_ratios)
    is_double_peak = np.array(is_double_peak)
    
    # Count the number of peaks that meet criteria
    double_peak_count = np.sum(is_double_peak)
    print(f"Analyzed {len(peaks)-1} peak pairs")
    print(f"Found {double_peak_count} peak pairs ({double_peak_count/(len(peaks)-1)*100:.1f}%) that meet double peak criteria")
    
    return {
        'peak_indices': peak_indices,
        'next_peak_indices': next_peak_indices,
        'distances': distances,
        'amp_ratios': amp_ratios,
        'width_ratios': width_ratios,
        'is_double_peak': is_double_peak
    }

def plot_double_peak_selection(app, figure, all_peaks, double_peak_data):
    """
    Create a plot showing the distance to next peak for all peaks, highlighting
    those that meet the double peak criteria.
    
    Parameters
    ----------
    app : Application
        The main application instance
    figure : matplotlib.figure.Figure
        Figure to plot on
    all_peaks : array
        Array of all peak indices
    double_peak_data : dict
        Dictionary of peak data from find_double_peaks function
        
    Returns
    -------
    matplotlib.figure.Figure
        Updated figure
    """
    figure.clear()
    ax = figure.add_subplot(111)
    
    # Get the minimum and maximum distance ranges (convert to ms)
    min_distance_ms = app.double_peak_min_distance.get() * 1000
    max_distance_ms = app.double_peak_max_distance.get() * 1000
    
    print(f"DEBUG - Distance range: min={min_distance_ms:.2f}ms, max={max_distance_ms:.2f}ms")
    
    # Plot horizontal lines for distance range with labels
    min_line = ax.axhline(y=min_distance_ms, color='r', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Min Distance ({min_distance_ms:.1f} ms)')
    max_line = ax.axhline(y=max_distance_ms, color='b', linestyle='--', alpha=0.7, linewidth=1.5,
               label=f'Max Distance ({max_distance_ms:.1f} ms)')
    
    # Create a highlighted area between min and max distance
    ax.axhspan(min_distance_ms, max_distance_ms, alpha=0.1, color='green', zorder=1)
    
    # Add text annotations for the distance range
    ax.text(0.99, min_distance_ms, f"{min_distance_ms:.1f} ms", color='r', fontsize=8,
           verticalalignment='bottom', horizontalalignment='right', transform=ax.get_yaxis_transform())
    
    ax.text(0.99, max_distance_ms, f"{max_distance_ms:.1f} ms", color='b', fontsize=8,
           verticalalignment='top', horizontalalignment='right', transform=ax.get_yaxis_transform())
    
    # If we have double peak data, plot it
    if double_peak_data and len(double_peak_data['distances']) > 0:
        # Extract data
        peak_indices = double_peak_data['peak_indices']
        distances = double_peak_data['distances']
        is_double_peak = double_peak_data['is_double_peak']
        
        # Convert distances to ms
        distances_ms = distances * 1000
        
        # Get time values for each peak
        times = app.t_value[all_peaks[peak_indices]] / 60  # Convert to minutes
        
        # Create a color array based on whether each peak meets criteria
        colors = np.array(['blue' if flag else 'darkgray' for flag in is_double_peak])
        
        # Plot all peaks with colors indicating selection status
        scatter = ax.scatter(
            times,
            distances_ms,
            c=colors,  # Array of colors based on is_double_peak
            marker='o',
            s=15,  # Smaller size for better visibility
            alpha=0.8,
            zorder=3
        )
        
        # Create custom legend entries
        double_peak_count = np.sum(is_double_peak)
        non_double_peak_count = len(is_double_peak) - double_peak_count
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   label=f'Double Peaks ({double_peak_count})', markersize=6),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgray', 
                   label=f'Other Peaks ({non_double_peak_count})', markersize=6)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Summary text
        total_peaks = len(all_peaks)
        percentage = (double_peak_count/len(peak_indices)*100) if len(peak_indices) > 0 else 0
        
        summary_text = (
            f"Total Peaks: {total_peaks}\n"
            f"Analyzed Pairs: {len(peak_indices)}\n"
            f"Double Peaks: {double_peak_count} ({percentage:.1f}%)"
        )
        
        ax.text(0.02, 0.96, summary_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Set y-axis limits based on the selection range with 20% padding
        y_range = max_distance_ms - min_distance_ms
        if y_range > 0:
            y_min = max(0, min_distance_ms - 0.2 * y_range)  # 20% padding below, but not less than 0
            y_max = max_distance_ms + 0.2 * y_range  # 20% padding above
            
            ax.set_ylim(y_min, y_max)
        else:
            # Fallback if min and max are the same
            ax.set_ylim(0, max_distance_ms * 1.4)
    else:
        # If no data, just show empty plot with proper range
        y_range = max_distance_ms - min_distance_ms
        if y_range > 0:
            y_min = max(0, min_distance_ms - 0.2 * y_range)
            y_max = max_distance_ms + 0.2 * y_range
        else:
            y_min = 0
            y_max = max_distance_ms * 1.4
            
        ax.set_ylim(y_min, y_max)
        
        # Add a text message
        ax.text(0.5, 0.5, "No peak data available.\nRun peak detection first.", 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Peak Distance (ms)')
    ax.set_title('Distance to Next Peak Analysis')
    ax.grid(True, alpha=0.3)
    
    figure.tight_layout()
    return figure

def plot_double_peaks_grid(app, double_peak_indices, peaks, t_value, filtered_signal, properties, page=0):
    """
    Create a grid plot of double peak pairs.
    
    Parameters
    ----------
    app : Application
        The main application instance
    double_peak_indices : list
        List of (primary_idx, secondary_idx, ...) tuples for double peaks
    peaks : array
        Array of peak indices
    t_value : array
        Time values
    filtered_signal : array
        Filtered signal values
    properties : dict
        Dictionary of peak properties
    page : int, optional
        Page number for pagination, by default 0
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with grid of double peak plots
    """
    print(f"Creating double peak grid, page {page+1}")
    
    # Create figure with grid arrangement
    fig = Figure(figsize=(15, 15))
    rows, cols = 5, 5
    grid = fig.add_gridspec(rows, cols, hspace=0.5, wspace=0.4)
    
    axs = np.array([[fig.add_subplot(grid[i, j]) for j in range(cols)] for i in range(rows)])
    axs = axs.ravel()
    
    # Clear all subplots and hide them initially
    for ax in axs:
        ax.clear()
        ax.set_visible(False)
    
    # Calculate the range of pairs to display
    peaks_per_page = rows * cols
    start_idx = page * peaks_per_page
    end_idx = min(start_idx + peaks_per_page, len(double_peak_indices))
    
    if start_idx >= len(double_peak_indices):
        print("No pairs to display on this page")
        return fig
    
    print(f"Displaying pairs {start_idx+1} to {end_idx} of {len(double_peak_indices)}")
    
    # Create a shared legend for the entire figure
    legend_lines = [
        Line2D([0], [0], color='b', linewidth=1, label='Signal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=5, label='Primary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=5, label='Secondary')
    ]
    
    # For each visible double peak pair
    for i, pair_idx in enumerate(range(start_idx, end_idx)):
        # Get grid position (row, col)
        row = i // cols
        col = i % cols
        
        # Get the primary and secondary peak indices
        primary_idx, secondary_idx, distance, amp_ratio, width_ratio = double_peak_indices[pair_idx]
        
        # Get the actual peaks
        primary_peak = peaks[primary_idx]
        secondary_peak = peaks[secondary_idx]
        
        # Determine a more focused window around the peaks
        # Calculate the interpeak distance in samples
        peak_distance_samples = abs(secondary_peak - primary_peak)
        
        # Add padding on each side (50% of the interpeak distance, but at least 10 samples)
        padding = max(int(peak_distance_samples * 0.5), 10)
        
        # Define window boundaries focusing just on the peaks and a small margin
        window_start = max(0, min(primary_peak, secondary_peak) - padding)
        window_end = min(len(filtered_signal), max(primary_peak, secondary_peak) + padding)
        
        # Get data in the window
        xData = t_value[window_start:window_end]
        yData = filtered_signal[window_start:window_end]
        
        if len(xData) == 0:
            continue
        
        # Find the background (minimum in the window)
        background = np.min(yData)
        
        # Convert time values to milliseconds relative to the window start
        # This ensures our x-axis starts at 0 and shows positive values
        x_ms = (xData - xData[0]) * 1000  # Convert to ms
        
        # Plot filtered signal
        axs[i].plot(x_ms, yData - background, 'b-', linewidth=0.8)
        
        # Calculate positions for markers relative to window start
        primary_x_ms = (t_value[primary_peak] - xData[0]) * 1000
        primary_y = filtered_signal[primary_peak] - background
        secondary_x_ms = (t_value[secondary_peak] - xData[0]) * 1000
        secondary_y = filtered_signal[secondary_peak] - background
        
        # Mark the peaks with correct positions
        axs[i].plot(primary_x_ms, primary_y, 'ro', markersize=4)
        axs[i].plot(secondary_x_ms, secondary_y, 'go', markersize=4)
        
        # Get peak widths
        try:
            # Get width information for primary peak
            primary_left_idx = int(properties['left_ips'][primary_idx])
            primary_right_idx = int(properties['right_ips'][primary_idx])
            primary_width_height = properties['width_heights'][primary_idx] - background
            
            # Plot width lines for primary peak
            primary_left_x = (t_value[primary_left_idx] - xData[0]) * 1000
            primary_right_x = (t_value[primary_right_idx] - xData[0]) * 1000
            
            axs[i].hlines(primary_width_height, primary_left_x, primary_right_x, 
                          color='r', linestyle='--', linewidth=0.8)
            
            # Get width information for secondary peak
            secondary_left_idx = int(properties['left_ips'][secondary_idx])
            secondary_right_idx = int(properties['right_ips'][secondary_idx])
            secondary_width_height = properties['width_heights'][secondary_idx] - background
            
            # Plot width lines for secondary peak
            secondary_left_x = (t_value[secondary_left_idx] - xData[0]) * 1000
            secondary_right_x = (t_value[secondary_right_idx] - xData[0]) * 1000
            
            axs[i].hlines(secondary_width_height, secondary_left_x, secondary_right_x, 
                          color='g', linestyle='--', linewidth=0.8)
        except (KeyError, IndexError) as e:
            # If width data is not available, just continue
            print(f"Width data not available for peak pair {pair_idx}: {e}")
        
        # Add ultra-compact peak information at the top of the plot
        # Using shorter format to avoid overlap: "#1: 3.4ms 0.8A 1.0W"
        info_text = f"#{start_idx+i+1}: {distance*1000:.1f}ms {amp_ratio:.1f}A {width_ratio:.1f}W"
        axs[i].set_title(info_text, fontsize=6, pad=2)
        
        # Set the x-limits to show the full window, not going negative
        axs[i].set_xlim(0, np.max(x_ms))
        
        # Only show x-labels on the bottom row
        if row == rows - 1:
            axs[i].set_xlabel('Time (ms)', fontsize=7)
        else:
            axs[i].set_xlabel('')
            axs[i].set_xticklabels([])
        
        # Only show y-labels on the leftmost column
        if col == 0:
            axs[i].set_ylabel('Amplitude', fontsize=7)
        else:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])
        
        # Make tick labels smaller
        axs[i].tick_params(axis='both', which='major', labelsize=6)
        
        # Enable grid but make it very light
        axs[i].grid(True, alpha=0.2, linestyle=':')
        
        # Make the subplot visible
        axs[i].set_visible(True)
    
    # Add legend at the bottom of the figure
    fig.legend(handles=legend_lines, loc='lower center', ncol=3, fontsize=10,
              bbox_to_anchor=(0.5, 0.02), frameon=True)
    
    # Add overall title and subtitle explaining the format
    if double_peak_indices:
        fig.suptitle(f"Double Peak Analysis (Page {page+1}/{(len(double_peak_indices)-1)//peaks_per_page+1})", 
                   fontsize=16, y=0.98)
        subtitle = "Format: #N: separation(ms) amplitude_ratio(A) width_ratio(W)"
        fig.text(0.5, 0.96, subtitle, ha='center', fontsize=10, style='italic')
    else:
        fig.suptitle("No Double Peaks Found", fontsize=16, y=0.98)
    
    # Adjust layout to accommodate the legend at the bottom
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    return fig

def analyze_double_peaks(app, profile_function=None):
    """
    Analyze distances between all consecutive peaks and identify double peaks.
    
    Parameters
    ----------
    app : Application
        The main application instance
    profile_function : callable, optional
        Function for profiling, by default None
        
    Returns
    -------
    tuple
        (double_peak_data, figures) containing the peak distance data and created figures
    """
    if app.filtered_signal is None or not hasattr(app, 'peaks') or app.peaks is None:
        app.preview_label.config(text="No peaks detected. Please run peak detection first.", foreground="red")
        return None, None
    
    try:
        # Get parameters for double peak detection
        parameters = {
            'min_distance': app.double_peak_min_distance.get(),
            'max_distance': app.double_peak_max_distance.get(),
            'min_amp_ratio': app.double_peak_min_amp_ratio.get(),
            'max_amp_ratio': app.double_peak_max_amp_ratio.get(),
            'min_width_ratio': app.double_peak_min_width_ratio.get(),
            'max_width_ratio': app.double_peak_max_width_ratio.get()
        }
        
        # Get time resolution
        time_resolution = app.time_resolution.get()
        
        # Get previously detected peaks
        peaks = app.peaks
        properties = {
            'prominences': app.peak_heights,
            'widths': app.peak_widths,
            'left_ips': getattr(app, 'peak_left_ips', np.zeros_like(peaks)),
            'right_ips': getattr(app, 'peak_right_ips', np.zeros_like(peaks)),
            'width_heights': getattr(app, 'peak_width_heights', np.zeros_like(peaks))
        }
        
        # Calculate distances and identify double peaks
        double_peak_data = find_double_peaks(peaks, properties, parameters, time_resolution)
        
        # Store double peak data in app
        app.double_peak_data = double_peak_data
        app.current_double_peak_page = 0
        
        # Create figures for double peak analysis
        selection_figure = Figure(figsize=(10, 4))
        plot_double_peak_selection(app, selection_figure, peaks, double_peak_data)
        
        # Create the grid view of double peaks - need to convert the data format
        double_peaks = []
        if double_peak_data and len(double_peak_data['is_double_peak']) > 0:
            # Create list of tuples for double peaks in the old format for compatibility
            for i, is_double in enumerate(double_peak_data['is_double_peak']):
                if is_double:
                    primary_idx = double_peak_data['peak_indices'][i]
                    secondary_idx = double_peak_data['next_peak_indices'][i]
                    distance = double_peak_data['distances'][i]
                    amp_ratio = double_peak_data['amp_ratios'][i]
                    width_ratio = double_peak_data['width_ratios'][i]
                    double_peaks.append((primary_idx, secondary_idx, distance, amp_ratio, width_ratio))
        
        # Store the double_peaks list in the app instance
        app.double_peaks = double_peaks
        
        # For backward compatibility, still create the grid figure
        if double_peaks:
            grid_figure = plot_double_peaks_grid(
                app, double_peaks, peaks, app.t_value, app.filtered_signal, 
                properties, page=app.current_double_peak_page
            )
        else:
            # Create an empty figure with a message if no double peaks found
            grid_figure = Figure(figsize=(10, 8))
            ax = grid_figure.add_subplot(111)
            ax.text(0.5, 0.5, "No double peaks found with current parameters.\nTry adjusting the parameters.",
                   ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            grid_figure.tight_layout()
        
        # Return the double peak data and figures
        return double_peak_data, (selection_figure, grid_figure)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        app.preview_label.config(text=f"Error analyzing double peaks: {str(e)}", foreground="red")
        return None, None

def show_next_double_peaks_page(app):
    """
    Show the next page of double peaks.
    
    Parameters
    ----------
    app : Application
        The main application instance
        
    Returns
    -------
    matplotlib.figure.Figure
        Updated figure with next page of double peaks
    """
    try:
        # Check if we have the necessary data
        if not hasattr(app, 'peaks') or app.peaks is None:
            app.preview_label.config(text="No peaks detected. Please run peak detection first.", foreground="red")
            return None
            
        if not hasattr(app, 'double_peaks') or app.double_peaks is None:
            app.preview_label.config(text="No double peak data available. Please analyze double peaks first.", foreground="red")
            return None
            
        # If we have no double peaks, exit
        if not app.double_peaks:
            app.preview_label.config(text="No double peaks found with current parameters.", foreground="red")
            return None
        
        # Calculate total number of pages
        peaks_per_page = 25  # 5x5 grid
        total_pages = (len(app.double_peaks) - 1) // peaks_per_page + 1
        
        # Initialize page number if not set
        if not hasattr(app, 'current_double_peak_page') or app.current_double_peak_page is None:
            app.current_double_peak_page = 0
            
        # Increment page number, wrapping around if needed
        app.current_double_peak_page = (app.current_double_peak_page + 1) % total_pages
        print(f"Moving to double peak page {app.current_double_peak_page + 1} of {total_pages}")
        
        # Ensure we have all required peak properties
        if not hasattr(app, 'peak_heights') or app.peak_heights is None:
            app.preview_label.config(text="Peak prominence data is missing. Please re-run peak detection.", foreground="red")
            return None
            
        if not hasattr(app, 'peak_widths') or app.peak_widths is None:
            app.preview_label.config(text="Peak width data is missing. Please re-run peak detection.", foreground="red")
            return None
        
        # Get properties for plotting with safe defaults
        properties = {
            'prominences': getattr(app, 'peak_heights', np.ones_like(app.peaks)),
            'widths': getattr(app, 'peak_widths', np.ones_like(app.peaks)),
            'left_ips': getattr(app, 'peak_left_ips', np.zeros_like(app.peaks)),
            'right_ips': getattr(app, 'peak_right_ips', np.zeros_like(app.peaks)),
            'width_heights': getattr(app, 'peak_width_heights', np.zeros_like(app.peaks))
        }
        
        # Create new grid figure with updated page
        grid_figure = plot_double_peaks_grid(
            app, app.double_peaks, app.peaks, app.t_value, app.filtered_signal, 
            properties, page=app.current_double_peak_page
        )
        
        return grid_figure
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        app.preview_label.config(text=f"Error navigating to next double peaks: {str(e)}", foreground="red")
        return None

def show_prev_double_peaks_page(app):
    """
    Show the previous page of double peaks.
    
    Parameters
    ----------
    app : Application
        The main application instance
        
    Returns
    -------
    matplotlib.figure.Figure
        Updated figure with previous page of double peaks
    """
    try:
        # Check if we have the necessary data
        if not hasattr(app, 'peaks') or app.peaks is None:
            app.preview_label.config(text="No peaks detected. Please run peak detection first.", foreground="red")
            return None
            
        if not hasattr(app, 'double_peaks') or app.double_peaks is None:
            app.preview_label.config(text="No double peak data available. Please analyze double peaks first.", foreground="red")
            return None
            
        # If we have no double peaks, exit
        if not app.double_peaks:
            app.preview_label.config(text="No double peaks found with current parameters.", foreground="red")
            return None
        
        # Calculate total number of pages
        peaks_per_page = 25  # 5x5 grid
        total_pages = (len(app.double_peaks) - 1) // peaks_per_page + 1
        
        # Initialize page number if not set
        if not hasattr(app, 'current_double_peak_page') or app.current_double_peak_page is None:
            app.current_double_peak_page = 0
            
        # Decrement page number, wrapping around if needed
        app.current_double_peak_page = (app.current_double_peak_page - 1) % total_pages
        print(f"Moving to double peak page {app.current_double_peak_page + 1} of {total_pages}")
        
        # Ensure we have all required peak properties
        if not hasattr(app, 'peak_heights') or app.peak_heights is None:
            app.preview_label.config(text="Peak prominence data is missing. Please re-run peak detection.", foreground="red")
            return None
            
        if not hasattr(app, 'peak_widths') or app.peak_widths is None:
            app.preview_label.config(text="Peak width data is missing. Please re-run peak detection.", foreground="red")
            return None
        
        # Get properties for plotting with safe defaults
        properties = {
            'prominences': getattr(app, 'peak_heights', np.ones_like(app.peaks)),
            'widths': getattr(app, 'peak_widths', np.ones_like(app.peaks)),
            'left_ips': getattr(app, 'peak_left_ips', np.zeros_like(app.peaks)),
            'right_ips': getattr(app, 'peak_right_ips', np.zeros_like(app.peaks)),
            'width_heights': getattr(app, 'peak_width_heights', np.zeros_like(app.peaks))
        }
        
        # Create new grid figure with updated page
        grid_figure = plot_double_peaks_grid(
            app, app.double_peaks, app.peaks, app.t_value, app.filtered_signal, 
            properties, page=app.current_double_peak_page
        )
        
        return grid_figure
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        app.preview_label.config(text=f"Error navigating to previous double peaks: {str(e)}", foreground="red")
        return None 