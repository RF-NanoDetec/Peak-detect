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
import matplotlib.pyplot as plt

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
        - 'distances': distances between peak maxima (in seconds)
        - 'start_distances': distances between peak starts (in seconds)
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
            'start_distances': np.array([]),
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
    start_distances = []
    amp_ratios = []
    width_ratios = []
    is_double_peak = []
    
    # Calculate information for each peak and its next peak
    for i in range(len(peaks) - 1):
        current_peak = peaks[i]
        next_peak = peaks[i+1]
        
        # Calculate peak-to-peak distance in seconds
        peak_distance = (next_peak - current_peak) * time_resolution
        
        # Calculate start-to-start distance in seconds
        # left_ips values are already absolute positions, not relative offsets
        current_start = properties['left_ips'][i]  # Use left_ips directly
        next_start = properties['left_ips'][i+1]  # Use left_ips directly
        start_distance = (next_start - current_start) * time_resolution
        
        # Print debug information for the first few peaks
        if i < 3:  # Only print first 3 peaks to avoid cluttering
            print(f"\nPeak {i} detailed analysis:")
            print(f"  Peak positions: current={current_peak}, next={next_peak}")
            print(f"  Left IPs (raw): current={properties['left_ips'][i]:.6f}, next={properties['left_ips'][i+1]:.6f}")
            print(f"  Left IPs (samples): current={int(properties['left_ips'][i])}, next={int(properties['left_ips'][i+1])}")
            print(f"  Left IPs (ms): current={properties['left_ips'][i]*time_resolution*1000:.3f}, next={properties['left_ips'][i+1]*time_resolution*1000:.3f}")
            print(f"  Start positions: current={current_start:.3f}, next={next_start:.3f}")
            print(f"  Peak-to-peak = {peak_distance*1000:.2f}ms")
            print(f"  Start-to-start = {start_distance*1000:.2f}ms")
            print(f"  Difference = {(start_distance - peak_distance)*1000:.2f}ms")
            print(f"  Peak widths: current={properties['widths'][i]:.3f}, next={properties['widths'][i+1]:.3f}")
        
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
        start_distances.append(start_distance)
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
    start_distances = np.array(start_distances)
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
        'start_distances': start_distances,
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
    
    # Create two subplots side by side
    ax1 = figure.add_subplot(121)
    ax2 = figure.add_subplot(122)
    
    # Get SEMANTIC theme colors
    min_dist_color = app.theme_manager.get_plot_color('line_dist_min')
    max_dist_color = app.theme_manager.get_plot_color('line_dist_max')
    span_color = app.theme_manager.get_plot_color('span_dist')
    double_marker_color = app.theme_manager.get_plot_color('marker_double')
    nondouble_marker_color = app.theme_manager.get_plot_color('marker_nondouble')
    text_color = app.theme_manager.get_plot_color('text.color') # For annotations
    
    # Get the minimum and maximum distance ranges (convert to ms)
    min_distance_ms = app.double_peak_min_distance.get() * 1000
    max_distance_ms = app.double_peak_max_distance.get() * 1000
    
    print(f"DEBUG - Distance range: min={min_distance_ms:.2f}ms, max={max_distance_ms:.2f}ms")
    
    # Plot horizontal lines for distance range with labels on both plots
    for ax in [ax1, ax2]:
        ax.axhline(y=min_distance_ms, color=min_dist_color, # Use color
                              linestyle='--', alpha=0.7, linewidth=1.5,
                              label=f'Min Distance ({min_distance_ms:.1f} ms)')
        ax.axhline(y=max_distance_ms, color=max_dist_color, # Use color
                              linestyle='--', alpha=0.7, linewidth=1.5,
                              label=f'Max Distance ({max_distance_ms:.1f} ms)')
        
        # Create a highlighted area between min and max distance
        ax.axhspan(min_distance_ms, max_distance_ms, alpha=0.15, # Slightly increased alpha
                   color=span_color, # Use color
                   zorder=1)
        
        # Add text annotations for the distance range using color
        ax.text(0.99, min_distance_ms, f"{min_distance_ms:.1f} ms", color=min_dist_color, # Use color for text
               fontsize=8, verticalalignment='bottom', horizontalalignment='right',
               transform=ax.get_yaxis_transform())
        
        ax.text(0.99, max_distance_ms, f"{max_distance_ms:.1f} ms", color=max_dist_color, # Use color for text
               fontsize=8, verticalalignment='top', horizontalalignment='right',
               transform=ax.get_yaxis_transform())
    
    # If we have double peak data, plot it
    if double_peak_data and len(double_peak_data['distances']) > 0:
        # Extract data
        peak_indices = double_peak_data['peak_indices']
        distances = double_peak_data['distances']
        start_distances = double_peak_data['start_distances']
        is_double_peak = double_peak_data['is_double_peak']
        
        # Convert distances to ms
        distances_ms = distances * 1000
        start_distances_ms = start_distances * 1000
        
        # Get time values for each peak
        times = app.t_value[all_peaks[peak_indices]] / 60  # Convert to minutes
        
        # Create a color array based on whether each peak meets criteria using theme colors
        colors = np.array([double_marker_color if flag else nondouble_marker_color
                           for flag in is_double_peak])
        
        # Plot peak-to-peak distances (Removed hardcoded colors)
        scatter1 = ax1.scatter(
            times, distances_ms, c=colors, marker='o', s=10, # Smaller size
            alpha=0.7, zorder=3) # Reduced alpha slightly
        
        # Plot start-to-start distances (Removed hardcoded colors)
        scatter2 = ax2.scatter(
            times, start_distances_ms, c=colors, marker='o', s=10,
            alpha=0.7, zorder=3)
        
        # Create custom legend entries
        double_peak_count = np.sum(is_double_peak)
        non_double_peak_count = len(is_double_peak) - double_peak_count
        
        legend_elements = [
            Line2D([0], [0], marker='o', color=double_marker_color, label='Double Peaks ({double_peak_count})', markersize=6),
            Line2D([0], [0], marker='o', color=nondouble_marker_color, label=f'Other Peaks ({non_double_peak_count})', markersize=6)
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Summary text for both plots
        total_peaks = len(all_peaks)
        percentage = (double_peak_count/len(peak_indices)*100) if len(peak_indices) > 0 else 0
        
        summary_text = (
            f"Total Peaks: {total_peaks}\n"
            f"Analyzed Pairs: {len(peak_indices)}\n"
            f"Double Peaks: {double_peak_count} ({percentage:.1f}%)"
        )
        
        ax1.text(0.02, 0.96, summary_text, transform=ax1.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
        # Set y-axis limits based on the selection range with 20% padding
        y_range = max_distance_ms - min_distance_ms
        if y_range > 0:
            y_min = max(0, min_distance_ms - 0.2 * y_range)  # 20% padding below, but not less than 0
            y_max = max_distance_ms + 0.2 * y_range  # 20% padding above
            
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
        else:
            # Fallback if min and max are the same
            ax1.set_ylim(0, max_distance_ms * 1.4)
            ax2.set_ylim(0, max_distance_ms * 1.4)
    else:
        # If no data, just show empty plot with proper range
        y_range = max_distance_ms - min_distance_ms
        if y_range > 0:
            y_min = max(0, min_distance_ms - 0.2 * y_range)
            y_max = max_distance_ms + 0.2 * y_range
        else:
            y_min = 0
            y_max = max_distance_ms * 1.4
            
        ax1.set_ylim(y_min, y_max)
        ax2.set_ylim(y_min, y_max)
        
        # Add a text message
        for ax in [ax1, ax2]:
            ax.text(0.5, 0.5, "No peak data available.\nRun peak detection first.", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Peak-to-Peak Distance (ms)')
    ax1.set_title('Peak Maximum Distance Analysis')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Start-to-Start Distance (ms)')
    ax2.set_title('Peak Start Distance Analysis')
    ax2.grid(True, alpha=0.3)
    
    figure.tight_layout()
    
    # Apply theme colors to the figure and axes
    app.theme_manager.apply_plot_theme(figure, [ax1, ax2])

    return figure

def plot_double_peaks_grid(app, double_peaks, peaks, t_value, filtered_signal, properties, double_peak_data, page=0):
    """
    Create a grid of plots showing double peak pairs.
    
    Parameters
    ----------
    app : Application
        The main application instance
    double_peaks : list
        List of tuples containing (primary_idx, secondary_idx, distance, amp_ratio, width_ratio)
    peaks : array
        Array of all peak indices
    t_value : array
        Time values for the signal
    filtered_signal : array
        Filtered signal values
    properties : dict
        Dictionary of peak properties from find_peaks_with_window
    double_peak_data : dict
        Dictionary containing double peak analysis data including start_distances
    page : int, optional
        Page number to display (0-based), by default 0
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the grid of double peak plots
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
    end_idx = min(start_idx + peaks_per_page, len(double_peaks))
    
    if start_idx >= len(double_peaks):
        print("No pairs to display on this page")
        # Apply theme even to empty figure before returning
        app.theme_manager.apply_plot_theme(fig, axs)
        return fig
    
    print(f"Displaying pairs {start_idx+1} to {end_idx} of {len(double_peaks)}")
    
    # Get SEMANTIC theme colors
    signal_color = app.theme_manager.get_plot_color('line_filtered')
    primary_marker_color = app.theme_manager.get_plot_color('marker_peak')
    secondary_marker_color = app.theme_manager.get_plot_color('marker_width') # Reusing width color for secondary
    primary_width_color = primary_marker_color
    secondary_width_color = secondary_marker_color
    grid_color = app.theme_manager.get_plot_color('grid.color') # Get grid color
    
    # Create a shared legend for the entire figure using theme colors
    legend_lines = [
        Line2D([0], [0], color=signal_color, linewidth=1, label='Signal'),
        Line2D([0], [0], marker='o', color='None', markerfacecolor=primary_marker_color, markersize=5, label='Primary'),
        Line2D([0], [0], marker='o', color='None', markerfacecolor=secondary_marker_color, markersize=5, label='Secondary')
    ]
    
    # For each visible double peak pair
    for i, (primary_idx, secondary_idx, distance, amp_ratio, width_ratio) in enumerate(double_peaks[start_idx:end_idx]):
        # Get grid position (row, col)
        row = i // cols
        col = i % cols
        
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
        axs[i].plot(x_ms, yData - background, color=signal_color, linestyle='-', linewidth=0.8)
        
        # Calculate positions for markers relative to window start
        primary_x_ms = (t_value[primary_peak] - xData[0]) * 1000
        primary_y = filtered_signal[primary_peak] - background
        secondary_x_ms = (t_value[secondary_peak] - xData[0]) * 1000
        secondary_y = filtered_signal[secondary_peak] - background
        
        # Mark the peaks with correct positions
        axs[i].plot(primary_x_ms, primary_y, marker='o', color=primary_marker_color, linestyle='None', markersize=4)
        axs[i].plot(secondary_x_ms, secondary_y, marker='o', color=secondary_marker_color, linestyle='None', markersize=4)
        
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
                          color=primary_width_color, linestyle='--', linewidth=0.8)
            
            # Get width information for secondary peak
            secondary_left_idx = int(properties['left_ips'][secondary_idx])
            secondary_right_idx = int(properties['right_ips'][secondary_idx])
            secondary_width_height = properties['width_heights'][secondary_idx] - background
            
            # Plot width lines for secondary peak
            secondary_left_x = (t_value[secondary_left_idx] - xData[0]) * 1000
            secondary_right_x = (t_value[secondary_right_idx] - xData[0]) * 1000
            
            axs[i].hlines(secondary_width_height, secondary_left_x, secondary_right_x, 
                          color=secondary_width_color, linestyle='--', linewidth=0.8)
        except (KeyError, IndexError) as e:
            # If width data is not available, just continue
            print(f"Width data not available for peak pair {i}: {e}")
        
        # Add ultra-compact peak information at the top of the plot
        # Using shorter format to avoid overlap: "#1: 3.4ms 0.8A 1.0W"
        info_text = (
            f"#{start_idx+i+1}: "
            f"P-P:{distance*1000:.1f}ms "
            f"S-S:{double_peak_data['start_distances'][i]*1000:.1f}ms "
            f"A:{amp_ratio:.1f} W:{width_ratio:.1f}"
        )
        axs[i].set_title(info_text, fontsize=7, pad=2)
        
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
        axs[i].tick_params(axis='both', which='major', labelsize=7)
        
        # Enable grid but make it very light
        axs[i].grid(True, alpha=0.2, linestyle=':')
        
        # Make the subplot visible
        axs[i].set_visible(True)
    
    # Add legend at the bottom of the figure
    fig.legend(handles=legend_lines, loc='lower center', ncol=3, fontsize=10,
              bbox_to_anchor=(0.5, 0.02), frameon=True)
    
    # Add overall title and subtitle explaining the format
    if double_peaks:
        fig.suptitle(f"Double Peak Analysis (Page {page+1}/{(len(double_peaks)-1)//peaks_per_page+1})", 
                   fontsize=16, y=0.98)
        subtitle = "Format: #N: separation(ms) amplitude_ratio(A) width_ratio(W)"
        fig.text(0.5, 0.96, subtitle, ha='center', fontsize=10, style='italic')
    else:
        fig.suptitle("No Double Peaks Found", fontsize=16, y=0.98)
    
    # Adjust layout to accommodate the legend at the bottom
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    # Apply theme colors explicitly to all axes
    app.theme_manager.apply_plot_theme(fig, axs)

    return fig

def analyze_double_peaks(app):
    """
    Analyze distances between all consecutive peaks and identify double peaks.
    
    Parameters
    ----------
    app : Application
        The main application instance
        
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
            'left_ips': app.peak_left_ips,
            'right_ips': app.peak_right_ips,
            'width_heights': app.peak_width_heights
        }
        
        # Calculate distances and identify double peaks
        double_peak_data = find_double_peaks(peaks, properties, parameters, time_resolution)
        
        # Store double peak data in app
        app.double_peak_data = double_peak_data
        app.current_double_peak_page = 0
        
        # Update histograms if they exist
        if hasattr(app, 'amp_hist_ax') and hasattr(app, 'width_hist_ax'):
            # Clear previous histograms
            app.amp_hist_ax.clear()
            app.width_hist_ax.clear()
            
            # === Apply Theme to Histograms ===
            app._update_histogram_theme(app.amp_hist_canvas, app.amp_hist_ax)
            app._update_histogram_theme(app.width_hist_canvas, app.width_hist_ax)
            # === End Apply Theme ===
            
            # Get indices of peaks within distance range
            distance_mask = (double_peak_data['distances'] >= parameters['min_distance']) & \
                          (double_peak_data['distances'] <= parameters['max_distance'])
            
            # Plot amplitude ratio histogram for selected peaks only
            if len(double_peak_data['amp_ratios']) > 0:
                selected_amp_ratios = double_peak_data['amp_ratios'][distance_mask]
                if len(selected_amp_ratios) > 0:
                    # Define theme-aware colors
                    is_dark = app.theme_manager.current_theme == 'dark'
                    bar_color = app.theme_manager.get_color('secondary') if is_dark else app.theme_manager.get_color('primary')
                    line_color = '#FF8A80' if is_dark else 'red' # Brighter red for dark
                    
                    app.amp_hist_ax.hist(
                        selected_amp_ratios,
                        bins=30, 
                        range=(0, 5),
                        density=True,
                        alpha=0.75, # Slightly increased alpha
                        color=bar_color # Use theme color
                    )
                    # Add vertical lines for current range with theme color
                    app.amp_hist_ax.axvline(
                        parameters['min_amp_ratio'],
                        color=line_color,
                        linestyle='--',
                        linewidth=1, # Thinner line
                        alpha=0.8 # More visible
                    )
                    app.amp_hist_ax.axvline(
                        parameters['max_amp_ratio'],
                        color=line_color,
                        linestyle='--',
                        linewidth=1,
                        alpha=0.8
                    )
            
            # Plot width ratio histogram for selected peaks only
            if len(double_peak_data['width_ratios']) > 0:
                selected_width_ratios = double_peak_data['width_ratios'][distance_mask]
                if len(selected_width_ratios) > 0:
                    # Define theme-aware colors (can reuse from above)
                    is_dark = app.theme_manager.current_theme == 'dark'
                    bar_color = app.theme_manager.get_color('secondary') if is_dark else app.theme_manager.get_color('primary')
                    line_color = '#FF8A80' if is_dark else 'red'
                    
                    app.width_hist_ax.hist(
                        selected_width_ratios,
                        bins=30, 
                        range=(0, 5),
                        density=True,
                        alpha=0.75,
                        color=bar_color # Use theme color
                    )
                    # Add vertical lines for current range with theme color
                    app.width_hist_ax.axvline(
                        parameters['min_width_ratio'],
                        color=line_color,
                        linestyle='--',
                        linewidth=1,
                        alpha=0.8
                    )
                    app.width_hist_ax.axvline(
                        parameters['max_width_ratio'],
                        color=line_color,
                        linestyle='--',
                        linewidth=1,
                        alpha=0.8
                    )
            
            # Update plot settings for both histograms (theme applied already)
            for ax in [app.amp_hist_ax, app.width_hist_ax]:
                ax.set_xlim(0, 5)
                ax.set_ylim(0, 1)
                ax.set_xticks([0, 1, 2, 3, 4, 5])
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)
                # Make tick labels smaller
                ax.tick_params(axis='both', which='major', labelsize=6)
                # Make the plot more compact
                ax.margins(x=0.05)
            
            # Update canvases
            app.amp_hist_canvas.draw()
            app.width_hist_canvas.draw()
        
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
                properties, double_peak_data, page=app.current_double_peak_page
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
            properties, app.double_peak_data, page=app.current_double_peak_page
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
            properties, app.double_peak_data, page=app.current_double_peak_page
        )
        
        return grid_figure
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        app.preview_label.config(text=f"Error navigating to previous double peaks: {str(e)}", foreground="red")
        return None 