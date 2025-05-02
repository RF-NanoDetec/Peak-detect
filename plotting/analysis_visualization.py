"""
Analysis visualization functions for the Peak Analysis Tool.

This module contains functions for advanced visualization of peak detection results and data analysis.
"""

import numpy as np
import traceback
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from matplotlib.lines import Line2D
from core.peak_analysis_utils import find_peaks_with_window
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def plot_data(app, profile_function=None):
    """Plot processed data with peaks in a new tab."""
    if app.filtered_signal is None:
        app.preview_label.config(
            text="Filtered signal not available. Please start the analysis first.",
            foreground=app.theme_manager.get_color('error')
            )
        return

    try:
        print("Starting plot_data function...")

        # Create a new figure for data plot
        # Figure background color is handled by theme manager
        app.data_figure = Figure(figsize=(10, 8))

        # Create subplots
        axes = app.data_figure.subplots(nrows=4, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios': [1, 1, 1, 1.2],
                                                   'hspace': 0.3})

        # Convert data to float32 for memory efficiency
        t = np.asarray(app.t_value, dtype=np.float32)  # Time already in seconds
        filtered_signal = np.asarray(app.filtered_signal, dtype=np.float32)

        # Find peaks with optimized parameters
        width_values = app.width_p.get().strip().split(',')
        
        # Use time_resolution directly instead of calculating from time differences
        rate = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution
        if rate <= 0:
            print(f"WARNING: Invalid time resolution ({rate}), using default of 0.0001")
            rate = 0.0001  # Default to 0.1ms sampling if invalid value
        
        sampling_rate = 1 / rate
        print(f"DEBUG - plot_data sampling rate: {sampling_rate:.1f} Hz (from time resolution {rate})")
        
        # Convert width values from ms to samples using the actual sampling rate
        # For example, 50ms at 10kHz sampling rate would be 500 samples
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        
        # Debug output for width parameters
        print(f"\nDEBUG - Width parameter information:")
        print(f"Original width values: {app.width_p.get()} (ms)")
        print(f"Sampling rate: {sampling_rate:.1f} Hz")
        print(f"Converted width_p (in samples): {width_p}")
        
        peaks, properties = find_peaks_with_window(
            filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get()
        )
        
        # Additional debug information after peak detection
        print(f"DEBUG - Peak detection results:")
        print(f"Found {len(peaks)} peaks")
        if len(peaks) > 0:
            print(f"Peak properties keys: {list(properties.keys())}")
            print(f"First few peak positions: {peaks[:5]}")
            if 'widths' in properties:
                print(f"First few peak widths (in samples): {properties['widths'][:5]}")
                print(f"Peak widths in seconds (min/mean/max): {np.min(properties['widths']*rate):.6f}/{np.mean(properties['widths']*rate):.6f}/{np.max(properties['widths']*rate):.6f}")
                print(f"Peak widths in ms (min/mean/max): {np.min(properties['widths']*rate*1000):.2f}/{np.mean(properties['widths']*rate*1000):.2f}/{np.max(properties['widths']*rate*1000):.2f}")
        
        # Also detect filtered peaks if the toggle is enabled
        filtered_out_peaks = np.array([])
        filtered_out_properties = {}
        
        if hasattr(app, 'show_filtered_peaks') and app.show_filtered_peaks.get():
            # Get the prominence ratio threshold
            prominence_ratio = app.prominence_ratio.get() if hasattr(app, 'prominence_ratio') else 0.8
            
            # Find all peaks without the prominence ratio filter
            all_peaks, all_properties = find_peaks_with_window(
                filtered_signal,
                width=width_p,
                prominence=app.height_lim.get(),
                distance=app.distance.get(),
                rel_height=app.rel_height.get()
            )
            
            # Identify which peaks would be filtered out
            if len(all_peaks) > 0:
                # Create mask for peaks that would be filtered out
                filtered_mask = np.ones(len(all_peaks), dtype=bool)
                
                # Check each peak against prominence ratio threshold
                for i, peak_idx in enumerate(all_peaks):
                    peak_height = filtered_signal[peak_idx]
                    peak_prominence = all_properties['prominences'][i]
                    
                    # Calculate ratio of prominence to peak height
                    if peak_height > 0:  # Avoid division by zero
                        ratio = peak_prominence / peak_height
                        
                        # If the ratio meets the threshold, it would be kept (not filtered)
                        if ratio >= prominence_ratio:
                            filtered_mask[i] = False
                
                # Extract the filtered-out peaks
                filtered_out_peaks = all_peaks[filtered_mask]
                
                # Extract their properties
                for key, values in all_properties.items():
                    if isinstance(values, np.ndarray) and len(values) == len(all_peaks):
                        if key not in filtered_out_properties:
                            filtered_out_properties[key] = values[filtered_mask]
            
            print(f"DEBUG - Found {len(filtered_out_peaks)} peaks that would be filtered out")
        else:
            # When toggle is off, apply prominence ratio filter to peaks
            prominence_ratio = app.prominence_ratio.get() if hasattr(app, 'prominence_ratio') else 0.8
            
            # Create mask for peaks that pass the prominence ratio filter
            filtered_mask = np.ones(len(peaks), dtype=bool)
            
            # Check each peak against prominence ratio threshold
            for i, peak_idx in enumerate(peaks):
                peak_height = filtered_signal[peak_idx]
                peak_prominence = properties['prominences'][i]
                
                # Calculate ratio of prominence to peak height
                if peak_height > 0:  # Avoid division by zero
                    ratio = peak_prominence / peak_height
                    
                    # If the ratio meets the threshold, it passes the filter
                    if ratio >= prominence_ratio:
                        filtered_mask[i] = True
                    else:
                        filtered_mask[i] = False
            
            # Keep only non-filtered peaks
            peaks = peaks[filtered_mask]
            
            # Update properties to include only non-filtered peaks
            for key, values in properties.items():
                if isinstance(values, np.ndarray) and len(values) == len(filtered_mask):
                    properties[key] = values[filtered_mask]
            
            print(f"DEBUG - Keeping {len(peaks)} peaks after prominence ratio filter")
        
        # Calculate peak properties
        widths = properties["widths"]  # Width in samples
        widths_in_seconds = widths * rate  # Convert from samples to seconds
        widths_in_ms = widths_in_seconds * 1000  # Convert from seconds to milliseconds
        
        prominences = properties["prominences"]
        peak_times = t[peaks]
        
        # Also calculate properties for filtered-out peaks if available
        if hasattr(filtered_out_peaks, 'size') and filtered_out_peaks.size > 0:
            filtered_widths = filtered_out_properties["widths"]
            filtered_widths_in_seconds = filtered_widths * rate
            filtered_widths_in_ms = filtered_widths_in_seconds * 1000
            filtered_prominences = filtered_out_properties["prominences"]
            filtered_peak_times = t[filtered_out_peaks]
        
        # Calculate areas under peaks
        areas = []
        for i, peak in enumerate(peaks):
            left_idx = int(properties["left_ips"][i])
            right_idx = int(properties["right_ips"][i])
            if left_idx < right_idx:
                areas.append(np.trapz(filtered_signal[left_idx:right_idx]))
            else:
                areas.append(0)
        areas = np.array(areas)
        
        # Also calculate areas for filtered peaks if available
        filtered_areas = []
        if hasattr(filtered_out_peaks, 'size') and filtered_out_peaks.size > 0 and 'left_ips' in filtered_out_properties and 'right_ips' in filtered_out_properties:
            for i, peak in enumerate(filtered_out_peaks):
                left_idx = int(filtered_out_properties["left_ips"][i])
                right_idx = int(filtered_out_properties["right_ips"][i])
                if left_idx < right_idx:
                    filtered_areas.append(np.trapz(filtered_signal[left_idx:right_idx]))
                else:
                    filtered_areas.append(0)
            filtered_areas = np.array(filtered_areas)

        # Get SEMANTIC theme colors for plotting elements
        scatter_color = app.theme_manager.get_plot_color('scatter_points')
        bar_color = app.theme_manager.get_plot_color('hist_bars') # Use hist color for bars too
        moving_avg_color = app.theme_manager.get_plot_color('moving_average')
        filtered_color = "#FF8080"  # Light red color for filtered peaks

        # Plot peak heights - CHANGED ORDER: First regular peaks, then filtered peaks on top
        axes[0].scatter(peak_times/60, prominences, s=1, alpha=0.5, color=scatter_color, label='Peak Heights')
        # Add filtered peaks if toggle is enabled (these will now appear on top)
        if hasattr(filtered_out_peaks, 'size') and filtered_out_peaks.size > 0:
            axes[0].scatter(filtered_peak_times/60, filtered_prominences, s=2, alpha=0.7, color=filtered_color, label='Filtered Peaks')
        axes[0].set_ylabel('Peak Heights')
        axes[0].grid(True) # alpha/color from rcParams
        axes[0].legend()
        axes[0].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Plot peak widths - CHANGED ORDER: First regular peaks, then filtered peaks on top
        axes[1].scatter(peak_times/60, widths_in_ms, s=1, alpha=0.5, color=scatter_color, label='Peak Widths (ms)')
        # Add filtered peaks if toggle is enabled (these will now appear on top)
        if hasattr(filtered_out_peaks, 'size') and filtered_out_peaks.size > 0:
            axes[1].scatter(filtered_peak_times/60, filtered_widths_in_ms, s=2, alpha=0.7, color=filtered_color, label='Filtered Peaks')
        axes[1].set_ylabel('Peak Widths (ms)')
        axes[1].grid(True)
        axes[1].legend()
        axes[1].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Plot peak areas - CHANGED ORDER: First regular peaks, then filtered peaks on top
        axes[2].scatter(peak_times/60, areas, s=1, alpha=0.5, color=scatter_color, label='Peak Areas')
        # Add filtered peaks if toggle is enabled (these will now appear on top)
        if hasattr(filtered_out_peaks, 'size') and filtered_out_peaks.size > 0 and len(filtered_areas) > 0:
            axes[2].scatter(filtered_peak_times/60, filtered_areas, s=2, alpha=0.7, color=filtered_color, label='Filtered Peaks')
        axes[2].set_ylabel('Peak Areas')
        axes[2].grid(True)
        axes[2].legend()
        axes[2].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Calculate and plot throughput
        interval = app.throughput_interval.get() if hasattr(app, 'throughput_interval') else 10  # seconds
        # Include filtered peaks in throughput calculation if toggle is enabled
        if hasattr(filtered_out_peaks, 'size') and filtered_out_peaks.size > 0 and app.show_filtered_peaks.get():
            all_peak_times = np.concatenate([peak_times, filtered_peak_times])
            bins = np.arange(0, np.max(all_peak_times), interval)
            bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
            
            # Calculate throughput for both included and filtered peaks
            total_throughput, _ = np.histogram(all_peak_times, bins=bins)
            included_throughput, _ = np.histogram(peak_times, bins=bins)
            
            # Use stacked bar chart to show both types
            filtered_throughput = total_throughput - included_throughput
            
            # Plot stacked bar chart
            axes[3].bar(bin_centers/60, included_throughput, width=(interval/60)*0.8,
                      color=bar_color, alpha=0.6, label=f'Included Peaks ({interval:.0f}s bins)')
            axes[3].bar(bin_centers/60, filtered_throughput, width=(interval/60)*0.8, 
                      bottom=included_throughput, color=filtered_color, alpha=0.6, 
                      label=f'Filtered Peaks ({interval:.0f}s bins)')
            
            # Set up throughput for moving average calculation
            throughput = total_throughput if app.show_filtered_peaks.get() else included_throughput
        else:
            bins = np.arange(0, np.max(t), interval)
            bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
            throughput, _ = np.histogram(peak_times, bins=bins)
            
            # Plot throughput bars using SEMANTIC color
            axes[3].bar(bin_centers/60, throughput, width=(interval/60)*0.8,
                      color=bar_color, alpha=0.6, label=f'Throughput ({interval:.0f}s bins)')

        # Add moving average line using SEMANTIC color
        window = 5  # Number of points for moving average
        moving_avg = np.convolve(throughput, np.ones(window)/window, mode='valid')
        moving_avg_times = bin_centers[window-1:]/60
        axes[3].plot(moving_avg_times, moving_avg,
                    color=moving_avg_color,
                    linewidth=1, label=f'{window}-point Moving Average')

        axes[3].set_ylabel(f'Peaks per {interval}s')
        axes[3].set_xlabel('Time (min)')
        axes[3].grid(True)
        axes[3].legend() # Removed fontsize

        # Add statistics annotation to throughput plot (Removed bbox, fontsize=8)
        stats_text = (f'Total Peaks: {len(peaks):,}\n'
                     f'Avg Rate: {len(peaks)/(np.max(t)-np.min(t))*60:.1f} peaks/min\n'
                     f'Max Rate: {np.max(throughput)/(interval/60):.1f} peaks/min\n'
                     f'Interval: {interval:.0f} s')
        axes[3].text(0.02, 0.98, stats_text,
                    transform=axes[3].transAxes,
                    verticalalignment='top',
                    fontsize=8) # Keep annotation font small

        # Update title and layout
        app.data_figure.suptitle('Peak Analysis Over Time', y=0.95)
        app.data_figure.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust rect for suptitle

        # Apply theme standard styles (bg, grid, text)
        app.theme_manager.apply_plot_theme(app.data_figure, axes)

        # Create or update the tab in plot_tab_control
        tab_name = "Peak Analysis"
        tab_exists = False
        canvas = None

        for tab_widget_id in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab_widget_id, "text") == tab_name:
                tab_frame = app.plot_tab_control.nametowidget(tab_widget_id)
                app.plot_tab_control.select(tab_frame)
                # Remove old canvas
                for widget in tab_frame.winfo_children():
                    widget.destroy()
                # Add new canvas
                canvas = FigureCanvasTkAgg(app.data_figure, master=tab_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                app.data_canvas = canvas # Update specific canvas reference
                tab_exists = True
                break

        if not tab_exists:
            new_tab = ttk.Frame(app.plot_tab_control)
            app.plot_tab_control.add(new_tab, text=tab_name)
            app.plot_tab_control.select(new_tab)
            # Create and pack the canvas
            canvas = FigureCanvasTkAgg(app.data_figure, master=new_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            app.data_canvas = canvas # Store specific canvas reference

        # Store the figure associated with the tab
        app.tab_figures[tab_name] = app.data_figure

        app.preview_label.config(
            text="Peak analysis plot created successfully",
            foreground=app.theme_manager.get_color('success')
            )

    except Exception as e:
        app.show_error("Error in plot_data", e)
        app.preview_label.config(
            text="Error creating peak analysis plot.",
            foreground=app.theme_manager.get_color('error')
            )
        # Don't re-raise, just show error in UI


def plot_scatter(app, profile_function=None):
    """Enhanced scatter plot for peak property correlations"""
    if app.filtered_signal is None:
        app.preview_label.config(
            text="Filtered signal not available. Please start the analysis first.",
            foreground=app.theme_manager.get_color('error')
            )
        return

    try:
        # Get peaks and properties
        width_values = app.width_p.get().strip().split(',')
        
        # Use time_resolution directly instead of calculating from time differences
        rate = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution
        if rate <= 0:
            print(f"WARNING: Invalid time resolution ({rate}), using default of 0.0001")
            rate = 0.0001  # Default to 0.1ms sampling if invalid value
        
        sampling_rate = 1 / rate
        print(f"DEBUG - plot_scatter sampling rate: {sampling_rate:.1f} Hz (from time resolution {rate})")
        
        # Convert width values from ms to samples using the actual sampling rate
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        
        # Debug output
        print(f"\nDEBUG - Width parameter information in plot_scatter:")
        print(f"Original width values: {app.width_p.get()} (ms)")
        print(f"Sampling rate: {sampling_rate:.1f} Hz")
        print(f"Converted width_p (in samples): {width_p}")

        # Check if we should show filtered peaks
        show_filtered_peaks = hasattr(app, 'show_filtered_peaks') and app.show_filtered_peaks.get()
        prominence_ratio_threshold = app.prominence_ratio.get() if hasattr(app, 'prominence_ratio') else 0.8
        
        # Get all peaks first without the prominence ratio filter
        all_peaks, all_properties = find_peaks_with_window(
            app.filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get()
        )
        
        # Get peak heights for all detected peaks
        peak_heights = app.filtered_signal[all_peaks]
        
        # Calculate prominence-to-height ratio for all peaks
        prominence_ratios = all_properties['prominences'] / peak_heights
        
        # Create mask for peaks that pass the filter (not filtered out)
        keep_mask = prominence_ratios >= prominence_ratio_threshold
        
        # Get peaks that pass the filter
        peaks_x_filter = all_peaks[keep_mask]
        
        # Extract properties for peaks that pass the filter
        properties = {}
        for prop_name, prop_values in all_properties.items():
            if isinstance(prop_values, np.ndarray) and len(prop_values) == len(all_peaks):
                properties[prop_name] = prop_values[keep_mask]
        
        # Get filtered-out peaks if showing them
        filtered_peaks_x = None
        filtered_properties = None
        
        if show_filtered_peaks:
            # Create mask for filtered peaks (those that would be filtered out)
            filtered_mask = ~keep_mask
            
            # Extract filtered-out peaks
            filtered_peaks_x = all_peaks[filtered_mask]
            
            # Extract properties for filtered-out peaks
            filtered_properties = {}
            for prop_name, prop_values in all_properties.items():
                if isinstance(prop_values, np.ndarray) and len(prop_values) == len(all_peaks):
                    filtered_properties[prop_name] = prop_values[filtered_mask]
            
            print(f"DEBUG - Filtered peaks in plot_scatter:")
            print(f"Found {len(filtered_peaks_x)} peaks filtered out by prominence ratio threshold {prominence_ratio_threshold}")
        
        # Additional debug information after peak detection
        print(f"DEBUG - Peak detection results in plot_scatter:")
        print(f"Found {len(peaks_x_filter)} peaks that pass the filter")
        
        # Calculate peak properties
        widths = properties["widths"]  # Width in samples
        widths_in_seconds = widths * rate  # Convert from samples to seconds
        widths_in_ms = widths_in_seconds * 1000  # Convert from seconds to milliseconds
        
        prominences = properties["prominences"]
        peak_times = app.t_value[peaks_x_filter]
        
        # Calculate peak areas
        window = np.round(widths, 0).astype(int) + 40
        peak_areas = np.zeros(len(peaks_x_filter))

        for i in range(len(peaks_x_filter)):
            # Ensure indices are within valid range
            start_idx = max(0, peaks_x_filter[i] - window[i])
            end_idx = min(len(app.filtered_signal), peaks_x_filter[i] + window[i])
            
            yData = app.filtered_signal[start_idx:end_idx]
            # Check if yData is empty before calculating minimum
            background = np.min(yData) if len(yData) > 0 else 0
            
            st = max(0, int(properties["left_ips"][i]))
            en = min(len(app.filtered_signal), int(properties["right_ips"][i]))
            
            # Only calculate area if we have valid indices
            if st < en:
                peak_areas[i] = np.sum(app.filtered_signal[st:en] - background)
            else:
                peak_areas[i] = 0
                print(f"WARNING: Invalid peak area calculation for peak {i} (st={st}, en={en})")
        
        # Calculate properties for filtered peaks if they exist
        filtered_widths_in_ms = None
        filtered_prominences = None
        filtered_peak_areas = None
        
        if show_filtered_peaks and filtered_peaks_x is not None and len(filtered_peaks_x) > 0:
            # Calculate filtered peak properties
            filtered_widths = filtered_properties["widths"]  # Width in samples
            filtered_widths_in_seconds = filtered_widths * rate  # Convert from samples to seconds
            filtered_widths_in_ms = filtered_widths_in_seconds * 1000  # Convert from seconds to milliseconds
            
            filtered_prominences = filtered_properties["prominences"]
            
            # Calculate areas for filtered peaks
            filtered_window = np.round(filtered_widths, 0).astype(int) + 40
            filtered_peak_areas = np.zeros(len(filtered_peaks_x))
            
            for i in range(len(filtered_peaks_x)):
                # Ensure indices are within valid range
                start_idx = max(0, filtered_peaks_x[i] - filtered_window[i])
                end_idx = min(len(app.filtered_signal), filtered_peaks_x[i] + filtered_window[i])
                
                yData = app.filtered_signal[start_idx:end_idx]
                # Check if yData is empty before calculating minimum
                background = np.min(yData) if len(yData) > 0 else 0
                
                st = max(0, int(filtered_properties["left_ips"][i]))
                en = min(len(app.filtered_signal), int(filtered_properties["right_ips"][i]))
                
                # Only calculate area if we have valid indices
                if st < en:
                    filtered_peak_areas[i] = np.sum(app.filtered_signal[st:en] - background)
                else:
                    filtered_peak_areas[i] = 0
                    print(f"WARNING: Invalid filtered peak area calculation for peak {i} (st={st}, en={en})")

        # Create DataFrame with all peak properties
        df_all = pd.DataFrame({
            "width": widths_in_ms,  # Use widths_in_ms directly
            "amplitude": prominences,
            "area": peak_areas
        })
        
        # Create DataFrame for filtered peaks if they exist
        df_filtered = None
        if show_filtered_peaks and filtered_widths_in_ms is not None:
            df_filtered = pd.DataFrame({
                "width": filtered_widths_in_ms,
                "amplitude": filtered_prominences,
                "area": filtered_peak_areas
            })
        
        # Debug output for peak widths
        if len(widths) > 0:
            print(f"\nDEBUG - Peak width information in plot_scatter:")
            print(f"Raw widths (samples): {np.min(widths):.1f}/{np.mean(widths):.1f}/{np.max(widths):.1f}")
            print(f"Widths in seconds: {np.min(widths_in_seconds):.6f}/{np.mean(widths_in_seconds):.6f}/{np.max(widths_in_seconds):.6f}")
            print(f"Widths in ms: {np.min(widths_in_ms):.2f}/{np.mean(widths_in_ms):.2f}/{np.max(widths_in_ms):.2f}")
            print(f"Widths in DataFrame (ms): {df_all['width'].min():.2f}/{df_all['width'].mean():.2f}/{df_all['width'].max():.2f}")
            
            # Calculate sampling rate for verification
            print(f"Median time difference between samples: {rate:.8f} seconds")
            print(f"Sampling rate: {sampling_rate:.1f} Hz")
            
            # Calculate average width in sample points
            print(f"Average width in sample points: {np.mean(widths):.2f}")

        # Create new figure
        # Figure bg color handled by theme manager
        new_figure = Figure(figsize=(12, 10))
        gs = new_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3) # Increased spacing slightly
        ax = [new_figure.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        # Axes bg color handled by theme manager

        # Get SEMANTIC theme colors
        hist_color = app.theme_manager.get_plot_color('hist_bars')
        hist_edge_color = app.theme_manager.get_plot_color('patch.edgecolor') # Use standard edge color
        filtered_color = "#FF8080" # Light red for filtered peaks
        scatter_color = app.theme_manager.get_plot_color('scatter_points') # Use same color as time-resolved

        # Parameters for scatter plots - use consistent values with time-resolved visualization
        scatter_size = 2  # Base size for dots
        scatter_alpha = 0.5  # Lower alpha for better visibility when overlapping
        filtered_alpha = 0.7  # Higher alpha for filtered peaks
        filtered_size = 3  # Slightly larger size for filtered peaks

        # Plot 1: Width vs Amplitude (simple scatter, no density gradient)
        # Scale dot size by area (normalized to a reasonable range)
        area_scaled = (df_all['area'] / df_all['area'].max()) * 10 + 1  # Scale to 1-11 size range
        ax[0].scatter(df_all['width'], df_all['amplitude'],
                    color=scatter_color, s=area_scaled, alpha=scatter_alpha)
                    
        # Add filtered peaks if available
        if show_filtered_peaks and df_filtered is not None and len(df_filtered) > 0:
            filtered_area_scaled = (df_filtered['area'] / df_all['area'].max()) * 10 + 1  # Use same scale as main data
            ax[0].scatter(df_filtered['width'], df_filtered['amplitude'], 
                    color=filtered_color, s=filtered_area_scaled, alpha=filtered_alpha, 
                    label=f'Filtered Peaks (n={len(df_filtered)})')
            ax[0].legend(fontsize=8)
            
        ax[0].set_xlabel('Width (ms)')
        ax[0].set_ylabel('Amplitude (counts)')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].grid(True) # Use theme grid style
        ax[0].set_title('Width vs Amplitude')
        corr1 = df_all['width'].corr(df_all['amplitude'])
        ax[0].text(0.05, 0.95, f'r = {corr1:.2f}', transform=ax[0].transAxes,
                fontsize=9, verticalalignment='top') # Keep annotation font small

        # Plot 2: Width vs Area (simple scatter, no density gradient)
        # Scale dot size by amplitude
        amplitude_scaled = (df_all['amplitude'] / df_all['amplitude'].max()) * 10 + 1  # Scale to 1-11 size range
        ax[1].scatter(df_all['width'], df_all['area'],
                    color=scatter_color, s=amplitude_scaled, alpha=scatter_alpha)
                    
        # Add filtered peaks if available
        if show_filtered_peaks and df_filtered is not None and len(df_filtered) > 0:
            filtered_amplitude_scaled = (df_filtered['amplitude'] / df_all['amplitude'].max()) * 10 + 1  # Use same scale
            ax[1].scatter(df_filtered['width'], df_filtered['area'], 
                    color=filtered_color, s=filtered_amplitude_scaled, alpha=filtered_alpha, 
                    label=f'Filtered Peaks (n={len(df_filtered)})')
            ax[1].legend(fontsize=8)
            
        ax[1].set_xlabel('Width (ms)')
        ax[1].set_ylabel('Area (counts)')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].grid(True)
        ax[1].set_title('Width vs Area')
        corr2 = df_all['width'].corr(df_all['area'])
        ax[1].text(0.05, 0.95, f'r = {corr2:.2f}', transform=ax[1].transAxes,
                fontsize=9, verticalalignment='top')

        # Plot 3: Amplitude vs Area (simple scatter, no density gradient)
        # Scale dot size by width
        width_scaled = (df_all['width'] / df_all['width'].max()) * 10 + 1  # Scale to 1-11 size range
        ax[2].scatter(df_all['amplitude'], df_all['area'],
                    color=scatter_color, s=width_scaled, alpha=scatter_alpha)
                    
        # Add filtered peaks if available
        if show_filtered_peaks and df_filtered is not None and len(df_filtered) > 0:
            filtered_width_scaled = (df_filtered['width'] / df_all['width'].max()) * 10 + 1  # Use same scale
            ax[2].scatter(df_filtered['amplitude'], df_filtered['area'], 
                    color=filtered_color, s=filtered_width_scaled, alpha=filtered_alpha, 
                    label=f'Filtered Peaks (n={len(df_filtered)})')
            ax[2].legend(fontsize=8)
            
        ax[2].set_xlabel('Amplitude (counts)')
        ax[2].set_ylabel('Area (counts)')
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].grid(True)
        ax[2].set_title('Amplitude vs Area')
        corr3 = df_all['amplitude'].corr(df_all['area'])
        ax[2].text(0.05, 0.95, f'r = {corr3:.2f}', transform=ax[2].transAxes,
                fontsize=9, verticalalignment='top')

        # Plot 4: Width distribution using SEMANTIC colors
        import seaborn as sns
        sns.histplot(data=df_all, x='width', bins=50, ax=ax[3],
                     color=hist_color,
                     edgecolor=hist_edge_color,
                     alpha=0.7)
                     
        # Add filtered peaks histogram if available
        if show_filtered_peaks and df_filtered is not None and len(df_filtered) > 0:
            sns.histplot(data=df_filtered, x='width', bins=50, ax=ax[3],
                      color=filtered_color,
                      edgecolor=hist_edge_color,
                      alpha=0.5,
                      label=f'Filtered Peaks (n={len(df_filtered)})')
            ax[3].legend(fontsize=8)
            
        ax[3].set_xlabel('Width (ms)')
        ax[3].set_ylabel('Count')
        ax[3].grid(True)
        ax[3].set_title('Width Distribution')
        
        # Include filtered peaks in statistics if they're shown
        total_peaks = len(df_all)
        if show_filtered_peaks and df_filtered is not None:
            filtered_count = len(df_filtered)
            total_with_filtered = total_peaks + filtered_count
            stats_text = (
                f'Mean: {df_all["width"].mean():.1f} ms\n'
                f'Median: {df_all["width"].median():.1f} ms\n'
                f'Std: {df_all["width"].std():.1f} ms\n'
                f'Min: {df_all["width"].min():.1f} ms\n'
                f'Max: {df_all["width"].max():.1f} ms\n'
                f'N: {total_peaks:,} peaks' +
                (f'\nFiltered: {filtered_count:,} peaks' if filtered_count > 0 else '') +
                (f'\nTotal: {total_with_filtered:,} peaks' if filtered_count > 0 else '')
            )
        else:
            stats_text = (
                f'Mean: {df_all["width"].mean():.1f} ms\n'
                f'Median: {df_all["width"].median():.1f} ms\n'
                f'Std: {df_all["width"].std():.1f} ms\n'
                f'Min: {df_all["width"].min():.1f} ms\n'
                f'Max: {df_all["width"].max():.1f} ms\n'
                f'N: {total_peaks:,} peaks')
                
        ax[3].text(0.95, 0.95, stats_text, transform=ax[3].transAxes,
                  fontsize=9, verticalalignment='top', horizontalalignment='right') # Keep font small

        # Main title (text color from apply_plot_theme)
        if show_filtered_peaks and df_filtered is not None and len(df_filtered) > 0:
            summary_stats = (
                f'Total Peaks: {total_peaks:,} | '
                f'Filtered Peaks: {len(df_filtered):,} | '
                f'Mean Area: {df_all["area"].mean():.1e} ± {df_all["area"].std():.1e} | '
                f'Mean Amplitude: {df_all["amplitude"].mean():.1f} ± {df_all["amplitude"].std():.1f}'
            )
        else:
            summary_stats = (
                f'Total Peaks: {total_peaks:,} | '
                f'Mean Area: {df_all["area"].mean():.1e} ± {df_all["area"].std():.1e} | '
                f'Mean Amplitude: {df_all["amplitude"].mean():.1f} ± {df_all["amplitude"].std():.1f}'
            )
            
        new_figure.suptitle('Peak Property Correlations\n' + summary_stats,
                           y=0.96) # Adjusted y slightly

        # Add legend explaining dot sizes
        # Add a small legend to explain dot sizes
        legend_text_1 = "Dot sizes in Width vs Amplitude plot represent peak area"
        legend_text_2 = "Dot sizes in Width vs Area plot represent peak amplitude"
        legend_text_3 = "Dot sizes in Amplitude vs Area plot represent peak width"
        new_figure.text(0.02, 0.03, legend_text_1 + "\n" + legend_text_2 + "\n" + legend_text_3, 
                        fontsize=8, color=app.theme_manager.get_plot_color('text'),
                        horizontalalignment='left', verticalalignment='bottom')

        # Apply theme standard styles (bg, grid, text)
        app.theme_manager.apply_plot_theme(new_figure, ax)

        # --- Update or create tab ---
        tab_name = "Peak Properties"
        tab_exists = False
        canvas = None

        for tab_widget_id in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab_widget_id, "text") == tab_name:
                tab_frame = app.plot_tab_control.nametowidget(tab_widget_id)
                app.plot_tab_control.select(tab_frame)
                # Remove old canvas
                for widget in tab_frame.winfo_children():
                    widget.destroy()
                # Add new canvas
                canvas = FigureCanvasTkAgg(new_figure, master=tab_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                # app.scatter_canvas = canvas # Optional: store specific canvas
                tab_exists = True
                break

        if not tab_exists:
            new_tab = ttk.Frame(app.plot_tab_control)
            app.plot_tab_control.add(new_tab, text=tab_name)
            app.plot_tab_control.select(new_tab)
            # Create and pack the canvas
            canvas = FigureCanvasTkAgg(new_figure, master=new_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            # app.scatter_canvas = canvas # Optional: store specific canvas

        # Store the figure associated with the tab
        app.tab_figures[tab_name] = new_figure

        app.preview_label.config(
            text="Peak properties plotted successfully",
            foreground=app.theme_manager.get_color('success')
            )

    except Exception as e:
        app.preview_label.config(
            text=f"Error creating scatter plot: {e}",
            foreground=app.theme_manager.get_color('error')
            )
        print(f"Detailed error: {str(e)}")
        traceback.print_exc() 