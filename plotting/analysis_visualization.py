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
from core.peak_analysis_utils import (
    find_peaks_with_window,
    compute_baseline_mask,
    compute_noise_stats,
    compute_snr_values,
)
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
        return False

    try:
        import logging
        logging.getLogger(__name__).debug("Starting plot_data function...")

        # Create a new figure for data plot
        # Figure background color is handled by theme manager
        app.data_figure = Figure(figsize=(10, 8))

        # Create subplots
        axes = app.data_figure.subplots(nrows=4, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios': [1, 1, 1, 1.2],
                                                   'hspace': 0.3})
        
        # Apply theme immediately to prevent white background flash
        app.theme_manager.apply_plot_theme(app.data_figure, axes)

        # Convert data to float32 for memory efficiency
        t = np.asarray(app.t_value, dtype=np.float32)  # Time already in seconds
        filtered_signal = np.asarray(app.filtered_signal, dtype=np.float32)

        # Get time resolution
        rate = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution
        if rate <= 0:
            print(f"WARNING: Invalid time resolution ({rate}), using default of 0.0001")
            rate = 0.0001  # Default to 0.1ms sampling if invalid value
        
        sampling_rate = 1 / rate
        print(f"DEBUG - plot_data sampling rate: {sampling_rate:.1f} Hz (from time resolution {rate})")
        
        # Convert width from ms to samples
        from core.data_utils import convert_width_ms_to_samples
        width_p = convert_width_ms_to_samples(app.width_p.get(), sampling_rate)
        
        # Debug output
        log = logging.getLogger(__name__)
        log.debug("\nDEBUG - Width parameter information in plot_data:")
        log.debug(f"Original width values: {app.width_p.get()} (ms)")
        log.debug(f"Sampling rate: {sampling_rate:.1f} Hz")
        log.debug(f"Converted width_p (in samples): {width_p}")
        
        # Get the prominence ratio threshold
        prominence_ratio = app.prominence_ratio.get()
        
        # First, find all peaks without applying the prominence ratio filter
        from scipy.signal import find_peaks, peak_widths
        unfiltered_peaks, unfiltered_properties = find_peaks(
            filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get()
        )
        
        # Calculate widths for unfiltered peaks
        if len(unfiltered_peaks) > 0:
            width_results = peak_widths(filtered_signal, unfiltered_peaks, rel_height=app.rel_height.get())
            unfiltered_properties['widths'] = width_results[0]
            unfiltered_properties['width_heights'] = width_results[1]
            unfiltered_properties['left_ips'] = width_results[2]
            unfiltered_properties['right_ips'] = width_results[3]
        
        # Now get the peaks with the prominence ratio filter applied
        peaks, properties = find_peaks_with_window(
            filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get(),
            prominence_ratio=prominence_ratio
        )
        
        # Additional debug information after peak detection
        log.debug("DEBUG - Peak detection results:")
        log.debug(f"Found {len(peaks)} peaks (after prominence ratio filter)")
        log.debug(f"Found {len(unfiltered_peaks)} peaks (before prominence ratio filter)")
        if len(peaks) > 0:
            log.debug(f"Peak properties keys: {list(properties.keys())}")
            log.debug(f"First few peak positions: {peaks[:5]}")
            if 'widths' in properties:
                log.debug(f"First few peak widths (in samples): {properties['widths'][:5]}")
                log.debug(f"Peak widths in seconds (min/mean/max): {np.min(properties['widths']*rate):.6f}/{np.mean(properties['widths']*rate):.6f}/{np.max(properties['widths']*rate):.6f}")
                log.debug(f"Peak widths in ms (min/mean/max): {np.min(properties['widths']*rate*1000):.2f}/{np.mean(properties['widths']*rate*1000):.2f}/{np.max(properties['widths']*rate*1000):.2f}")
        
        # Identify filtered peaks (those in unfiltered_peaks but not in peaks)
        filtered_out_peaks = []
        filtered_out_properties = {}
        
        if hasattr(app, 'show_filtered_peaks') and app.show_filtered_peaks.get():
            # Find indices of peaks that are in unfiltered_peaks but not in peaks
            filtered_out_indices = []
            
            # This requires a bit of set logic, but numpy arrays aren't hashable
            # So we convert to lists or use np.isin
            if len(unfiltered_peaks) > 0:
                filtered_mask = np.ones(len(unfiltered_peaks), dtype=bool)
                
                for i, peak_idx in enumerate(unfiltered_peaks):
                    # Check if this peak is present in the filtered peaks list
                    if np.any(peaks == peak_idx):
                        filtered_mask[i] = False
                
                # Get the filtered out peaks
                filtered_out_peaks = unfiltered_peaks[filtered_mask]
                
                # Extract their properties
                for key, values in unfiltered_properties.items():
                    if isinstance(values, np.ndarray) and len(values) == len(unfiltered_peaks):
                        if key not in filtered_out_properties:
                            filtered_out_properties[key] = values[filtered_mask]
            
            log.debug(f"DEBUG - Found {len(filtered_out_peaks)} peaks that would be filtered out")
        
        # Calculate peak properties
        widths = properties["widths"]  # Width in samples
        widths_in_seconds = widths * rate  # Convert from samples to seconds
        widths_in_ms = widths_in_seconds * 1000  # Convert from seconds to milliseconds
        
        prominences = properties["prominences"]
        peak_times = t[peaks]

        # Persist peak properties on app for downstream summaries/export
        app.peaks = peaks
        app.peak_heights = prominences
        app.peak_widths = widths
        app.peak_left_ips = properties["left_ips"] if "left_ips" in properties else None
        app.peak_right_ips = properties["right_ips"] if "right_ips" in properties else None
        
        # Also calculate properties for filtered-out peaks if available
        if len(filtered_out_peaks) > 0:
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
        if len(filtered_out_peaks) > 0 and 'left_ips' in filtered_out_properties and 'right_ips' in filtered_out_properties:
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
        if len(filtered_out_peaks) > 0 and hasattr(app, 'show_filtered_peaks') and app.show_filtered_peaks.get():
            axes[0].scatter(filtered_peak_times/60, filtered_prominences, s=2, alpha=0.7, color=filtered_color, label='Filtered Peaks')
        axes[0].set_ylabel('Peak Heights')
        axes[0].grid(True) # alpha/color from rcParams
        axes[0].legend()
        axes[0].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Plot peak widths - CHANGED ORDER: First regular peaks, then filtered peaks on top
        axes[1].scatter(peak_times/60, widths_in_ms, s=1, alpha=0.5, color=scatter_color, label='Peak Widths (ms)')
        # Add filtered peaks if toggle is enabled (these will now appear on top)
        if len(filtered_out_peaks) > 0 and hasattr(app, 'show_filtered_peaks') and app.show_filtered_peaks.get():
            axes[1].scatter(filtered_peak_times/60, filtered_widths_in_ms, s=2, alpha=0.7, color=filtered_color, label='Filtered Peaks')
        axes[1].set_ylabel('Peak Widths (ms)')
        axes[1].grid(True)
        axes[1].legend()
        axes[1].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Plot peak areas - CHANGED ORDER: First regular peaks, then filtered peaks on top
        axes[2].scatter(peak_times/60, areas, s=1, alpha=0.5, color=scatter_color, label='Peak Areas')
        # Add filtered peaks if toggle is enabled (these will now appear on top)
        if len(filtered_out_peaks) > 0 and len(filtered_areas) > 0 and hasattr(app, 'show_filtered_peaks') and app.show_filtered_peaks.get():
            axes[2].scatter(filtered_peak_times/60, filtered_areas, s=2, alpha=0.7, color=filtered_color, label='Filtered Peaks')
        axes[2].set_ylabel('Peak Areas')
        axes[2].grid(True)
        axes[2].legend()
        axes[2].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Calculate and plot throughput
        interval = app.throughput_interval.get() if hasattr(app, 'throughput_interval') else 10  # seconds
        # Include filtered peaks in throughput calculation if toggle is enabled
        if len(filtered_out_peaks) > 0 and hasattr(app, 'show_filtered_peaks') and app.show_filtered_peaks.get():
            all_peak_times = np.concatenate([peak_times, filtered_peak_times])
            bins = np.arange(0, np.max(all_peak_times) + interval, interval)
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
            bins = np.arange(0, np.max(t) + interval, interval)
            bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
            throughput, _ = np.histogram(peak_times, bins=bins)
            
            # Plot throughput bars using SEMANTIC color
            axes[3].bar(bin_centers/60, throughput, width=(interval/60)*0.8, color=bar_color, alpha=0.6, 
                      label=f'Peak Throughput ({interval:.0f}s bins)')

        # Add moving average line using SEMANTIC color
        window = 5  # Number of points for moving average
        if len(throughput) >= window:
            moving_avg = np.convolve(throughput, np.ones(window)/window, mode='valid')
            moving_avg_times = bin_centers[window-1:]/60
            axes[3].plot(moving_avg_times, moving_avg,
                        color=moving_avg_color,
                        linewidth=1, label=f'{window}-point Moving Average')

        # Add statistics annotation to throughput plot
        stats_text = (f'Total Peaks: {len(peaks):,}\n'
                     f'Avg Rate: {len(peaks)/(np.max(t)-np.min(t))*60:.1f} peaks/min\n'
                     f'Max Rate: {np.max(throughput) if len(throughput) > 0 else 0:.1f} peaks/{interval}s\n'
                     f'Interval: {interval:.0f} s')
        axes[3].text(0.02, 0.98, stats_text,
                    transform=axes[3].transAxes,
                    verticalalignment='top',
                    fontsize=8)

        axes[3].set_ylabel(f'Peaks per {interval}s')
        axes[3].set_xlabel('Time (min)')
        axes[3].grid(True)
        axes[3].legend()

        # Update title and layout
        app.data_figure.suptitle('Peak Analysis Over Time', y=0.95)
        app.data_figure.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust rect for suptitle

        # Apply theme again to ensure everything is properly styled
        app.theme_manager.apply_plot_theme(app.data_figure, axes)
        # Use canvas background on axes for readability in dark mode
        for ax in axes:
            ax.set_facecolor(app.theme_manager.get_color('canvas_bg'))

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

        # Update statistics in UI
        total_peaks = len(peaks)
        filtered_out = len(filtered_out_peaks) if len(filtered_out_peaks) > 0 else 0
        filtered_kept = total_peaks
        
        # Update summary in UI
        preview_text = (
            f"Peaks: {filtered_kept} kept, {filtered_out} filtered out\n"
            f"Prominence ratio threshold: {prominence_ratio:.2f}\n"
            f"Average peak area: {np.mean(areas) if len(areas) > 0 else 0:.2f} ± {np.std(areas) if len(areas) > 0 else 0:.2f}"
        )
        
        # Update the right panel results summary
        if hasattr(app, 'results_summary'):
            # Use UI utility to update summary
            app.update_results_summary_with_ui = getattr(app, 'update_results_summary_with_ui', None)
            if callable(app.update_results_summary_with_ui):
                app.update_results_summary_with_ui(events=filtered_kept, peak_areas=areas, preview_text=preview_text)
            else:
                from ui.ui_utils import update_results_summary_with_ui
                update_results_summary_with_ui(app, events=filtered_kept, peak_areas=areas, preview_text=preview_text)

        app.preview_label.config(
            text="Peak analysis plot created successfully",
            foreground=app.theme_manager.get_color('success')
            )
            
        return True

    except Exception as e:
        import traceback
        import logging
        logging.getLogger(__name__).error(f"Error in plot_data: {str(e)}\n{traceback.format_exc()}")
        app.preview_label.config(
            text=f"Error creating peak analysis plot: {str(e)}",
            foreground=app.theme_manager.get_color('error')
            )
        return False


def plot_scatter(app, profile_function=None):
    """Enhanced scatter plot for peak property correlations and SNR distribution"""
    if app.filtered_signal is None:
        app.preview_label.config(
            text="Filtered signal not available. Please start the analysis first.",
            foreground=app.theme_manager.get_color('error')
            )
        return False

    try:
        # Create new figure
        # Figure bg color handled by theme manager
        new_figure = Figure(figsize=(12, 10))
        gs = new_figure.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        ax = [new_figure.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
        
        # Apply theme immediately to prevent white background flash
        app.theme_manager.apply_plot_theme(new_figure, ax)

        # Get time resolution for converting widths from samples to seconds
        rate = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution
        if rate <= 0:
            print(f"WARNING: Invalid time resolution ({rate}), using default of 0.0001")
            rate = 0.0001  # Default to 0.1ms sampling if invalid value
        
        # Convert width parameter to samples
        width_values = app.width_p.get().strip().split(',')
        sampling_rate = 1 / rate
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        
        # Debug output
        print(f"\nDEBUG - Width parameter information in plot_scatter:")
        print(f"Original width values: {app.width_p.get()} (ms)")
        print(f"Sampling rate: {sampling_rate:.1f} Hz")
        print(f"Converted width_p (in samples): {width_p}")

        # Check if we should show filtered peaks
        show_filtered_peaks = hasattr(app, 'show_filtered_peaks') and app.show_filtered_peaks.get()
        prominence_ratio = app.prominence_ratio.get()
        
        # First, find all peaks without applying the prominence ratio filter
        t = np.asarray(app.t_value, dtype=np.float32)
        from scipy.signal import find_peaks, peak_widths
        unfiltered_peaks, unfiltered_properties = find_peaks(
            app.filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get()
        )
        
        # Calculate widths for unfiltered peaks
        if len(unfiltered_peaks) > 0:
            width_results = peak_widths(app.filtered_signal, unfiltered_peaks, rel_height=app.rel_height.get())
            unfiltered_properties['widths'] = width_results[0]
            unfiltered_properties['width_heights'] = width_results[1]
            unfiltered_properties['left_ips'] = width_results[2]
            unfiltered_properties['right_ips'] = width_results[3]
        
        # Now get peaks with the prominence ratio filter applied
        peaks, properties = find_peaks_with_window(
            app.filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get(),
            prominence_ratio=prominence_ratio
        )
        
        if len(peaks) == 0:
            app.preview_label.config(
                text="No peaks found with current parameters. Try adjusting threshold or width range.",
                foreground=app.theme_manager.get_color('warning')
            )
            return False
        
        # Identify filtered peaks (those in unfiltered_peaks but not in peaks)
        filtered_out_peaks = []
        filtered_out_properties = {}
        
        if show_filtered_peaks:
            # Find indices of peaks that are in unfiltered_peaks but not in peaks
            if len(unfiltered_peaks) > 0:
                filtered_mask = np.ones(len(unfiltered_peaks), dtype=bool)
                
                for i, peak_idx in enumerate(unfiltered_peaks):
                    # Check if this peak is present in the filtered peaks list
                    if np.any(peaks == peak_idx):
                        filtered_mask[i] = False
                
                # Get the filtered out peaks
                filtered_out_peaks = unfiltered_peaks[filtered_mask]
                
                # Extract their properties
                for key, values in unfiltered_properties.items():
                    if isinstance(values, np.ndarray) and len(values) == len(unfiltered_peaks):
                        if key not in filtered_out_properties:
                            filtered_out_properties[key] = values[filtered_mask]
            
            print(f"DEBUG - Found {len(filtered_out_peaks)} peaks that would be filtered out")
        
        # Calculate peak properties
        prominences = properties['prominences']
        widths = properties['widths']  # Width in samples
        widths_in_seconds = widths * rate  # Convert from samples to seconds
        widths_in_ms = widths_in_seconds * 1000  # Convert from seconds to milliseconds
        
        # Calculate filtered peak properties if available
        if len(filtered_out_peaks) > 0:
            filtered_prominences = filtered_out_properties['prominences']
            filtered_widths = filtered_out_properties['widths']
            filtered_widths_in_ms = filtered_widths * rate * 1000
        
        # Calculate areas under peaks
        areas = []
        for i, peak in enumerate(peaks):
            left_idx = int(properties["left_ips"][i])
            right_idx = int(properties["right_ips"][i])
            if left_idx < right_idx:
                areas.append(np.trapz(app.filtered_signal[left_idx:right_idx]))
            else:
                areas.append(0)
        areas = np.array(areas)
        
        # Calculate areas for filtered peaks if available
        filtered_areas = []
        if len(filtered_out_peaks) > 0 and 'left_ips' in filtered_out_properties and 'right_ips' in filtered_out_properties:
            for i, peak in enumerate(filtered_out_peaks):
                left_idx = int(filtered_out_properties["left_ips"][i])
                right_idx = int(filtered_out_properties["right_ips"][i])
                if left_idx < right_idx:
                    filtered_areas.append(np.trapz(app.filtered_signal[left_idx:right_idx]))
                else:
                    filtered_areas.append(0)
            filtered_areas = np.array(filtered_areas)
        
        # Get SEMANTIC theme colors for plotting elements
        scatter_color = app.theme_manager.get_plot_color('scatter_points')
        filtered_color = "#FF8080"  # Light red for filtered peaks
        hist_color = app.theme_manager.get_plot_color('hist_bars')
        hist_edge_color = app.theme_manager.get_plot_color('patch.edgecolor')

        # Compute baseline-based noise and SNR values
        baseline_mask = compute_baseline_mask(
            signal_length=len(app.filtered_signal),
            peak_indices=peaks,
            widths_in_samples=properties['widths'] if 'widths' in properties else np.array([]),
            multiplier=2.0,
            left_indices=(properties['left_ips'] if 'left_ips' in properties else None),
            right_indices=(properties['right_ips'] if 'right_ips' in properties else None),
        )
        baseline_signal = app.filtered_signal[baseline_mask]
        noise_std, noise_mad_std, baseline_mean = compute_noise_stats(baseline_signal)
        snr_values = compute_snr_values(prominences, noise_std)

        # Persist for reuse
        app.noise_std = noise_std
        app.noise_mad_std = noise_mad_std
        app.baseline_mean = baseline_mean
        app.snr_values = snr_values

        # Parameters for scatter plots
        scatter_size = 4  # Base size for dots
        scatter_alpha = 0.2  # Alpha for visibility
        
        # Plot 1: Prominence vs Width
        ax[0].scatter(prominences, widths_in_ms, s=scatter_size, alpha=scatter_alpha, color=scatter_color, label='Kept Peaks')
        if show_filtered_peaks and len(filtered_out_peaks) > 0:
            ax[0].scatter(filtered_prominences, filtered_widths_in_ms, s=scatter_size, alpha=scatter_alpha, color=filtered_color, label='Filtered Peaks')
            
        ax[0].set_xlabel('Peak Prominence')
        ax[0].set_ylabel('Width (ms)')
        ax[0].set_title('Prominence vs Width')
        ax[0].grid(True)
        ax[0].legend(fontsize=8)
        
        # Plot 2: Prominence vs Area
        ax[1].scatter(prominences, areas, s=scatter_size, alpha=scatter_alpha, color=scatter_color, label='Kept Peaks')
        if show_filtered_peaks and len(filtered_out_peaks) > 0 and len(filtered_areas) > 0:
            ax[1].scatter(filtered_prominences, filtered_areas, s=scatter_size, alpha=scatter_alpha, 
                        color=filtered_color, label=f'Filtered Peaks ({len(filtered_out_peaks)})')
                
        ax[1].set_xlabel('Peak Prominence')
        ax[1].set_ylabel('Area')
        ax[1].set_title('Prominence vs Area')
        ax[1].grid(True)
        ax[1].legend(fontsize=8)
        
        # Plot 3: Width vs Area
        ax[2].scatter(widths_in_ms, areas, s=scatter_size, alpha=scatter_alpha, color=scatter_color, label='Kept Peaks')
        if show_filtered_peaks and len(filtered_out_peaks) > 0 and len(filtered_areas) > 0:
            ax[2].scatter(filtered_widths_in_ms, filtered_areas, s=scatter_size, alpha=scatter_alpha, 
                        color=filtered_color, label=f'Filtered Peaks ({len(filtered_out_peaks)})')
        ax[2].set_xlabel('Width (ms)')
        ax[2].set_ylabel('Area')
        ax[2].set_title('Width vs Area')
        ax[2].grid(True)
        ax[2].legend(fontsize=8)
        
        # Plot 4: Width distribution histogram
        import seaborn as sns
        
        # Create a pandas Series for the histogram
        width_series = pd.Series(widths_in_ms, name='Width (ms)')
        sns.histplot(width_series, bins=30, color=hist_color, edgecolor=hist_edge_color, alpha=0.7, ax=ax[3])
        
        if show_filtered_peaks and len(filtered_out_peaks) > 0:
            filtered_width_series = pd.Series(filtered_widths_in_ms, name='Width (ms)')
            sns.histplot(filtered_width_series, bins=30, color=filtered_color, edgecolor=hist_edge_color, 
                        alpha=0.5, ax=ax[3], label=f'Filtered ({len(filtered_out_peaks)})')
            ax[3].legend(fontsize=8)
            
        ax[3].set_xlabel('Width (ms)')
        ax[3].set_ylabel('Count')
        ax[3].set_title('Width Distribution')
        ax[3].grid(True)

        # Plot 5: SNR distribution histogram
        if snr_values.size > 0:
            sns.histplot(pd.Series(snr_values, name='SNR'), bins=30, color=hist_color, edgecolor=hist_edge_color,
                        alpha=0.7, ax=ax[4])
            ax[4].set_xlabel('SNR (Height / Noise Std)')
            ax[4].set_ylabel('Count')
            ax[4].set_title('SNR Distribution')
            ax[4].grid(True)
            # Annotate stats
            snr_stats = (
                f"Mean: {np.mean(snr_values):.2f}\n"
                f"Median: {np.median(snr_values):.2f}\n"
                f"SD: {np.std(snr_values):.2f}\n"
                f"Min/Max: {np.min(snr_values):.2f}/{np.max(snr_values):.2f}\n"
                f"N: {len(snr_values)}"
            )
            ax[4].text(0.95, 0.95, snr_stats, transform=ax[4].transAxes,
                      fontsize=8, verticalalignment='top', horizontalalignment='right')
        else:
            ax[4].axis('off')

        # Plot 6: SNR over time scatter
        if snr_values.size > 0:
            peak_times = t[peaks]
            ax[5].scatter(peak_times/60, snr_values, s=scatter_size, alpha=scatter_alpha, color=scatter_color)
            ax[5].set_xlabel('Time (min)')
            ax[5].set_ylabel('SNR')
            ax[5].set_title('SNR over Time')
            ax[5].grid(True)
        else:
            ax[5].axis('off')
        
        # Add statistics to width distribution plot
        stats_text = (
            f'Mean: {np.mean(widths_in_ms):.1f} ms\n'
            f'Median: {np.median(widths_in_ms):.1f} ms\n'
            f'SD: {np.std(widths_in_ms):.1f} ms\n'
            f'Min: {np.min(widths_in_ms):.1f} ms\n'
            f'Max: {np.max(widths_in_ms):.1f} ms\n'
            f'N: {len(peaks)}'
        )
        ax[3].text(0.95, 0.95, stats_text, transform=ax[3].transAxes,
                  fontsize=8, verticalalignment='top', horizontalalignment='right')
        
        # Update all axes with correct scale
        for i, axis in enumerate(ax):
            if i < 3:  # Only apply log scale to the first three plots
                axis.set_xscale('log' if app.log_scale_enabled.get() else 'linear')
                axis.set_yscale('log' if app.log_scale_enabled.get() else 'linear')
            else:  # Histograms and time scatter in linear scale
                axis.set_xscale('linear')
                axis.set_yscale('linear')
        
        # Add main title with summary statistics
        total_peaks = len(peaks)
        filtered_count = len(filtered_out_peaks) if len(filtered_out_peaks) > 0 else 0
        
        summary_text = (
            f'Peak Property Analysis - Total: {total_peaks} peaks'
            + (f', Filtered: {filtered_count} peaks' if filtered_count > 0 else '')
            + f' - Prominence Ratio: {prominence_ratio:.2f}'
            + (f' - Noise Std: {noise_std:.3g}' if noise_std > 0 else '')
        )
        new_figure.suptitle(summary_text, fontsize=12, y=0.98)
        
        # Apply theme again to ensure everything is properly styled
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

        # Store the figure associated with the tab
        app.tab_figures[tab_name] = new_figure

        app.preview_label.config(
            text="Peak properties plot created successfully",
            foreground=app.theme_manager.get_color('success')
            )
            
        return True

    except Exception as e:
        import traceback
        print(f"Error in plot_scatter: {str(e)}")
        traceback.print_exc()
        app.preview_label.config(
            text=f"Error creating scatter plot: {str(e)}",
            foreground=app.theme_manager.get_color('error')
            )
        return False 