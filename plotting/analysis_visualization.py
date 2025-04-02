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
        app.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
        return

    try:
        print("Starting plot_data function...")

        # Create a new figure for data plot
        app.data_figure = Figure(figsize=(10, 8))

        # Create subplots with proper spacing
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
        
        # Calculate peak properties
        widths = properties["widths"]  # Width in samples
        widths_in_seconds = widths * rate  # Convert from samples to seconds
        widths_in_ms = widths_in_seconds * 1000  # Convert from seconds to milliseconds
        
        prominences = properties["prominences"]
        peak_times = t[peaks]
        
        # Debug output for peak widths
        if len(widths) > 0:
            print(f"\nDEBUG - Peak width information:")
            print(f"Raw widths (samples): {np.min(widths):.1f}/{np.mean(widths):.1f}/{np.max(widths):.1f}")
            print(f"Widths in seconds: {np.min(widths_in_seconds):.6f}/{np.mean(widths_in_seconds):.6f}/{np.max(widths_in_seconds):.6f}")
            print(f"Widths in ms: {np.min(widths_in_ms):.2f}/{np.mean(widths_in_ms):.2f}/{np.max(widths_in_ms):.2f}")
            
            # Calculate sampling rate for verification
            print(f"Median time difference between samples: {rate:.8f} seconds")
            print(f"Sampling rate: {sampling_rate:.1f} Hz")
            
            # Calculate average width in sample points
            print(f"Average width in sample points: {np.mean(widths):.2f}")
        
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

        # Plot peak heights
        axes[0].scatter(peak_times/60, prominences, s=1, alpha=0.5, color='black', label='Peak Heights')
        axes[0].set_ylabel('Peak Heights')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[0].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Plot peak widths
        axes[1].scatter(peak_times/60, widths_in_ms, s=1, alpha=0.5, color='black', label='Peak Widths (ms)')
        axes[1].set_ylabel('Peak Widths (ms)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        axes[1].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Plot peak areas
        axes[2].scatter(peak_times/60, areas, s=1, alpha=0.5, color='black', label='Peak Areas')
        axes[2].set_ylabel('Peak Areas')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=8)
        axes[2].set_yscale('log' if app.log_scale_enabled.get() else 'linear')

        # Calculate and plot throughput
        interval = 10  # seconds
        bins = np.arange(0, np.max(t), interval)
        bin_centers = (bins[:-1] + bins[1:]) / 2  # Calculate bin centers
        throughput, _ = np.histogram(peak_times, bins=bins)

        # Plot throughput with proper styling
        axes[3].bar(bin_centers/60, throughput,
                   width=(interval/60)*0.8,  # Adjust bar width
                   color='black',
                   alpha=0.5,
                   label=f'Throughput ({interval}s bins)')

        # Add moving average line
        window = 5  # Number of points for moving average
        moving_avg = np.convolve(throughput, np.ones(window)/window, mode='valid')
        moving_avg_times = bin_centers[window-1:]/60
        axes[3].plot(moving_avg_times, moving_avg,
                    color='red',
                    linewidth=1,
                    label=f'{window}-point Moving Average')

        axes[3].set_ylabel(f'Peaks per {interval}s')
        axes[3].set_xlabel('Time (min)')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(fontsize=8)

        # Add statistics annotation to throughput plot
        stats_text = (f'Total Peaks: {len(peaks):,}\n'
                     f'Avg Rate: {len(peaks)/(np.max(t)-np.min(t))*60:.1f} peaks/min\n'
                     f'Max Rate: {np.max(throughput)/(interval/60):.1f} peaks/min')
        axes[3].text(0.02, 0.98, stats_text,
                    transform=axes[3].transAxes,
                    verticalalignment='top',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8))

        # Update title and layout
        app.data_figure.suptitle('Peak Analysis Over Time', y=0.95)
        app.data_figure.tight_layout()

        # Create or update the tab in plot_tab_control
        tab_name = "Peak Analysis"
        tab_exists = False

        for tab in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab, "text") == tab_name:
                app.plot_tab_control.forget(tab)  # Remove existing tab to update it
                break

        new_tab = ttk.Frame(app.plot_tab_control)
        app.plot_tab_control.add(new_tab, text=tab_name)
        app.plot_tab_control.select(new_tab)

        # Create new canvas in the tab
        app.data_canvas = FigureCanvasTkAgg(app.data_figure, new_tab)
        app.data_canvas.draw()
        app.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        app.preview_label.config(text="Peak analysis plot created successfully", foreground="green")

        app.tab_figures["Peak Analysis"] = app.data_figure

    except Exception as e:
        app.show_error("Error in plot_data", e)
        raise  # Re-raise the exception for debugging


def plot_scatter(app, profile_function=None):
    """Enhanced scatter plot for peak property correlations"""
    if app.filtered_signal is None:
        app.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
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

        peaks_x_filter, properties = find_peaks_with_window(
            app.filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get()
        )
        
        # Additional debug information after peak detection
        print(f"DEBUG - Peak detection results in plot_scatter:")
        print(f"Found {len(peaks_x_filter)} peaks")
        if len(peaks_x_filter) > 0:
            print(f"Peak properties keys: {list(properties.keys())}")
            print(f"First few peak positions: {peaks_x_filter[:5]}")
            if 'widths' in properties:
                print(f"First few peak widths (in samples): {properties['widths'][:5]}")
                print(f"Peak widths in seconds (min/mean/max): {np.min(properties['widths']*rate):.6f}/{np.mean(properties['widths']*rate):.6f}/{np.max(properties['widths']*rate):.6f}")
                print(f"Peak widths in ms (min/mean/max): {np.min(properties['widths']*rate*1000):.2f}/{np.mean(properties['widths']*rate*1000):.2f}/{np.max(properties['widths']*rate*1000):.2f}")
        
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
            yData = app.filtered_signal[peaks_x_filter[i] - window[i]:peaks_x_filter[i] + window[i]]
            background = np.min(yData)
            st = int(properties["left_ips"][i])
            en = int(properties["right_ips"][i])
            peak_areas[i] = np.sum(app.filtered_signal[st:en] - background)

        # Create DataFrame with all peak properties
        df_all = pd.DataFrame({
            "width": widths_in_ms,  # Use widths_in_ms directly
            "amplitude": prominences,
            "area": peak_areas
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

        # Create new figure with adjusted size and spacing
        new_figure = Figure(figsize=(12, 10))
        gs = new_figure.add_gridspec(2, 2, hspace=0.25, wspace=0.3)
        ax = [new_figure.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

        # Color map for density
        cmap = plt.cm.viridis

        # Plot 1: Width vs Amplitude
        density1 = stats.gaussian_kde(np.vstack([df_all['width'], df_all['amplitude']]))
        density1_points = density1(np.vstack([df_all['width'], df_all['amplitude']]))

        sc1 = ax[0].scatter(df_all['width'], df_all['amplitude'],
                          c=density1_points,
                          s=5,
                          alpha=0.6,
                          cmap=cmap)
        ax[0].set_xlabel('Width (ms)', fontsize=10)
        ax[0].set_ylabel('Amplitude (counts)', fontsize=10)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].grid(True, alpha=0.3)
        ax[0].set_title('Width vs Amplitude', fontsize=12)

        # Add correlation coefficient
        corr1 = df_all['width'].corr(df_all['amplitude'])
        ax[0].text(0.05, 0.95, f'r = {corr1:.2f}',
                  transform=ax[0].transAxes,
                  fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8))

        # Plot 2: Width vs Area
        density2 = stats.gaussian_kde(np.vstack([df_all['width'], df_all['area']]))
        density2_points = density2(np.vstack([df_all['width'], df_all['area']]))

        sc2 = ax[1].scatter(df_all['width'], df_all['area'],
                          c=density2_points,
                          s=5,
                          alpha=0.6,
                          cmap=cmap)
        ax[1].set_xlabel('Width (ms)', fontsize=10)
        ax[1].set_ylabel('Area (counts)', fontsize=10)
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].grid(True, alpha=0.3)
        ax[1].set_title('Width vs Area', fontsize=12)

        corr2 = df_all['width'].corr(df_all['area'])
        ax[1].text(0.05, 0.95, f'r = {corr2:.2f}',
                  transform=ax[1].transAxes,
                  fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8))

        # Plot 3: Amplitude vs Area
        density3 = stats.gaussian_kde(np.vstack([df_all['amplitude'], df_all['area']]))
        density3_points = density3(np.vstack([df_all['amplitude'], df_all['area']]))

        sc3 = ax[2].scatter(df_all['amplitude'], df_all['area'],
                          c=density3_points,
                          s=5,
                          alpha=0.6,
                          cmap=cmap)
        ax[2].set_xlabel('Amplitude (counts)', fontsize=10)
        ax[2].set_ylabel('Area (counts)', fontsize=10)
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].grid(True, alpha=0.3)
        ax[2].set_title('Amplitude vs Area', fontsize=12)

        corr3 = df_all['amplitude'].corr(df_all['area'])
        ax[2].text(0.05, 0.95, f'r = {corr3:.2f}',
                  transform=ax[2].transAxes,
                  fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8))

        # Plot 4: Width distribution with statistics
        import seaborn as sns
        sns.histplot(data=df_all, x='width', bins=50, ax=ax[3], color='darkblue', alpha=0.6)
        ax[3].set_xlabel('Width (ms)', fontsize=10)
        ax[3].set_ylabel('Count', fontsize=10)
        ax[3].grid(True, alpha=0.3)
        ax[3].set_title('Width Distribution', fontsize=12)

        # Add statistics to histogram
        stats_text = (
            f'Mean: {df_all["width"].mean():.1f} ms\n'
            f'Median: {df_all["width"].median():.1f} ms\n'
            f'Std: {df_all["width"].std():.1f} ms\n'
            f'Min: {df_all["width"].min():.1f} ms\n'
            f'Max: {df_all["width"].max():.1f} ms\n'
            f'N: {len(df_all):,} peaks')
        ax[3].text(0.95, 0.95, stats_text,
                  transform=ax[3].transAxes,
                  fontsize=9,
                  bbox=dict(facecolor='white', alpha=0.8),
                  verticalalignment='top',
                  horizontalalignment='right')

        # Add colorbar for density
        cbar_ax = new_figure.add_axes([0.92, 0.15, 0.02, 0.7])
        new_figure.colorbar(sc1, cax=cbar_ax, label='Density')

        # Main title with summary statistics
        summary_stats = (
            f'Total Peaks: {len(peaks_x_filter):,} | '
            f'Mean Area: {df_all["area"].mean():.1e} ± {df_all["area"].std():.1e} | '
            f'Mean Amplitude: {df_all["amplitude"].mean():.1f} ± {df_all["amplitude"].std():.1f}'
        )
        new_figure.suptitle('Peak Property Correlations\n' + summary_stats,
                           y=0.95, fontsize=14)

        # Update or create tab in plot_tab_control
        tab_name = "Peak Properties"
        tab_exists = False

        for tab in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab, "text") == tab_name:
                app.plot_tab_control.forget(tab)
                break

        new_tab = ttk.Frame(app.plot_tab_control)
        app.plot_tab_control.add(new_tab, text=tab_name)
        app.plot_tab_control.select(new_tab)

        new_canvas = FigureCanvasTkAgg(new_figure, new_tab)
        new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        new_canvas.draw_idle()

        app.preview_label.config(text="Peak properties plotted successfully", foreground="green")
        app.tab_figures["Peak Properties"] = new_figure

    except Exception as e:
        app.preview_label.config(text=f"Error creating scatter plot: {e}", foreground="red")
        print(f"Detailed error: {str(e)}")
        traceback.print_exc() 