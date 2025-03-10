import numpy as np
import traceback
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy.signal import find_peaks
from peak_analysis_utils import *
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def plot_raw_data(app, profile_function=None):
    """
    Optimized plotting of raw data
    
    Parameters:
    -----------
    app : Application instance
        The main application instance containing necessary attributes and methods
    profile_function : decorator, optional
        Performance profiling decorator if available
    """
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

def start_analysis(app, profile_function=None):
    """Optimized analysis and plotting of filtered data"""
    if app.data is None:
        app.show_error("No data loaded. Please load files first.")
        return

    try:
        # Initialize progress
        total_steps = 4
        app.update_progress_bar(0, total_steps)

        # Get parameters and prepare data
        normalization_factor = app.normalization_factor.get()
        big_counts = app.big_counts.get()
        current_cutoff = app.cutoff_value.get()

        t = app.data['Time - Plot 0'].values * 1e-4
        x = app.data['Amplitude - Plot 0'].values
        rate = np.median(np.diff(t))
        app.fs = 1 / rate

        # Update progress
        app.update_progress_bar(1)

        # Apply filtering
        if current_cutoff > 0:
            app.filtered_signal = apply_butterworth_filter(2, current_cutoff, 'lowpass', app.fs, x)
            calculated_cutoff = current_cutoff
        else:
            app.filtered_signal, calculated_cutoff = adjust_lowpass_cutoff(
                x, app.fs, big_counts, normalization_factor
            )
            app.cutoff_value.set(calculated_cutoff)

        # Update progress
        app.update_progress_bar(2)

        # Create a common mask for both signals
        max_points = 10000
        if len(x) > max_points:
            # Calculate stride
            stride = len(x) // max_points

            # Create base mask
            mask = np.zeros(len(x), dtype=bool)
            mask[::stride] = True

            # Find peaks in both signals
            mean_x, std_x = np.mean(x), np.std(x)
            mean_filtered, std_filtered = np.mean(app.filtered_signal), np.std(app.filtered_signal)

            peaks_raw, _ = find_peaks(x, height=mean_x + 3 * std_x)
            peaks_filtered, _ = find_peaks(app.filtered_signal, height=mean_filtered + 3 * std_filtered)
            all_peaks = np.unique(np.concatenate([peaks_raw, peaks_filtered]))

            # Create peaks mask and expand peaks by convolution
            peaks_mask = np.zeros(len(x), dtype=bool)
            peaks_mask[all_peaks] = True
            peaks_mask = np.convolve(peaks_mask.astype(int), np.ones(11, dtype=int), mode='same') > 0

            # Find significant changes in both signals
            diff_raw = np.abs(np.diff(x, prepend=x[0]))
            diff_filtered = np.abs(np.diff(app.filtered_signal, prepend=app.filtered_signal[0]))

            threshold_raw = 5 * np.std(diff_raw)
            threshold_filtered = 5 * np.std(diff_filtered)

            changes_raw = np.where(diff_raw > threshold_raw)[0]
            changes_filtered = np.where(diff_filtered > threshold_filtered)[0]
            all_changes = np.unique(np.concatenate([changes_raw, changes_filtered]))

            # Create changes mask and expand changes by convolution
            changes_mask = np.zeros(len(x), dtype=bool)
            changes_mask[all_changes] = True
            changes_mask = np.convolve(changes_mask.astype(int), np.ones(3, dtype=int), mode='same') > 0

            # Combine masks
            mask |= peaks_mask | changes_mask

            # Apply mask to both signals
            t_plot = t[mask] / 60  # Convert to minutes
            x_plot = x[mask]
            filtered_plot = app.filtered_signal[mask]
        else:
            t_plot = t / 60
            x_plot = x
            filtered_plot = app.filtered_signal

        # Create plot
        app.figure.clear()
        ax = app.figure.add_subplot(111)

        # Plot decimated data
        ax.plot(
            t_plot,
            x_plot,
            color='black',
            linewidth=0.05,
            label=f'Raw Data ({len(x_plot):,} points)',
            alpha=0.4,
        )

        ax.plot(
            t_plot,
            filtered_plot,
            color='blue',
            linewidth=0.05,
            label=f'Filtered Data ({len(filtered_plot):,} points)',
            alpha=0.9,
        )

        # Customize plot
        ax.set_xlabel('Time (min)', fontsize=12)
        ax.set_ylabel('Amplitude (counts)', fontsize=12)
        ax.set_title('Raw and Filtered Signals (Optimized View)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10)

        # Add filtering parameters annotation
        filter_text = (
            f'Cutoff: {calculated_cutoff:.1f} Hz\n'
            f'Total points: {len(app.filtered_signal):,}\n'
            f'Plotted points: {len(filtered_plot):,}'
        )
        ax.text(
            0.02,
            0.98,
            filter_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.8),
        )

        # Update progress
        app.update_progress_bar(3)

        # Update or create tab
        tab_name = "Smoothed Data"
        tab_exists = False

        for tab in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab, "text") == tab_name:
                app.plot_tab_control.select(tab)
                tab_exists = True
                break

        if not tab_exists:
            new_tab = ttk.Frame(app.plot_tab_control)
            app.plot_tab_control.add(new_tab, text=tab_name)
            app.plot_tab_control.select(new_tab)
            app.canvas = FigureCanvasTkAgg(app.figure, new_tab)
            app.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Update the canvas
        app.canvas.draw_idle()

        # Final progress update
        app.update_progress_bar(4)

        # Update status
        app.preview_label.config(
            text=(
                f"Analysis completed (Cutoff: {calculated_cutoff:.1f} Hz, "
                f"Decimated from {len(app.filtered_signal):,} to {len(filtered_plot):,} points)"
            ),
            foreground="green",
        )

        app.tab_figures["Smoothed Data"] = app.figure

    except Exception as e:
        app.show_error("Error during analysis", e)
        app.update_progress_bar(0)

def run_peak_detection(app, profile_function=None):
    """Run peak detection and overlay peaks on existing plot"""
    if app.filtered_signal is None:
        app.show_error("Filtered signal not available. Please start the analysis first.")
        return

    try:
        # Initialize progress
        total_steps = 3
        app.update_progress_bar(0, total_steps)

        # Get the current axes
        ax = app.figure.gca()

        # Remove previously plotted peaks and width indicators
        lines_to_remove = []
        for line in ax.lines:
            # Check if this is a peak marker (red x marker) or not a main data line
            if (line.get_color() == 'red' and line.get_marker() == 'x') or \
               'Detected Peaks' in str(line.get_label()):
                lines_to_remove.append(line)

        # Remove the marked lines
        for line in lines_to_remove:
            line.remove()

        # Clear horizontal lines (peak width indicators)
        for collection in ax.collections:
            collection.remove()

        # Get parameters
        height_lim_factor = app.height_lim.get()
        distance = app.distance.get()
        rel_height = app.rel_height.get()
        width_p = [int(float(x) * 10) for x in app.width_p.get().split(',')]

        # Update progress
        app.update_progress_bar(1)

        # Find peaks
        peaks_x_filter, amp_x_filter = find_peaks_with_window(
            app.filtered_signal,
            width=width_p,
            prominence=height_lim_factor,
            distance=distance,
            rel_height=rel_height
        )

        # Check if any peaks were found
        if len(peaks_x_filter) == 0:
            app.show_error("No peaks found with current parameters. Try adjusting threshold or width range.")
            return

        # Calculate peak areas
        window = np.round(amp_x_filter['widths'], 0).astype(int) + 40
        peak_areas = np.zeros(len(peaks_x_filter))
        start_indices = np.zeros(len(peaks_x_filter))
        end_indices = np.zeros(len(peaks_x_filter))

        for i in range(len(peaks_x_filter)):
            # Get window indices
            start_idx = max(0, peaks_x_filter[i] - window[i])
            end_idx = min(len(app.filtered_signal), peaks_x_filter[i] + window[i])

            yData = app.filtered_signal[start_idx:end_idx]
            background = np.min(yData)

            st = int(amp_x_filter["left_ips"][i])
            en = int(amp_x_filter["right_ips"][i])

            start_indices[i] = st
            end_indices[i] = en
            peak_areas[i] = np.sum(app.filtered_signal[st:en] - background)

        # Update progress
        app.update_progress_bar(2)

        # Add peak markers to the existing plot
        ax.plot(app.t_value[peaks_x_filter]*1e-4 / 60,
                app.filtered_signal[peaks_x_filter],
                'rx',
                markersize=5,
                label=f'Detected Peaks ({len(peaks_x_filter)})')

        # Update legend
        ax.legend(fontsize=10)

        # Calculate peak intervals
        peak_times = app.t_value[peaks_x_filter]*1e-4
        intervals = np.diff(peak_times)

        if len(intervals) > 0:
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
        else:
            mean_interval = 0
            std_interval = 0

        # Update results summary
        summary_text = (
            f"Number of peaks detected: {len(peaks_x_filter)}\n"
            f"Average peak area: {np.mean(peak_areas):.2f} ± {np.std(peak_areas):.2f}\n"
            f"Average interval: {mean_interval:.2f} ± {std_interval:.2f} seconds\n"
            f"Peak detection threshold: {height_lim_factor}"
        )
        app.update_results_summary(summary_text)

        # Update canvas
        app.canvas.draw_idle()

        # Final progress update
        app.update_progress_bar(3)

        app.preview_label.config(
            text=f"Peak detection completed: {len(peaks_x_filter)} peaks detected",
            foreground="green"
        )

    except Exception as e:
        app.show_error("Error during peak detection", e)
        app.update_progress_bar(0)  # Reset on error

def plot_filtered_peaks(app, profile_function=None):
    if app.filtered_signal is None:
        app.preview_label.config(text="Filtered signal not available. Please start the analysis first.", foreground="red")
        return

    try:
        # Get peaks and properties
        width_values = app.width_p.get().strip().split(',')
        width_p = [int(float(value.strip()) * 10) for value in width_values]

        peaks_x_filter, amp_x_filter = find_peaks_with_window(
            app.filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get()
        )

        if len(peaks_x_filter) == 0:
            app.preview_label.config(text="No peaks found with current parameters", foreground="red")
            return

        # Divide measurement into segments and select representative peaks
        total_peaks = len(peaks_x_filter)
        num_segments = 10  # We want 10 peaks from different segments
        segment_size = total_peaks // num_segments

        # Store the current segment offset in the class if it doesn't exist
        if not hasattr(app, 'segment_offset'):
            app.segment_offset = 0

        # Select peaks from different segments
        selected_peaks = []
        for i in range(num_segments):
            segment_start = (i * segment_size + app.segment_offset) % total_peaks
            peak_idx = segment_start
            if peak_idx < total_peaks:
                selected_peaks.append(peak_idx)

        window = 3*np.round(amp_x_filter['widths'], 0).astype(int)

        # Create new figure
        new_figure = Figure(figsize=(10, 8))
        axs = []
        for i in range(2):
            row = []
            for j in range(5):
                row.append(new_figure.add_subplot(2, 5, i*5 + j + 1))
            axs.append(row)

        handles, labels = [], []

        # Plot selected peaks
        for idx, peak_idx in enumerate(selected_peaks):
            i = peak_idx
            start_idx = max(0, peaks_x_filter[i] - window[i])
            end_idx = min(len(app.t_value*1e-4), peaks_x_filter[i] + window[i])

            xData = app.t_value[start_idx:end_idx]*1e-4
            yData_sub = app.filtered_signal[start_idx:end_idx]

            if len(xData) == 0:
                continue

            background = np.min(yData_sub)
            yData = yData_sub - background

            ax = axs[idx // 5][idx % 5]

            # Plot filtered data
            line1, = ax.plot((xData - xData[0]) * 1e3, yData,
                           color='blue',
                           label='Filtered',
                           alpha=0.8,
                           linewidth=0.5)

            # Plot peak marker
            peak_time = app.t_value[peaks_x_filter[i]]*1e-4
            peak_height = app.filtered_signal[peaks_x_filter[i]] - background
            line2, = ax.plot((peak_time - xData[0]) * 1e3,
                           peak_height,
                           "x",
                           color='red',
                           ms=10,
                           label='Peak')

            # Plot raw data
            raw_data = app.x_value[start_idx:end_idx]
            corrected_signal = raw_data - background
            line3, = ax.plot((xData - xData[0]) * 1e3,
                           corrected_signal,
                           color='black',
                           label='Raw',
                           alpha=0.5,
                           linewidth=0.3)

            # Plot width lines
            left_idx = int(amp_x_filter["left_ips"][i])
            right_idx = int(amp_x_filter["right_ips"][i])
            width_height = amp_x_filter["width_heights"][i] - background

            line4 = ax.hlines(y=width_height,
                            xmin=(app.t_value[left_idx]*1e-4 - xData[0]) * 1e3,
                            xmax=(app.t_value[right_idx]*1e-4 - xData[0]) * 1e3,
                            color="red",
                            linestyles='-',
                            alpha=0.8)
            line4 = Line2D([0], [0], color='red', linestyle='-', label='Peak Width')

            # Add peak number label
            ax.text(0.02, 0.98, f'Peak #{i+1}',  # i+1 to start counting from 1 instead of 0
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(facecolor='white',
                             edgecolor='none',
                             alpha=0.7))

            # Customize subplot
            ax.set_xlabel('Time (ms)', fontsize=10)
            ax.set_ylabel('Counts', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(axis='both', labelsize=9)

            # Add padding to y-axis limits
            ymin, ymax = ax.get_ylim()
            y_padding = (ymax - ymin) * 0.15
            ax.set_ylim(ymin - y_padding, ymax + y_padding)

            if idx == 0:
                handles.extend([line3, line1, line2, line4])

        # Remove individual legends from subplots (with check)
        for ax_row in axs:
            for ax in ax_row:
                legend = ax.get_legend()
                if legend is not None:  # Only remove if legend exists
                    legend.remove()

        # Create handles for the legend (move this before the legend creation)
        handles = [
            Line2D([0], [0], color='black', alpha=0.5, linewidth=0.3, label='Raw Data'),
            Line2D([0], [0], color='blue', alpha=0.8, linewidth=0.5, label='Filtered Data'),
            Line2D([0], [0], color='red', marker='x', linestyle='None', label='Peak'),
            Line2D([0], [0], color='red', linestyle='-', alpha=0.8, label='Peak Width')
        ]

        # Add a single, optimized legend
        new_figure.legend(
            handles=handles,
            labels=['Raw Data', 'Filtered Data', 'Peak', 'Peak Width'],
            loc='center',
            bbox_to_anchor=(0.5, 0.98),
            ncol=4,
            fontsize=8,
            framealpha=0.9,
            edgecolor='gray',
            borderaxespad=0.5,
            columnspacing=1.0,
            handletextpad=0.5,
        )

        # Adjust the layout
        new_figure.subplots_adjust(top=0.92)
        new_figure.suptitle('Individual Peak Analysis', fontsize=12, y=0.96)
        new_figure.tight_layout(rect=[0, 0, 1, 0.92])

        # Update or create tab in plot_tab_control
        tab_name = "Exemplary Peaks"  # Changed from "Filtered Peaks Plot"
        tab_exists = False

        for tab in app.plot_tab_control.tabs():
            if app.plot_tab_control.tab(tab, "text") == tab_name:
                app.plot_tab_control.forget(tab)  # Remove existing tab to update it
                break

        new_tab = ttk.Frame(app.plot_tab_control)
        app.plot_tab_control.add(new_tab, text=tab_name)
        app.plot_tab_control.select(new_tab)

        new_canvas = FigureCanvasTkAgg(new_figure, new_tab)
        new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        new_canvas.draw_idle()

        app.tab_figures["Exemplary Peaks"] = new_figure

    except Exception as e:
        app.preview_label.config(text=f"Error plotting example peaks: {str(e)}", foreground="red")
        print(f"Detailed error: {str(e)}")
        traceback.print_exc()

def show_next_peaks(app):
    """Show the next set of peaks in the filtered peaks plot"""
    if not hasattr(app, 'segment_offset'):
        app.segment_offset = 0

    # Increment the offset
    app.segment_offset += 1

    # Reset to beginning if we've reached the end
    if app.segment_offset >= len(app.filtered_signal):
        app.segment_offset = 0
        app.preview_label.config(
            text="Reached end of peaks, returning to start",
            foreground="blue"
        )

    # Replot with new offset
    app.plot_filtered_peaks()

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
        t = np.asarray(app.t_value, dtype=np.float32)*1e-4
        filtered_signal = np.asarray(app.filtered_signal, dtype=np.float32)

        # Find peaks with optimized parameters
        width_values = app.width_p.get().strip().split(',')
        width_p = [int(float(value.strip()) * 10) for value in width_values]

        peaks, properties = find_peaks_with_window(
            filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get()
        )

        # Calculate peak properties
        widths = properties["widths"]
        prominences = properties["prominences"]
        peak_times = t[peaks]

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
        axes[0].set_yscale('log')

        # Plot peak widths
        axes[1].scatter(peak_times/60, widths/10, s=1, alpha=0.5, color='black', label='Peak Widths (ms)')
        axes[1].set_ylabel('Peak Widths (ms)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        axes[1].set_yscale('log')

        # Plot peak areas
        axes[2].scatter(peak_times/60, areas, s=1, alpha=0.5, color='black', label='Peak Areas')
        axes[2].set_ylabel('Peak Areas')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=8)
        axes[2].set_yscale('log')

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
        width_p = [int(float(value.strip()) * 10) for value in width_values]

        peaks_x_filter, properties = find_peaks_with_window(
            app.filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get()
        )

        # Calculate peak areas
        window = np.round(properties['widths'], 0).astype(int) + 40
        peak_areas = np.zeros(len(peaks_x_filter))

        for i in range(len(peaks_x_filter)):
            yData = app.filtered_signal[peaks_x_filter[i] - window[i]:peaks_x_filter[i] + window[i]]
            background = np.min(yData)
            st = int(properties["left_ips"][i])
            en = int(properties["right_ips"][i])
            peak_areas[i] = np.sum(app.filtered_signal[st:en] - background)

        # Create DataFrame with all peak properties
        df_all = pd.DataFrame({
            "width": properties['widths'] / 10,  # Convert to ms directly
            "amplitude": properties['prominences'],
            "area": peak_areas
        })

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
            f'N: {len(df_all):,}')
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


