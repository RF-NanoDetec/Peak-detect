"""
Peak visualization functions for the Peak Analysis Tool.

This module contains functions for detecting and visualizing peaks in data.
"""

import numpy as np
import traceback
from scipy.signal import find_peaks
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from matplotlib.lines import Line2D
from core.peak_analysis_utils import find_peaks_with_window

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
        width_values = app.width_p.get().strip().split(',')

        # Update progress
        app.update_progress_bar(1)

        # Reset PeakDetector
        app.peak_detector.reset()
        
        # Use PeakDetector instance to detect peaks
        app.peak_detector.detect_peaks(
            app.filtered_signal,
            app.t_value,
            height_lim_factor,
            distance,
            rel_height,
            width_values
        )
        
        # Calculate peak areas using the PeakDetector
        peak_areas, start_indices, end_indices = app.peak_detector.calculate_peak_areas(app.filtered_signal)
        
        # Get detected peaks
        peaks_x_filter = app.peak_detector.peaks_indices
        amp_x_filter = app.peak_detector.peaks_properties

        # Check if any peaks were found
        if len(peaks_x_filter) == 0:
            app.show_error("No peaks found with current parameters. Try adjusting threshold or width range.")
            return

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

        app.update_results_summary(events=len(peaks_x_filter), peak_areas=peak_areas, preview_text=summary_text)

        # Set or update app.peaks property for use in other functions
        app.peaks = peaks_x_filter
        app.peak_heights = amp_x_filter['prominences']
        app.peak_widths = amp_x_filter['widths']

        # Update figure
        app.canvas.draw()

        # Update progress and status
        app.update_progress_bar(3)
        app.status_indicator.set_state('success')
        app.status_indicator.set_text(f"Peak detection completed, found {len(peaks_x_filter)} peaks")

        # Return peaks data
        return peaks_x_filter, amp_x_filter, peak_areas

    except Exception as e:
        # Show error in status and provide traceback
        error_msg = str(e)
        app.status_indicator.set_state('error')
        app.status_indicator.set_text(f"Error: {error_msg}")
        app.show_error("Peak Detection Error", traceback.format_exc())
        return None


def plot_filtered_peaks(app, profile_function=None):
    """Plot individual peaks in a grid layout for detailed analysis"""
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


def show_next_peaks(app, profile_function=None):
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

    # Call plot_filtered_peaks with the same profile_function parameter
    return plot_filtered_peaks(app, profile_function) 