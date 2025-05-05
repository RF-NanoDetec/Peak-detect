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
import matplotlib.pyplot as plt
from core.peak_detection import detect_peaks

def run_peak_detection(app, profile_function=None):
    """Run peak detection and overlay peaks on existing plot"""
    if app.filtered_signal is None:
        app.show_error("Filtered signal not available. Please start the analysis first.")
        return None

    if not hasattr(app, 'figure') or not app.figure.get_axes():
         app.show_error("No plot available to overlay peaks onto. Please run analysis first.")
         return None
    ax = app.figure.gca()

    try:
        # Initialize progress
        total_steps = 2
        app.update_progress_bar(0, total_steps)

        # Remove previously plotted peaks and width indicators
        lines_to_remove = []
        peak_marker_label = 'Detected Peaks'
        for line in ax.lines:
            if line.get_label() == peak_marker_label:
                lines_to_remove.append(line)

        for line in lines_to_remove:
            line.remove()

        collections_to_remove = []
        for collection in ax.collections:
            if isinstance(collection, plt.collections.LineCollection):
                collections_to_remove.append(collection)
        for collection in collections_to_remove:
            collection.remove()

        # Get peak detection parameters
        height_lim_factor = app.height_lim.get()
        distance = app.distance.get()
        rel_height = app.rel_height.get()
        width_values = app.width_p.get().strip().split(',')
        
        # Get the prominence ratio threshold
        prominence_ratio = app.prominence_ratio.get()

        # Get time resolution - handle both Tkinter variable and float value
        time_res = app.time_resolution.get() if hasattr(app.time_resolution, 'get') else app.time_resolution
        
        # --- Call Core Peak Detection Function --- 
        peaks_x_filter, properties = detect_peaks(
            signal=app.filtered_signal,
            time_resolution=time_res,
            height_lim=height_lim_factor,
            distance=distance,
            prominence_ratio=prominence_ratio,
            rel_height=rel_height,
            width_range=width_values # Pass width range in ms
        )
        
        # Check if detection was successful
        if peaks_x_filter is None or properties is None or len(peaks_x_filter) == 0:
            app.show_error("No peaks found with current parameters. Try adjusting threshold or width range.")
            if hasattr(app, 'canvas'): app.canvas.draw()
            return None, None, None # Return None for all values

        # Extract areas from properties (already calculated by detect_peaks)
        peak_areas = properties.get('areas', np.zeros(len(peaks_x_filter)))

        # Update progress
        app.update_progress_bar(1)

        # Get SEMANTIC color for peak markers
        peak_marker_color = app.theme_manager.get_plot_color('marker_peak')

        # Add peak markers using SEMANTIC theme color
        ax.plot(app.t_value[peaks_x_filter] / 60,
                app.filtered_signal[peaks_x_filter],
                marker='x',
                color=peak_marker_color,
                linestyle='None',
                markersize=5,
                label=peak_marker_label)

        # Update legend (fonts handled by apply_plot_theme)
        ax.legend()

        # Calculate peak intervals
        peak_times = app.t_value[peaks_x_filter]  # Time values already in seconds
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
        app.peak_heights = properties['prominences']
        app.peak_widths = properties['widths']

        # Apply theme standard styles (bg, grid, text)
        app.theme_manager.apply_plot_theme(app.figure, [ax])

        # Update figure
        if hasattr(app, 'canvas'):
             app.canvas.draw()
        else:
             print("Warning: app.canvas not found during peak detection update.")

        # Update progress and status using theme colors
        app.update_progress_bar(2)
        app.status_indicator.set_state('success')
        app.status_indicator.set_text(f"Peak detection completed, found {len(peaks_x_filter)} peaks")

        # Return peaks data
        return peaks_x_filter, properties, peak_areas

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
        app.preview_label.config(
            text="Filtered signal not available. Please start the analysis first.",
            foreground=app.theme_manager.get_color('error')
            )
        return False

    try:
        # Get peaks and properties
        width_values = app.width_p.get().strip().split(',')
        width_p = [int(float(value.strip()) * 10) for value in width_values]
        
        # Get the prominence ratio threshold
        prominence_ratio = app.prominence_ratio.get()

        peaks_x_filter, amp_x_filter = find_peaks_with_window(
            app.filtered_signal,
            width=width_p,
            prominence=app.height_lim.get(),
            distance=app.distance.get(),
            rel_height=app.rel_height.get(),
            prominence_ratio=prominence_ratio
        )

        if len(peaks_x_filter) == 0:
            app.preview_label.config(
                text="No peaks found with current parameters",
                foreground=app.theme_manager.get_color('warning')
                )
            return False

        # Divide measurement into segments and select representative peaks
        total_peaks = len(peaks_x_filter)
        num_segments = 10  # We want 10 peaks from different segments
        segment_size = total_peaks // num_segments

        # Store the current segment offset in the class if it doesn't exist
        if not hasattr(app, 'segment_offset'):
            app.segment_offset = 0

        # Ensure segment_offset is within valid range
        app.segment_offset = app.segment_offset % total_peaks

        # Select peaks from different segments
        selected_peaks = []
        for i in range(num_segments):
            # Calculate segment index with offset
            segment_idx = (i * segment_size + app.segment_offset) % total_peaks
            if segment_idx < total_peaks:
                selected_peaks.append(segment_idx)

        window = 3*np.round(amp_x_filter['widths'], 0).astype(int)

        # Get SEMANTIC theme colors
        filtered_line_color = app.theme_manager.get_plot_color('line_filtered')
        peak_marker_color = app.theme_manager.get_plot_color('marker_peak')
        raw_line_color = app.theme_manager.get_plot_color('line_raw')
        width_marker_color = app.theme_manager.get_plot_color('marker_width')

        # Create new figure for the grid
        new_figure = Figure(figsize=(12, 8))
        axs_flat = []
        for i in range(2):
            for j in range(5):
                ax = new_figure.add_subplot(2, 5, i*5 + j + 1)
                axs_flat.append(ax)

        # Plot selected peaks
        for idx, peak_idx in enumerate(selected_peaks):
            if idx >= len(axs_flat): break

            ax = axs_flat[idx]
            i = peak_idx
            start_idx = max(0, peaks_x_filter[i] - window[i])
            end_idx = min(len(app.t_value), peaks_x_filter[i] + window[i])

            xData = app.t_value[start_idx:end_idx]
            yData_sub = app.filtered_signal[start_idx:end_idx]

            if len(xData) == 0:
                continue

            background = np.min(yData_sub)
            yData = yData_sub - background

            # Plot filtered data segment using SEMANTIC color with thin line
            ax.plot((xData - xData[0]) * 1e3, yData,
                    color=filtered_line_color,
                    label=None,  # Remove label from actual plot
                    alpha=0.8,
                    linewidth=0.7)

            # Plot peak marker using SEMANTIC color
            peak_time = app.t_value[peaks_x_filter[i]]
            peak_height = app.filtered_signal[peaks_x_filter[i]] - background
            ax.plot((peak_time - xData[0]) * 1e3,
                    peak_height,
                    marker="x",
                    color=peak_marker_color,
                    linestyle='None',
                    ms=8,
                    label=None)  # Remove label from actual plot

            # Plot raw data segment using SEMANTIC color with thin line
            raw_data = app.x_value[start_idx:end_idx]
            corrected_signal = raw_data - background
            ax.plot((xData - xData[0]) * 1e3,
                    corrected_signal,
                    color=raw_line_color,
                    label=None,  # Remove label from actual plot
                    alpha=0.6,
                    linewidth=0.5)

            # Plot width lines using SEMANTIC color
            left_idx = int(amp_x_filter["left_ips"][i])
            right_idx = int(amp_x_filter["right_ips"][i])
            width_height = amp_x_filter["width_heights"][i] - background

            ax.hlines(y=width_height,
                     xmin=(app.t_value[left_idx] - xData[0]) * 1e3,
                     xmax=(app.t_value[right_idx] - xData[0]) * 1e3,
                     color=width_marker_color,
                     linestyles='-',
                     linewidth=1.0,
                     label=None)  # Remove label from actual plot

            # Customize subplot axes (fonts handled by apply_plot_theme)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (counts)")
            ax.set_title(f"Peak {peak_idx+1}", fontsize=10)
            ax.grid(True, linestyle=':')

            # Minimal ticks for clarity
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.tick_params(axis='both', which='major', labelsize=8)

        # Remove individual legends from subplots (with check)
        for ax in axs_flat:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # Create custom legend with thicker lines
        legend_lines = [
            Line2D([0], [0], color=filtered_line_color, linewidth=2.0, alpha=0.8, label='Filtered Data'),
            Line2D([0], [0], color=peak_marker_color, marker='x', linestyle='None', markersize=8, label='Peak'),
            Line2D([0], [0], color=raw_line_color, linewidth=2.0, alpha=0.6, label='Raw Data'),
            Line2D([0], [0], color=width_marker_color, linestyle='-', linewidth=2.0, label='Peak Width')
        ]

        # Add the custom legend with thicker lines in the top right
        new_figure.legend(handles=legend_lines, loc='upper right', ncol=2,
                         fontsize=9, bbox_to_anchor=(0.98, 0.98))

        # Adjust the layout - removed right margin adjustment since we don't need to shift plots
        new_figure.subplots_adjust(top=0.92)
        # Include offset in title to make navigation clearer
        new_figure.suptitle(f"Filtered Peaks Detail (Offset: {app.segment_offset}, showing 10 peaks)",
                           fontsize=14)
        new_figure.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Apply theme standard styles (bg, grid, text)
        app.theme_manager.apply_plot_theme(new_figure, axs_flat)

        # Update or create tab in plot_tab_control
        tab_name = "Filtered Peaks"
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
            canvas = FigureCanvasTkAgg(new_figure, master=new_tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Store the figure
        app.tab_figures[tab_name] = new_figure

        # Update status
        app.preview_label.config(
            text=f"Plotted {len(selected_peaks)} filtered peaks (offset: {app.segment_offset})",
            foreground=app.theme_manager.get_color('success')
        )
        
        return True

    except Exception as e:
        app.preview_label.config(
            text=f"Error plotting filtered peaks: {str(e)}",
            foreground=app.theme_manager.get_color('error')
        )
        traceback.print_exc()
        return False


def show_next_peaks(app, profile_function=None):
    """Navigate to the next set of peaks in the filtered peaks plot."""
    if not hasattr(app, 'segment_offset'):
        app.segment_offset = 0

    if app.filtered_signal is None or not hasattr(app, 'peaks') or len(app.peaks) == 0:
         app.preview_label.config(text="No peaks available to navigate.", foreground=app.theme_manager.get_color('warning'))
         return False

    total_peaks = len(app.peaks)
    
    # Increment offset by 1, wrapping around if necessary
    app.segment_offset = (app.segment_offset + 1) % total_peaks
    
    # Debug print
    print(f"Next peaks: Offset changed to {app.segment_offset}")

    # Trigger redraw
    success = plot_filtered_peaks(app, profile_function)
    if success:
         # Status is updated by plot_filtered_peaks
         pass
    else:
         # Error message is shown by plot_filtered_peaks
         app.segment_offset = (app.segment_offset - 1 + total_peaks) % total_peaks # Revert offset on error
         return False
    return True

def show_prev_peaks(app, profile_function=None):
    """Navigate to the previous set of peaks in the filtered peaks plot."""
    if not hasattr(app, 'segment_offset'):
        app.segment_offset = 0

    if app.filtered_signal is None or not hasattr(app, 'peaks') or len(app.peaks) == 0:
         app.preview_label.config(text="No peaks available to navigate.", foreground=app.theme_manager.get_color('warning'))
         return False

    total_peaks = len(app.peaks)
    
    # Decrement offset by 1, wrapping around if necessary
    app.segment_offset = (app.segment_offset - 1 + total_peaks) % total_peaks
    
    # Debug print
    print(f"Prev peaks: Offset changed to {app.segment_offset}")

    # Trigger redraw
    success = plot_filtered_peaks(app, profile_function)
    if success:
         # Status is updated by plot_filtered_peaks
         pass
    else:
         # Error message is shown by plot_filtered_peaks
         app.segment_offset = (app.segment_offset + 1) % total_peaks # Revert offset on error
         return False
    return True 