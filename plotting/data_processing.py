"""
Data processing functions for the Peak Analysis Tool.
"""

import numpy as np
import traceback
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy.signal import find_peaks
from core.peak_analysis_utils import apply_butterworth_filter, adjust_lowpass_cutoff

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