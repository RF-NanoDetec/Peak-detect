"""
Plotting module for the Peak Analysis Tool.
Provides visualization functionality for signals, peaks, and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector, RectangleSelector
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import logging
from dataclasses import dataclass

from ..config import config
from ..peak_detection import Peak

logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    """Configuration for plot appearance."""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    style: str = 'seaborn'
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'signal': 'blue',
                'filtered': 'green',
                'peaks': 'red',
                'baseline': 'black',
                'background': 'white',
                'grid': 'gray'
            }

class PlotManager:
    """Manages plot creation and updates."""
    
    def __init__(self, master, config: Optional[PlotConfig] = None):
        """
        Initialize plot manager.
        
        Args:
            master: Tkinter master widget
            config: Optional plot configuration
        """
        self.master = master
        self.config = config or PlotConfig()
        self.figures: Dict[str, Figure] = {}
        self.canvases: Dict[str, FigureCanvasTkAgg] = {}
        self.toolbars: Dict[str, NavigationToolbar2Tk] = {}
        
        # Set plot style
        plt.style.use(self.config.style)
        
    def create_figure(
        self,
        name: str,
        parent=None,
        add_toolbar: bool = True
    ) -> Tuple[Figure, FigureCanvasTkAgg]:
        """
        Create a new figure and canvas.
        
        Args:
            name: Unique identifier for the figure
            parent: Parent widget (defaults to master)
            add_toolbar: Whether to add navigation toolbar
            
        Returns:
            Tuple of (figure, canvas)
        """
        try:
            parent = parent or self.master
            
            # Create figure
            fig = Figure(figsize=self.config.figsize, dpi=self.config.dpi)
            fig.patch.set_facecolor(self.config.colors['background'])
            
            # Create canvas
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill='both', expand=True)
            
            # Add toolbar if requested
            if add_toolbar:
                toolbar = NavigationToolbar2Tk(canvas, parent)
                toolbar.update()
                self.toolbars[name] = toolbar
            
            # Store references
            self.figures[name] = fig
            self.canvases[name] = canvas
            
            return fig, canvas
            
        except Exception as e:
            logger.error(f"Error creating figure {name}: {str(e)}")
            raise
    
    def plot_signal(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        filtered: Optional[np.ndarray] = None,
        peaks: Optional[List[Peak]] = None,
        baseline: Optional[np.ndarray] = None,
        figure_name: str = 'main',
        title: str = 'Signal Analysis'
    ) -> None:
        """
        Plot signal data with optional components.
        
        Args:
            time: Time array
            signal: Original signal array
            filtered: Optional filtered signal array
            peaks: Optional list of detected peaks
            baseline: Optional baseline array
            figure_name: Name of figure to plot on
            title: Plot title
        """
        try:
            # Get or create figure
            if figure_name not in self.figures:
                fig, _ = self.create_figure(figure_name)
            else:
                fig = self.figures[figure_name]
                fig.clear()
            
            # Create subplot
            ax = fig.add_subplot(111)
            
            # Plot original signal
            ax.plot(
                time, signal,
                color=self.config.colors['signal'],
                alpha=0.5,
                label='Original Signal'
            )
            
            # Plot filtered signal if available
            if filtered is not None:
                ax.plot(
                    time, filtered,
                    color=self.config.colors['filtered'],
                    label='Filtered Signal'
                )
            
            # Plot baseline if available
            if baseline is not None:
                ax.plot(
                    time, baseline,
                    color=self.config.colors['baseline'],
                    linestyle='--',
                    label='Baseline'
                )
            
            # Plot peaks if available
            if peaks is not None:
                peak_indices = [peak.index for peak in peaks]
                peak_heights = [peak.height for peak in peaks]
                ax.scatter(
                    time[peak_indices],
                    peak_heights,
                    color=self.config.colors['peaks'],
                    marker='x',
                    s=100,
                    label='Peaks'
                )
            
            # Customize plot
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title(title)
            ax.grid(True, color=self.config.colors['grid'], alpha=0.3)
            ax.legend()
            
            # Update canvas
            self.canvases[figure_name].draw()
            
        except Exception as e:
            logger.error(f"Error plotting signal: {str(e)}")
            raise
    
    def plot_peak_details(
        self,
        peak: Peak,
        time: np.ndarray,
        signal: np.ndarray,
        window: int = 100,
        figure_name: str = 'peak_detail'
    ) -> None:
        """
        Plot detailed view of a single peak.
        
        Args:
            peak: Peak object to plot
            time: Time array
            signal: Signal array
            window: Number of points around peak to show
            figure_name: Name of figure to plot on
        """
        try:
            # Get or create figure
            if figure_name not in self.figures:
                fig, _ = self.create_figure(figure_name)
            else:
                fig = self.figures[figure_name]
                fig.clear()
            
            # Create subplot
            ax = fig.add_subplot(111)
            
            # Calculate window indices
            start_idx = max(0, peak.index - window)
            end_idx = min(len(signal), peak.index + window)
            
            # Plot peak region
            ax.plot(
                time[start_idx:end_idx],
                signal[start_idx:end_idx],
                color=self.config.colors['signal']
            )
            
            # Mark peak
            ax.axvline(
                time[peak.index],
                color=self.config.colors['peaks'],
                linestyle='--',
                alpha=0.5
            )
            
            # Add peak information
            info_text = (
                f"Height: {peak.height:.2f}\n"
                f"Width: {peak.width:.2f}\n"
                f"Area: {peak.area:.2f}\n"
                f"Prominence: {peak.prominence:.2f}"
            )
            ax.text(
                0.95, 0.95,
                info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            # Customize plot
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Peak Detail (Index: {peak.index})')
            ax.grid(True, color=self.config.colors['grid'], alpha=0.3)
            
            # Update canvas
            self.canvases[figure_name].draw()
            
        except Exception as e:
            logger.error(f"Error plotting peak details: {str(e)}")
            raise
    
    def plot_statistics(
        self,
        peaks: List[Peak],
        figure_name: str = 'statistics'
    ) -> None:
        """
        Plot statistical analysis of peaks.
        
        Args:
            peaks: List of Peak objects
            figure_name: Name of figure to plot on
        """
        try:
            # Get or create figure
            if figure_name not in self.figures:
                fig, _ = self.create_figure(figure_name)
            else:
                fig = self.figures[figure_name]
                fig.clear()
            
            # Create subplots
            fig.subplots_adjust(hspace=0.4)
            axes = fig.subplots(2, 2)
            
            # Extract peak properties
            heights = [p.height for p in peaks]
            widths = [p.width for p in peaks]
            areas = [p.area for p in peaks]
            prominences = [p.prominence for p in peaks]
            
            # Plot histograms
            axes[0, 0].hist(heights, bins=30, color=self.config.colors['peaks'])
            axes[0, 0].set_title('Peak Heights')
            
            axes[0, 1].hist(widths, bins=30, color=self.config.colors['peaks'])
            axes[0, 1].set_title('Peak Widths')
            
            axes[1, 0].hist(areas, bins=30, color=self.config.colors['peaks'])
            axes[1, 0].set_title('Peak Areas')
            
            axes[1, 1].hist(prominences, bins=30, color=self.config.colors['peaks'])
            axes[1, 1].set_title('Peak Prominences')
            
            # Update canvas
            self.canvases[figure_name].draw()
            
        except Exception as e:
            logger.error(f"Error plotting statistics: {str(e)}")
            raise
    
    def clear_figure(self, name: str) -> None:
        """
        Clear specified figure.
        
        Args:
            name: Name of figure to clear
        """
        if name in self.figures:
            self.figures[name].clear()
            self.canvases[name].draw()
    
    def remove_figure(self, name: str) -> None:
        """
        Remove specified figure and associated widgets.
        
        Args:
            name: Name of figure to remove
        """
        if name in self.figures:
            if name in self.toolbars:
                self.toolbars[name].destroy()
                del self.toolbars[name]
            
            self.canvases[name].get_tk_widget().destroy()
            del self.canvases[name]
            del self.figures[name]

# Example usage
if __name__ == "__main__":
    import tkinter as tk
    
    # Create test window
    root = tk.Tk()
    root.geometry("800x600")
    
    # Create plot manager
    plot_manager = PlotManager(root)
    
    # Generate test data
    t = np.linspace(0, 10, 1000)
    s = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, len(t))
    
    # Create test peak
    test_peak = Peak(
        index=500,
        height=1.2,
        width=0.5,
        prominence=1.0,
        left_base=480,
        right_base=520,
        area=2.0,
        center_mass=5.0
    )
    
    # Plot test data
    plot_manager.plot_signal(t, s, peaks=[test_peak])
    
    root.mainloop()
