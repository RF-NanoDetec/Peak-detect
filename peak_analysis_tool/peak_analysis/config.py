"""
Configuration settings for the Peak Analysis Tool.

This module contains all configuration constants and settings used throughout the application.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Tuple, Dict

@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    MAX_WORKERS: int = os.cpu_count() * 2
    LOGGING_LEVEL: int = logging.INFO
    LOGGING_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOGGING_FILE: str = 'peak_analysis.log'
    TEMP_DIR: str = os.path.join(os.path.expanduser('~'), '.peak_analysis', 'temp')
    CACHE_DIR: str = os.path.join(os.path.expanduser('~'), '.peak_analysis', 'cache')

@dataclass
class GuiConfig:
    """GUI-specific configuration settings."""
    WINDOW_SIZE: Tuple[int, int] = (1920, 1080)
    MIN_WINDOW_SIZE: Tuple[int, int] = (800, 600)
    PADDING: int = 5
    WIDGET_WIDTH: int = 15
    
    # Font configurations
    FONTS: Dict[str, tuple] = field(default_factory=lambda: {
        'default': ('Arial', 10),
        'title': ('Arial', 12, 'bold'),
        'header': ('Arial', 11, 'bold'),
        'small': ('Arial', 8),
        'monospace': ('Courier', 10)
    })
    
    # Color scheme
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'bg': '#f0f0f0',
        'fg': '#333333',
        'highlight': '#0078d7',
        'error': '#ff0000',
        'success': '#00ff00',
        'warning': '#ffa500',
        'info': '#0000ff'
    })
    
    # Theme settings
    THEME: str = 'clam'  # Available: 'clam', 'alt', 'default', 'classic'

@dataclass
class PlotConfig:
    """Plot-specific configuration settings."""
    DPI: int = 100
    EXPORT_DPI: int = 300
    FIGURE_SIZE: Tuple[float, float] = (12, 8)
    LINE_WIDTH: float = 0.5
    FONT_SIZE: int = 10
    TITLE_SIZE: int = 12
    LABEL_SIZE: int = 8
    MAX_PLOT_POINTS: int = 10000
    
    # Plot style settings
    STYLE: str = 'seaborn'  # Available: 'seaborn', 'ggplot', 'classic', etc.
    GRID: bool = True
    GRID_STYLE: str = '--'
    GRID_ALPHA: float = 0.7
    
    # Color maps
    COLORMAP: str = 'viridis'
    PEAK_COLOR: str = 'red'
    BASELINE_COLOR: str = 'blue'
    
    # Export settings
    EXPORT_FORMATS: Tuple[str, ...] = ('png', 'pdf', 'svg')
    DEFAULT_EXPORT_FORMAT: str = 'png'

@dataclass
class AnalysisConfig:
    """Analysis-specific configuration settings."""
    # Default analysis parameters
    DEFAULT_NORMALIZATION: float = 1.0
    DEFAULT_BIG_COUNTS: int = 100
    DEFAULT_HEIGHT_LIM: float = 20.0
    DEFAULT_DISTANCE: int = 30
    DEFAULT_REL_HEIGHT: float = 0.85
    DEFAULT_WIDTH_RANGE: str = "1,200"
    
    # Signal processing parameters
    BUTTERWORTH_ORDER: int = 2
    MIN_CUTOFF_FREQ: float = 50.0
    MAX_CUTOFF_FREQ: float = 10000.0
    
    # Peak detection parameters
    MIN_PEAK_DISTANCE: int = 10
    MIN_PEAK_WIDTH: int = 1
    MAX_PEAK_WIDTH: int = 2000
    PEAK_PROMINENCE_FACTOR: float = 0.1
    
    # Analysis thresholds
    NOISE_THRESHOLD: float = 3.0  # Standard deviations above mean
    OUTLIER_THRESHOLD: float = 2.5  # IQR multiplier for outlier detection
    
    # Processing chunks
    CHUNK_SIZE: int = 1000000  # Number of points to process at once
    OVERLAP: int = 1000  # Overlap between chunks

@dataclass
class FileConfig:
    """File handling configuration settings."""
    # Supported file formats
    SUPPORTED_FORMATS: Tuple[str, ...] = ('.txt', '.csv', '.dat')
    DEFAULT_DELIMITER: str = '\t'
    ENCODING: str = 'utf-8'
    
    # Column names
    TIME_COLUMN: str = 'Time - Plot 0'
    AMPLITUDE_COLUMN: str = 'Amplitude - Plot 0'
    
    # Export settings
    EXPORT_DELIMITER: str = ','
    EXPORT_DECIMAL: str = '.'
    EXPORT_ENCODING: str = 'utf-8'
    
    # File paths
    DEFAULT_SAVE_DIR: str = os.path.expanduser('~/Documents/Peak_Analysis')
    DEFAULT_LOAD_DIR: str = os.path.expanduser('~/Documents')

class Config:
    """Main configuration class that combines all config components."""
    
    def __init__(self):
        """Initialize configuration components."""
        self.system = SystemConfig()
        self.gui = GuiConfig()
        self.plot = PlotConfig()
        self.analysis = AnalysisConfig()
        self.file = FileConfig()
        
        # Create necessary directories
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.system.TEMP_DIR,
            self.system.CACHE_DIR,
            self.file.DEFAULT_SAVE_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            level=self.system.LOGGING_LEVEL,
            format=self.system.LOGGING_FORMAT,
            filename=self.system.LOGGING_FILE
        )
    
    @property
    def version(self) -> str:
        """Return the current version of the configuration."""
        return "1.0.0"
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'system': vars(self.system),
            'gui': vars(self.gui),
            'plot': vars(self.plot),
            'analysis': vars(self.analysis),
            'file': vars(self.file)
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'Config':
        """Load configuration from file."""
        import json
        config = cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        for section, values in data.items():
            if hasattr(config, section):
                section_config = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        return config

# Create global configuration instance
config = Config()

# Export for easy access
system_config = config.system
gui_config = config.gui
plot_config = config.plot
analysis_config = config.analysis
file_config = config.file

def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def reset_config():
    """Reset configuration to default values."""
    global config
    config = Config()

# Example usage:
if __name__ == "__main__":
    # Access configuration values
    print(f"Max workers: {config.system.MAX_WORKERS}")
    print(f"Window size: {config.gui.WINDOW_SIZE}")
    print(f"Plot DPI: {config.plot.DPI}")
    print(f"Default normalization: {config.analysis.DEFAULT_NORMALIZATION}")
    print(f"Supported formats: {config.file.SUPPORTED_FORMATS}")
    
    # Save configuration
    config.save_config('config.json')
    
    # Load configuration
    new_config = Config.load_config('config.json')
