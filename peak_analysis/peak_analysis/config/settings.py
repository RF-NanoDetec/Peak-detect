import os

class Config:
    """Configuration constants"""
    MAX_WORKERS = os.cpu_count() * 2
    MAX_PLOT_POINTS = 10000
    PROGRESS_RESET_DELAY = 500  # milliseconds
    DEFAULT_WINDOW_SIZE = (1920, 1080)
    
    class Colors:
        """Color constants"""
        ERROR = "red"
        SUCCESS = "green"
        INFO = "blue"
        WARNING = "orange"
    
    class Plot:
        """Plot configuration"""
        DPI = 100  # Increased from default 100
        FIGURE_SIZE = (12, 8)  # Inches
        EXPORT_DPI = 300  # Higher DPI for exports
        LINE_WIDTH = 0.5
        FONT_SIZE = 10
        TITLE_SIZE = 12
        LABEL_SIZE = 8