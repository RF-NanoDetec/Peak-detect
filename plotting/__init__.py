"""
Plotting package for the Peak Analysis Tool.

This package contains modules for various plotting and visualization functions:
- raw_data: Functions for displaying raw data
- data_processing: Functions for processing and filtering data
- peak_visualization: Functions for visualizing peaks
- analysis_visualization: Functions for data analysis visualization
"""

# This file intentionally left mostly empty for now.
# We're gradually migrating functions from plot_functions.py to this package.
# Each function will be moved to its appropriate module first.

# Import the modules to make their functions available when importing from plotting
from . import raw_data
from . import data_processing
from . import peak_visualization
from . import analysis_visualization

# These will be uncommented as we move functions to these modules
# from .analysis_visualization import * 