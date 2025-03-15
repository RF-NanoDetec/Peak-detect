# Codebase Reorganization

This document describes the reorganization of the Peak Analysis Tool codebase.

## Goals

The goal of this reorganization was to improve the codebase by:

1. **Modularity**: Organizing code into logical, focused modules
2. **Reusability**: Making components reusable across the application
3. **Maintainability**: Making it easier to update and extend the code
4. **Readability**: Improving code organization to make it easier to understand
5. **Performance**: Optimizing key operations for better performance

## Directory Structure Changes

From a flat structure, we moved to a more organized package-based structure:

### Before

```
peak_analysis_tool/
├── main.py
├── peak_detection.py
├── peak_analysis_utils.py
├── plot_functions.py
├── images/
├── config/
└── ui/
```

### After

```
peak_analysis_tool/
│
├── core/                  # Core functionality
│   ├── __init__.py
│   ├── peak_detection.py
│   └── peak_analysis_utils.py
│
├── plotting/              # Plotting functionality
│   ├── __init__.py
│   ├── raw_data.py
│   ├── data_processing.py
│   ├── peak_visualization.py
│   └── analysis_visualization.py
│
├── ui/                    # UI components
│   ├── __init__.py
│   ├── theme.py
│   ├── tooltips.py
│   └── status_indicator.py
│
├── config/                # Configuration
│   ├── __init__.py
│   └── settings.py
│
├── utils/                 # General utilities
│   ├── __init__.py
│   ├── performance.py
│   └── file_handling.py
│
├── resources/             # Static resources
│   └── images/
│
├── app.py                 # New application entry point
├── main.py                # Original application (to be refactored gradually)
├── run_peak_analysis.bat  # Windows launcher
├── run_peak_analysis.sh   # Unix launcher
└── README.md
```

## Specific Changes

### 1. Core Package

- Created `core/` package for essential algorithms and data structures
- Moved `peak_detection.py` to `core/peak_detection.py`
- Moved `peak_analysis_utils.py` to `core/peak_analysis_utils.py`
- Updated imports to use relative imports within the package

### 2. Utils Package

- Created `utils/` package for general utility functions
- Created `utils/performance.py` for profiling and performance monitoring
- Created `utils/file_handling.py` for file operations
- Extracted file handling code from the main application

### 3. Resources

- Created `resources/images/` directory for static resources
- Moved image files from `images/` to `resources/images/`
- Updated references to use the new paths

### 4. Entry Point

- Created `app.py` as a new entry point for the application
- Moved splash screen and setup code to `app.py`
- Created launcher scripts for Windows and Unix

## Future Improvements

1. **Complete Main Refactoring**: Continue refactoring `main.py` by moving UI components to appropriate modules
2. **Dependency Injection**: Implement proper dependency injection for better testability
3. **Configuration System**: Enhance the configuration system for better customization
4. **Testing**: Add unit and integration tests for key components
5. **Documentation**: Add more comprehensive documentation

## Benefits

This reorganization provides several benefits:

1. **Easier Maintenance**: Code is now organized in logical modules, making it easier to understand and maintain
2. **Better Separation of Concerns**: Each module has a clear responsibility
3. **Improved Reusability**: Utility functions and core algorithms can be reused across the application
4. **Clearer Entry Point**: New `app.py` provides a clean entry point for the application
5. **Resource Management**: Better organization of static resources

## Usage

The application can be started using:

- Windows: `run_peak_analysis.bat`
- Unix: `./run_peak_analysis.sh`
- Directly: `python app.py` 