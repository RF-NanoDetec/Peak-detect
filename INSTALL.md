# Installation Guide for Peak Analysis Tool

This guide provides instructions for installing and setting up the Peak Analysis Tool on different operating systems.

## Prerequisites

Before installing the Peak Analysis Tool, ensure you have the following prerequisites:

1. **Python 3.8+**: The application requires Python 3.8 or newer.
2. **Git**: Required if you're cloning the repository (optional).
3. **Virtual Environment**: Recommended for isolating the application dependencies.

## Windows Installation

### 1. Install Python

1. Download the latest Python version from [python.org](https://www.python.org/downloads/).
2. Run the installer and check "Add Python to PATH" during installation.
3. Verify the installation by opening Command Prompt and typing:
   ```
   python --version
   ```

### 2. Download the Application

#### Option A: Clone the Repository (if you have Git installed)
```
git clone https://github.com/your-repository/peak-analysis-tool.git
cd peak-analysis-tool
```

#### Option B: Download as ZIP
1. Download the ZIP file from the GitHub repository.
2. Extract the ZIP file to a location of your choice.
3. Open Command Prompt and navigate to the extracted folder.

### 3. Create and Activate a Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies

```
pip install -r requirements.txt
```

### 5. Run the Application

```
python app.py
```

Or use the batch file:
```
run_peak_analysis.bat
```

## macOS and Linux Installation

### 1. Install Python

#### macOS:
```
brew install python3
```

#### Ubuntu/Debian:
```
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### Fedora:
```
sudo dnf install python3 python3-pip
```

### 2. Download the Application

#### Option A: Clone the Repository
```
git clone https://github.com/your-repository/peak-analysis-tool.git
cd peak-analysis-tool
```

#### Option B: Download as ZIP
1. Download the ZIP file from the GitHub repository.
2. Extract the ZIP file to a location of your choice.
3. Open Terminal and navigate to the extracted folder.

### 3. Create and Activate a Virtual Environment

```
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```
pip install -r requirements.txt
```

### 5. Run the Application

```
python app.py
```

Or use the shell script:
```
chmod +x run_peak_analysis.sh
./run_peak_analysis.sh
```

## Troubleshooting

### Common Issues

1. **Missing tkinter**: If you encounter an error related to tkinter, install it:
   - **Windows**: Tkinter comes with Python installer. Reinstall Python with the "tcl/tk and IDLE" option.
   - **macOS**: `brew install python-tk`
   - **Ubuntu/Debian**: `sudo apt install python3-tk`
   - **Fedora**: `sudo dnf install python3-tkinter`

2. **Missing libraries**:
   ```
   pip install --upgrade -r requirements.txt
   ```

3. **Permission errors**:
   - **Windows**: Run Command Prompt as Administrator.
   - **macOS/Linux**: Use `sudo` where necessary.

### Logs Location

If the application fails to start or encounters errors, check the logs:
- Windows: `logs/app.log` in the application directory
- macOS/Linux: `logs/app.log` in the application directory

## Development Setup

For development purposes, you might want to install additional packages:

```
pip install -r requirements-dev.txt
```

## Building from Source

To build the application from source, follow these additional steps:

1. Install PyInstaller:
   ```
   pip install pyinstaller
   ```

2. Build the executable:
   ```
   pyinstaller --onefile --windowed --icon=resources/images/icon.ico app.py
   ```

3. The executable will be in the `dist` directory.

## Contact & Support

If you encounter any issues during installation or usage, please submit an issue on GitHub or contact the development team. 