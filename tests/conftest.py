import os

# Force a headless Matplotlib backend for test runs to avoid Tk/Tcl requirements
os.environ.setdefault("MPLBACKEND", "Agg")


