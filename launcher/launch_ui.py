"""
Peak Analysis Tool Launcher
Starts the backend service and opens the web UI in the default browser.
"""

import os
import socket
import subprocess
import sys
import time
import webbrowser

HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}"


def is_listening(host: str, port: int) -> bool:
    """Check if a port is already being listened to."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False


def main():
    """Main launcher function."""
    print("=================================")
    print("Peak Analysis Tool Launcher")
    print("=================================")
    print()

    # Try to find the service executable
    # First, check if it's in the same directory (installed version)
    exe_name = "PeakService.exe"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Possible locations for the exe
    possible_paths = [
        os.path.join(script_dir, exe_name),  # Same directory as launcher
        os.path.join(script_dir, "..", exe_name),  # Parent directory
        os.path.join(script_dir, "..", "dist", exe_name),  # Development dist folder
    ]

    exe_path = None
    for path in possible_paths:
        if os.path.exists(path):
            exe_path = path
            break

    # Check if service is already running
    if is_listening(HOST, PORT):
        print(f"✓ Service already running on {URL}")
    else:
        if exe_path:
            print(f"Starting service: {exe_path}")
            try:
                # Start the service in background
                subprocess.Popen(
                    [exe_path],
                    close_fds=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                    if sys.platform == "win32"
                    else 0,
                )

                # Wait for service to start
                print("Waiting for service to start", end="")
                for i in range(30):  # Wait up to 6 seconds
                    if is_listening(HOST, PORT):
                        print(" Done!")
                        break
                    print(".", end="", flush=True)
                    time.sleep(0.2)
                else:
                    print()
                    print("⚠ Service may not have started properly")
            except Exception as e:
                print(f"✗ Error starting service: {e}")
                input("Press Enter to exit...")
                return
        else:
            print("✗ Service executable not found!")
            print("   Searched in:")
            for path in possible_paths:
                print(f"   - {path}")
            print()
            print(
                "Please ensure PeakService.exe is in the same directory as this launcher."
            )
            input("Press Enter to exit...")
            return

    # Open the web UI in browser
    print(f"Opening web UI: {URL}")
    try:
        webbrowser.open(URL)
        print()
        print("✓ Browser should open automatically")
        print(f"  If not, navigate to: {URL}")
        print()
        print("To stop the service, close this window or use Task Manager")
    except Exception as e:
        print(f"✗ Could not open browser: {e}")
        print(f"  Please open manually: {URL}")


if __name__ == "__main__":
    main()
    # Keep the window open so user can see the output
    input("\nPress Enter to exit...")
