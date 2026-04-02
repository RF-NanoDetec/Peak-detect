"""
Utility script to fetch and display performance timing data from the API.

Usage:
    python tools/get_timing_data.py [result_id]
    
If no result_id is provided, shows all available timing data.
"""

import sys
import json
import requests
from typing import Dict, Any, Optional

BASE_URL = "http://127.0.0.1:8765/api"


def format_timing_report(timing_data: Dict[str, Any], result_id: str = "") -> str:
    """Format timing data into a readable report."""
    lines = []
    
    if result_id:
        lines.append(f"\n{'='*70}")
        lines.append(f"Timing Report for: {result_id}")
        lines.append(f"{'='*70}")
    else:
        lines.append(f"\n{'='*70}")
        lines.append(f"Timing Report: {timing_data.get('operation', 'unknown')}")
        lines.append(f"{'='*70}")
    
    total_time = timing_data.get('total_time_seconds', 0)
    lines.append(f"Total Time: {total_time:.3f} seconds ({total_time*1000:.1f} ms)")
    lines.append("")
    
    phases = timing_data.get('phases', {})
    if phases:
        lines.append("Phase Breakdown:")
        lines.append(f"{'Phase Name':<40} {'Time (s)':<12} {'%':<8} {'Calls':<8}")
        lines.append("-" * 70)
        
        # Sort by time (descending)
        sorted_phases = sorted(
            phases.items(),
            key=lambda x: x[1].get('total_seconds', 0),
            reverse=True
        )
        
        for phase_name, phase_data in sorted_phases:
            time_sec = phase_data.get('total_seconds', 0)
            percentage = phase_data.get('percentage', 0)
            call_count = phase_data.get('call_count', 0)
            lines.append(
                f"{phase_name:<40} {time_sec:>11.3f}s  {percentage:>6.1f}%  {call_count:>7}"
            )
    
    slowest = timing_data.get('slowest_phase', {})
    if slowest and slowest.get('name'):
        lines.append("")
        lines.append(f"Slowest Phase: {slowest['name']}")
        lines.append(f"  Time: {slowest.get('time_seconds', 0):.3f}s")
        lines.append(f"  Percentage: {slowest.get('percentage', 0):.1f}%")
    
    metadata = timing_data.get('metadata', {})
    if metadata:
        lines.append("")
        lines.append("Metadata:")
        for key, value in metadata.items():
            lines.append(f"  {key}: {value}")
    
    lines.append(f"{'='*70}\n")
    return "\n".join(lines)


def get_timing(result_id: str) -> Optional[Dict[str, Any]]:
    """Get timing data for a specific result ID."""
    try:
        response = requests.get(f"{BASE_URL}/performance/timing/{result_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            print(f"❌ No timing data found for result ID: {result_id}")
            return None
        else:
            print(f"❌ Error fetching timing data: {response.status_code}")
            print(f"   {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")
        print(f"   Expected server at: {BASE_URL}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def get_all_timings() -> Dict[str, Dict[str, Any]]:
    """Get all timing data."""
    try:
        response = requests.get(f"{BASE_URL}/performance/timing", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error fetching timing data: {response.status_code}")
            print(f"   {response.text}")
            return {}
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")
        print(f"   Expected server at: {BASE_URL}")
        return {}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


def save_timing_to_file(timing_data: Dict[str, Any], filename: str):
    """Save timing data to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(timing_data, f, indent=2)
        print(f"✅ Saved timing data to: {filename}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")


def main():
    """Main function."""
    print("Performance Timing Data Viewer")
    print("=" * 70)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Server is not responding correctly")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running!")
        print(f"   Please start the server: python -m service.app")
        sys.exit(1)
    
    # Get result ID from command line or show all
    if len(sys.argv) > 1:
        result_id = sys.argv[1]
        print(f"\nFetching timing data for: {result_id}")
        timing_data = get_timing(result_id)
        
        if timing_data:
            print(format_timing_report(timing_data, result_id))
            
            # Ask if user wants to save
            save = input("\nSave to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"timing_{result_id}.json"
                save_timing_to_file(timing_data, filename)
        else:
            sys.exit(1)
    else:
        print("\nFetching all timing data...")
        all_timings = get_all_timings()
        
        if not all_timings:
            print("❌ No timing data available.")
            print("   Run some operations (load data, apply filter, detect peaks) first.")
            sys.exit(1)
        
        print(f"\nFound {len(all_timings)} timing report(s):\n")
        
        for result_id, timing_data in all_timings.items():
            print(format_timing_report(timing_data, result_id))
        
        # Ask if user wants to save all
        save = input("\nSave all to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = "all_timings.json"
            save_timing_to_file(all_timings, filename)


if __name__ == "__main__":
    main()





