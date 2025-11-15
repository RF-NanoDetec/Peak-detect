"""
Quick test script to verify the timing system is working.

This script:
1. Tests that the server is running
2. Creates a simple test to generate timing data
3. Fetches and displays the timing data
"""

import sys
import requests
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:8765/api"


def check_server():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running!")
        print("   Please start it with: python -m service.app")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False


def create_test_file():
    """Create a simple test data file."""
    test_file = Path("test_timing_data.txt")
    
    # Create a simple test file with 1000 data points
    with open(test_file, 'w') as f:
        f.write("Time - Plot 0\tAmplitude - Plot 0\n")
        for i in range(1000):
            t = i * 0.0001  # 0.1 ms resolution
            amp = 100 + 50 * (i % 100) / 100  # Simple pattern
            f.write(f"{t}\t{amp}\n")
    
    print(f"✅ Created test file: {test_file}")
    return str(test_file.absolute())


def test_load_data(file_path: str):
    """Test loading data and get timing."""
    print("\n📊 Testing data loading...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/files/open",
            json={
                "paths": [file_path],
                "mode": "single",
                "time_resolution": 1e-4,
                "apply_dead_time_correction": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            result_id = result["resultId"]
            print(f"✅ Data loaded successfully")
            print(f"   Result ID: {result_id}")
            print(f"   Total points: {result['meta']['total_points']}")
            
            # Wait a moment for timing to be saved
            time.sleep(0.5)
            
            # Get timing data
            timing_response = requests.get(
                f"{BASE_URL}/performance/timing/{result_id}",
                timeout=5
            )
            
            if timing_response.status_code == 200:
                timing_data = timing_response.json()
                print(f"\n⏱️  Timing Data Retrieved:")
                print(f"   Operation: {timing_data['operation']}")
                print(f"   Total time: {timing_data['total_time_seconds']:.3f}s")
                
                slowest = timing_data.get('slowest_phase', {})
                if slowest.get('name'):
                    print(f"   Slowest phase: {slowest['name']} ({slowest['time_seconds']:.3f}s, {slowest['percentage']:.1f}%)")
                
                return result_id, timing_data
            else:
                print(f"⚠️  Could not retrieve timing data (status: {timing_response.status_code})")
                return result_id, None
        else:
            print(f"❌ Failed to load data: {response.status_code}")
            print(f"   {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def display_timing_summary(timing_data):
    """Display a summary of timing data."""
    if not timing_data:
        return
    
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"Operation: {timing_data['operation']}")
    print(f"Total Time: {timing_data['total_time_seconds']:.3f}s")
    print("\nTop 5 Phases:")
    
    phases = timing_data.get('phases', {})
    sorted_phases = sorted(
        phases.items(),
        key=lambda x: x[1].get('total_seconds', 0),
        reverse=True
    )[:5]
    
    for phase_name, phase_data in sorted_phases:
        time_sec = phase_data.get('total_seconds', 0)
        percentage = phase_data.get('percentage', 0)
        print(f"  {phase_name:<40} {time_sec:>8.3f}s ({percentage:>5.1f}%)")
    
    print("="*70)


def main():
    """Main test function."""
    print("Performance Timing System Test")
    print("="*70)
    
    # Check server
    if not check_server():
        sys.exit(1)
    
    # Create test file
    test_file = create_test_file()
    
    try:
        # Test loading data
        result_id, timing_data = test_load_data(test_file)
        
        if result_id:
            print(f"\n✅ Test completed successfully!")
            
            if timing_data:
                display_timing_summary(timing_data)
                print("\n💡 To view full timing data, run:")
                print(f"   python tools/get_timing_data.py {result_id}")
            else:
                print("\n⚠️  Timing data not available yet (may need to wait a moment)")
        else:
            print("\n❌ Test failed")
            sys.exit(1)
            
    finally:
        # Clean up test file
        test_file_path = Path(test_file)
        if test_file_path.exists():
            try:
                test_file_path.unlink()
                print(f"\n🧹 Cleaned up test file")
            except:
                pass


if __name__ == "__main__":
    main()


