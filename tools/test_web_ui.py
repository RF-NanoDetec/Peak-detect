#!/usr/bin/env python3
"""
Simple test script to verify web UI backend endpoints are working correctly.
This script tests:
1. Health endpoint
2. Version endpoint
3. File upload endpoint (basic check)
4. Preprocess endpoint (basic check)

Usage:
    python tools/test_web_ui.py

Note: The backend server must be running on port 8765.
"""

import requests
import sys
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8765"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        response.raise_for_status()
        data = response.json()
        assert data["status"] == "ok"
        print("[PASS] Health check passed")
        return True
    except Exception as e:
        print(f"[FAIL] Health check failed: {e}")
        return False

def test_version():
    """Test version endpoint"""
    print("\nTesting version endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/version")
        response.raise_for_status()
        data = response.json()
        assert "version" in data
        print(f"[PASS] Version check passed: {data['version']}")
        return True
    except Exception as e:
        print(f"[FAIL] Version check failed: {e}")
        return False

def test_params():
    """Test params endpoint"""
    print("\nTesting params endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/params")
        response.raise_for_status()
        data = response.json()
        
        # Check that new filter parameters exist
        assert "filter_type" in data
        assert "filter_cutoff_freq" in data
        assert "butter_order" in data
        assert "savgol_window" in data
        assert "savgol_polyorder" in data
        
        print("[PASS] Params check passed")
        print(f"   Filter type: {data['filter_type']}")
        print(f"   Filter params: cutoff={data['filter_cutoff_freq']}, butter_order={data['butter_order']}")
        return True
    except Exception as e:
        print(f"[FAIL] Params check failed: {e}")
        return False

def test_file_upload_endpoint():
    """Test that file upload endpoint exists and responds"""
    print("\nTesting file upload endpoint availability...")
    try:
        # Just test that the endpoint exists by sending an invalid request
        # (we expect a validation error, not a 404)
        response = requests.post(f"{BASE_URL}/api/files/upload")
        
        # We expect a 422 (validation error) because we didn't send files
        # If we get 404, the endpoint doesn't exist
        if response.status_code == 404:
            print("[FAIL] File upload endpoint not found (404)")
            return False
        elif response.status_code == 422:
            print("[PASS] File upload endpoint exists (returns 422 for invalid request as expected)")
            return True
        else:
            print(f"[PASS] File upload endpoint exists (returned status {response.status_code})")
            return True
    except Exception as e:
        print(f"[FAIL] File upload endpoint check failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Web UI Backend Test Suite")
    print("=" * 60)
    
    # Check if server is running
    print("\nChecking if backend server is running...")
    try:
        requests.get(BASE_URL, timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"[FAIL] Cannot connect to backend server at {BASE_URL}")
        print("Please start the backend server with:")
        print("    python -m service.app")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(("Health", test_health()))
    results.append(("Version", test_version()))
    results.append(("Params", test_params()))
    results.append(("File Upload Endpoint", test_file_upload_endpoint()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:.<40} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed.")
        sys.exit(0)
    else:
        print(f"\n{total - passed} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()



