# Peak Analysis Tool Tests

This directory contains the quick, automated tests for the current source-run application.
The goal is to protect the current web workflow rather than preserve old
interactive validation scripts.

## What runs in local `pytest`
- `tests/unit/` contains deterministic unit tests for core numerical and backend
  behavior: loading, time scaling, dead-time correction, decimation, peak
  parameter validation, peak width detection, double-peak classification,
  export tables, and in-memory result storage. Run them with:
  ```bash
  pytest tests/unit
  ```

## Manual API smoke
With the backend already running, use the manual API smoke script when you want a quick endpoint check:
```bash
python tools/test_web_ui.py
```

Despite the historical file name, this is not a browser UI test. It only checks
selected backend API endpoints on `http://127.0.0.1:8765`.

## Legacy test classification
- Keep: `tests/unit/test_data_utils.py`, `tests/unit/test_photon_correction.py`,
  `tests/unit/test_service_functions.py`, and `tests/unit/test_storage.py`.
  These are fast and protect current behavior.
- Rewrite as deterministic unit tests: the old peak-width and time-resolution
  scripts. Their core ideas are covered by small synthetic-signal tests instead
  of random, plot-heavy scripts.
- Delete: generated legacy fixtures such as `test_peak_data.txt` and
  `test_time_res_*.txt`. They were large artifacts from manual scripts, not
  stable source fixtures.
- Treat as manual smoke only: `tools/test_web_ui.py`. Do not include it in the
  default pytest run unless it is converted to a proper integration test with
  explicit server setup.

## Adding new tests
- Prefer small, deterministic unit tests under `tests/unit/`.
- Use generated temporary fixtures for small files instead of committing large
  generated data files.
- If a test requires a live server, document it as an integration or smoke check outside the default unit run.
