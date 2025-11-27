# Peak Analysis Tool Tests

This directory now separates quick, automated unit tests from legacy, plot-heavy validation scripts.

## What runs in CI/local `pytest`
- `tests/unit/` contains lightweight unit tests (e.g., dead-time correction, decimation). Run them with:
  ```bash
  pytest tests/unit
  ```

## Legacy/manual scripts (skipped in pytest)
The following files remain for manual validation and visual inspection and are skipped by default:
- `tests/run_peak_width_test.py`
- `tests/run_time_resolution_test.py`
- `tests/test_peak_width.py`
- `tests/test_peak_widths.py`
- `tests/test_time_resolution.py`

Run them explicitly if you need the plots or data artifacts:
```bash
python tests/run_peak_width_test.py
python tests/run_time_resolution_test.py
```

These scripts generate data files and images (e.g., `test_peak_data.txt`, `peak_width_test_results.png`) for inspection but are intentionally excluded from automated test runs to keep CI fast and deterministic.

## Adding new tests
- Prefer small, deterministic unit tests under `tests/unit/`.
- If a test requires plotting or large data generation, mark it as manual (skip in pytest) and document how to run it. 
