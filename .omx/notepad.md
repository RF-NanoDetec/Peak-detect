

## WORKING MEMORY
[2026-04-02T19:02:27.079Z] Cleanup/perf plan: (1) switch DetectionChart full-res fetch from JSON to binary endpoint + typed-array parser, (2) enable chunked local preview parsing so file previews do not read whole files into memory, (3) remove pandas/DataFrame creation from load_data_from_paths hot path while preserving preview_head/time_range API outputs, then verify with lint/build/targeted tests.
