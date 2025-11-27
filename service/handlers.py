from __future__ import annotations

from typing import Dict, Any, List
import tempfile
import os
import struct

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import StreamingResponse

from .models import (
    LoadFilesRequest,
    LoadFilesResponse,
    FileMeta,
    Params,
    ProcessRunRequest,
    DetectPeaksRequest,
    TaskStatusResponse,
)
from .storage import InMemoryStore
from .jobs import TaskManager, TaskStatus
from core.service_functions import (
    load_data_from_paths,
    analyze_double_peaks_pure,
    export_peaks_to_csv_data,
    export_double_peaks_to_csv_data,
    export_unified_peaks_data,
)
from core.data_analysis import analyze_time_resolved_pure
from core.data_utils import decimate_for_plot
from core.data_utils import validate_peak_params
from core.peak_analysis_utils import estimate_peak_widths
from core.peak_detection import calculate_auto_threshold
from scipy.signal import find_peaks
from .plotting import generate_peak_regions_plot  # Only used for peak inspection now
from core.peak_analysis_utils import find_peaks_with_window
from core.timing import TimingTracker, create_tracker, get_tracker
from . import cache as cache_module
from io import StringIO, BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import logging

router = APIRouter(prefix="/api")


def get_store(request: Request) -> InMemoryStore:
    store = getattr(request.app.state, "store", None)
    if not store:
        raise HTTPException(status_code=500, detail="Storage not initialized")
    return store


def get_task_manager(request: Request) -> TaskManager:
    tm = getattr(request.app.state, "task_manager", None)
    if not tm:
        raise HTTPException(status_code=500, detail="Task manager not initialized")
    return tm


@router.post("/files/open", response_model=LoadFilesResponse)
def open_files(payload: LoadFilesRequest, store: InMemoryStore = Depends(get_store)) -> LoadFilesResponse:
    # Create timing tracker for data loading
    tracker = create_tracker("data_loading")
    tracker.start()
    logger = logging.getLogger(__name__)
    
    # Validate paths exist
    import os
    with tracker.phase("file_validation"):
        missing_files = []
        for path in payload.paths:
            if not os.path.isabs(path):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Path '{path}' is not an absolute path. Please provide full path (e.g., C:\\Users\\...\\file.txt) or use the file picker."
                )
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            raise HTTPException(
                status_code=404,
                detail=f"File(s) not found: {', '.join(missing_files)}. Please check the path and try again."
            )
    
    # Convert protocol to dict if present
    protocol_dict = payload.protocol.dict() if payload.protocol else None
    
    # Load data with detailed timing
    with tracker.phase("load_data_from_paths"):
        data = load_data_from_paths(
            paths=payload.paths,
            mode=payload.mode,
            timestamps=payload.timestamps,
            time_resolution=payload.time_resolution,
            apply_dead_time_correction=payload.apply_dead_time_correction,
            dead_time_ns=payload.dead_time_ns,
            protocol=protocol_dict,
        )
    
    tracker.add_metadata("file_count", len(payload.paths))
    tracker.add_metadata("total_points", len(data["time"]))
    tracker.add_metadata("time_resolution", payload.time_resolution)
    
    with tracker.phase("save_result"):
        result_id = store.save_result(
            {
                "time": data["time"],
                "amplitude": data["amplitude"],
                "meta": {
                    "loaded_files": data["loaded_files"],
                    "time_resolution": payload.time_resolution,
                    "protocol": data.get("protocol"),
                    "corrections": data.get("corrections"),
                },
                "frame_preview": data["preview_head"],
            }
        )
    
    tracker.stop()
    
    # Store timing data
    timing_summary = tracker.get_summary()
    store.save_timing(result_id, timing_summary)
    logger.debug(f"Saved timing data for result_id: {result_id}")
    
    # Log summary
    logger.info(f"Data loading completed in {tracker.get_total_time():.3f}s")
    tracker.log_summary(logger)
    return LoadFilesResponse(
        resultId=result_id,
        meta=FileMeta(
            total_files=data["count"],
            total_points=len(data["time"]),
            time_range=[float(data["time_range"][0]), float(data["time_range"][1])],
            files=data["loaded_files"],
            protocol=payload.protocol,
            corrections=data.get("corrections"),
        ),
        preview=data["preview_head"],
    )


@router.post("/files/upload", response_model=LoadFilesResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    time_resolution: float = Form(1e-4),
    apply_dead_time_correction: bool = Form(False),
    dead_time_ns: float = Form(43.0),
    protocol_json: str = Form(""),
    store: InMemoryStore = Depends(get_store)
) -> LoadFilesResponse:
    """Upload files from browser and process them"""
    import json
    import logging
    logger = logging.getLogger(__name__)
    
    # Create timing tracker for data loading
    tracker = create_tracker("data_loading")
    tracker.start()
    
    temp_dir = tempfile.mkdtemp()
    temp_paths = []
    
    try:
        # Save uploaded files to temporary directory
        with tracker.phase("save_uploaded_files"):
            for file in files:
                temp_path = os.path.join(temp_dir, file.filename or "upload.txt")
                with open(temp_path, 'wb') as f:
                    # Read in chunks to avoid loading entire file into memory at once
                    while content := await file.read(1024 * 1024):  # 1MB chunks
                        f.write(content)
                temp_paths.append(temp_path)
        
        # Parse protocol JSON if provided
        protocol_dict = None
        if protocol_json and protocol_json.strip():
            try:
                protocol_dict = json.loads(protocol_json)
            except json.JSONDecodeError:
                pass
        
        # Process files using existing logic
        with tracker.phase("load_data_from_paths"):
            data = load_data_from_paths(
                paths=temp_paths,
                mode='batch' if len(temp_paths) > 1 else 'single',
                timestamps=None,
                time_resolution=time_resolution,
                apply_dead_time_correction=apply_dead_time_correction,
                dead_time_ns=dead_time_ns,
                protocol=protocol_dict,
            )
        
        tracker.add_metadata("file_count", len(files))
        tracker.add_metadata("total_points", len(data["time"]))
        tracker.add_metadata("time_resolution", time_resolution)
        
        with tracker.phase("save_result"):
            result_id = store.save_result(
                {
                    "time": data["time"],
                    "amplitude": data["amplitude"],
                    "meta": {
                        "loaded_files": [f.filename or "upload.txt" for f in files],
                    "time_resolution": time_resolution,
                    "protocol": data.get("protocol"),
                    "corrections": data.get("corrections"),
                },
                "frame_preview": data["preview_head"],
            }
        )
        
        tracker.stop()
        
        # Store timing data
        timing_summary = tracker.get_summary()
        store.save_timing(result_id, timing_summary)
        logger.debug(f"Saved timing data for result_id: {result_id}")
        
        # Log summary
        logger.info(f"Data loading (upload) completed in {tracker.get_total_time():.3f}s")
        tracker.log_summary(logger)
        
        # Convert protocol dict back to ProtocolInfo for response
        from .models import ProtocolInfo
        protocol_info = ProtocolInfo(**protocol_dict) if protocol_dict else None
        
        return LoadFilesResponse(
            resultId=result_id,
            meta=FileMeta(
                total_files=data["count"],
                total_points=len(data["time"]),
                time_range=[float(data["time_range"][0]), float(data["time_range"][1])],
                files=[f.filename or "upload.txt" for f in files],
                protocol=protocol_info,
                corrections=data.get("corrections"),
            ),
            preview=data["preview_head"],
        )
    finally:
        # Cleanup temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


@router.get("/results/{result_id}")
def get_result(result_id: str, store: InMemoryStore = Depends(get_store)) -> Dict[str, Any]:
    result = store.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    # Return time and amplitude arrays as lists for JSON serialization
    return {
        "time": result["time"].tolist(),
        "amplitude": result["amplitude"].tolist(),
        "meta": result.get("meta") or {},
    }

@router.get("/results/{result_id}/binary")
def get_result_binary(result_id: str, store: InMemoryStore = Depends(get_store)) -> StreamingResponse:
    result = store.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    time_arr = np.ascontiguousarray(result["time"], dtype=np.float64)
    amplitude_arr = np.ascontiguousarray(result["amplitude"], dtype=np.float32)
    count = int(len(time_arr))

    header = struct.pack("<4sB3xI4x", b"PABD", 2, count)

    def iter_chunks():
        yield header
        yield time_arr.tobytes(order="C")
        yield amplitude_arr.tobytes(order="C")

    headers = {
        "X-Result-Points": str(count),
        "X-Result-Time-Resolution": str(result.get("meta", {}).get("time_resolution", "")),
    }

    return StreamingResponse(iter_chunks(), media_type="application/octet-stream", headers=headers)

@router.get("/data/preview")
def get_data_preview(
    resultId: str,
    filteredResultId: str = None,
    limit: int = 10000,
    full: bool = False,
    store: InMemoryStore = Depends(get_store),
) -> Dict[str, Any]:
    """
    Return a decimated preview of the data for quick plotting.
    If filteredResultId is provided, returns BOTH original and filtered with SYNCHRONIZED decimation.
    This ensures the data points align perfectly for comparison.
    """
    result = store.get_result(resultId)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    time_arr = result["time"]
    amp_arr = result["amplitude"]

    # If full requested, return the entire arrays; otherwise decimate for plotting
    if full:
        time_dec, amp_dec = time_arr, amp_arr
    else:
        # Decimate for plotting using ORIGINAL data to determine indices
        time_dec, amp_dec = decimate_for_plot(time_arr, amp_arr, max_points=limit)

    total_points = int(len(time_arr))
    decimated = False if full else (len(time_dec) < total_points)

    response = {
        "time": time_dec.tolist(),
        "amplitude": amp_dec.tolist(),
        "decimated": decimated,
        "total_points": total_points,
    }

    # If filtered data requested, get it using THE SAME time indices (or full array)
    if filteredResultId:
        filtered_result = store.get_result(filteredResultId)
        if filtered_result:
            filtered_amp = filtered_result["amplitude"]
            if full:
                # Return the full filtered array (assumed aligned to original)
                response["filtered_amplitude"] = filtered_amp.tolist()
            else:
                # Find indices efficiently using searchsorted (O(m log n) instead of O(m*n))
                # Since time arrays are monotonic, we can use binary search
                indices = np.searchsorted(time_arr, time_dec)
                # Clamp indices to valid range
                indices = np.clip(indices, 0, len(filtered_amp) - 1)
                
                # Get filtered amplitudes at those SAME indices
                filtered_amp_dec = filtered_amp[indices]
                response["filtered_amplitude"] = filtered_amp_dec.tolist()

    return response

# Removed /data/plot-image and /data/plot-detection-image endpoints
# All charts now use uPlot for interactive rendering on the frontend

@router.post("/preprocess/run")
def preprocess_run(
    payload: ProcessRunRequest,
    request: Request,
    store: InMemoryStore = Depends(get_store),
    tm: TaskManager = Depends(get_task_manager),
) -> Dict[str, str]:
    """
    Apply preprocessing filters asynchronously.
    Returns a taskId for progress tracking.
    """
    base = store.get_result(payload.resultId)
    if not base:
        raise HTTPException(status_code=404, detail="Base result not found")

    def work(progress_callback, *, base_result, params: Params, original_result_id: str):
        import logging
        from scipy import signal as scipy_signal
        logger = logging.getLogger(__name__)
        
        # Create timing tracker for filtering
        tracker = create_tracker("filtering")
        tracker.start()
        
        time_data = base_result["time"]
        amplitude_data = base_result["amplitude"]
        
        # Log array sizes to confirm full-data processing
        logger.info(f"PREPROCESSING: Starting with {len(time_data)} data points (FULL ARRAY)")
        
        progress_callback({"progress": 1, "status": "running", "message": "Starting preprocessing"})
        progress_callback({"progress": 10, "status": "running", "message": "Loading data"})
        
        # Apply filter based on type
        with tracker.phase("copy_data"):
            filtered_amplitude = amplitude_data.copy()
        
        filter_type = getattr(params, 'filter_type', 'none')
        logger.info(f"PREPROCESSING: Filter type = {filter_type}")
        
        tracker.add_metadata("filter_type", filter_type)
        tracker.add_metadata("data_points", len(amplitude_data))
        
        progress_callback({"progress": 30, "status": "running", "message": f"Applying {filter_type} filter"})
        
        # For 'none' filter, just pass through quickly
        if filter_type == 'none':
            logger.info("PREPROCESSING: No filter selected, passing through data")
            progress_callback({"progress": 50, "status": "running", "message": "No filter applied"})
        elif filter_type == 'butterworth':
            # Apply Butterworth filter
            with tracker.phase("butterworth_filter"):
                cutoff_freq = getattr(params, 'filter_cutoff_freq', 1000.0)
                butter_order = getattr(params, 'butter_order', 4)
                
                # Calculate sampling rate from time resolution
                time_res = base_result["meta"].get("time_resolution", 1e-4)
                fs = 1.0 / time_res
                nyquist = fs / 2.0
                
                # Normalize cutoff frequency
                normalized_cutoff = cutoff_freq / nyquist
                if normalized_cutoff >= 1.0:
                    normalized_cutoff = 0.99
                
                with tracker.phase("butterworth_design"):
                    b, a = scipy_signal.butter(butter_order, normalized_cutoff, btype='low')
                
                with tracker.phase("butterworth_apply"):
                    filtered_amplitude = scipy_signal.filtfilt(b, a, amplitude_data)
                
                tracker.add_metadata("cutoff_freq", cutoff_freq)
                tracker.add_metadata("butter_order", butter_order)
            
        elif filter_type == 'savgol':
            # Apply Savitzky-Golay filter
            with tracker.phase("savgol_filter"):
                window_length = getattr(params, 'savgol_window', 51)
                polyorder = getattr(params, 'savgol_polyorder', 3)
                
                # Ensure window length is odd and valid
                if window_length % 2 == 0:
                    window_length += 1
                if window_length > len(amplitude_data):
                    window_length = len(amplitude_data) if len(amplitude_data) % 2 == 1 else len(amplitude_data) - 1
                if window_length < polyorder + 2:
                    window_length = polyorder + 2
                    if window_length % 2 == 0:
                        window_length += 1
                
                filtered_amplitude = scipy_signal.savgol_filter(amplitude_data, window_length, polyorder)
                
                tracker.add_metadata("window_length", window_length)
                tracker.add_metadata("polyorder", polyorder)
        
        progress_callback({"progress": 70, "status": "running", "message": "Saving filtered data"})
        
        # Save the filtered result
        with tracker.phase("save_result"):
            result_id = store.save_result(
                {
                    "time": time_data,
                    "amplitude": filtered_amplitude,
                    "meta": base_result.get("meta") or {},
                    "original_result_id": original_result_id,
                }
            )
        
        progress_callback({"progress": 90, "status": "running", "message": "Finalizing"})
        
        tracker.stop()
        
        # Store timing data
        timing_summary = tracker.get_summary()
        store.save_timing(result_id, timing_summary)
        
        # Log summary
        logger.info(f"Filtering completed in {tracker.get_total_time():.3f}s")
        tracker.log_summary(logger)
        
        return {"resultId": result_id, "filtered": filter_type != 'none'}

    def on_progress(task_id: str, update: dict):
        # Broadcast via app helper
        try:
            request.app.broadcast_progress(task_id, update)  # type: ignore[attr-defined]
        except Exception:
            pass

    task_id = tm.submit(work, on_progress=on_progress, base_result=base, params=payload.params, original_result_id=payload.resultId)
    return {"taskId": task_id}

@router.post("/detect/run")
def detect_run(payload: DetectPeaksRequest, store: InMemoryStore = Depends(get_store)) -> Dict[str, Any]:
    import logging
    logger = logging.getLogger(__name__)
    
    # Create timing tracker for peak detection
    tracker = create_tracker("peak_detection")
    tracker.start()
    
    with tracker.phase("load_data"):
        base = store.get_result(payload.resultId)
        if not base:
            raise HTTPException(status_code=404, detail="Base result not found")
        
        # If filteredResultId provided, prefer filtered signal for detection
        filtered_result = None
        filtered_result_id = getattr(payload, "filteredResultId", None)
        logger.info(f"PEAK DETECTION: filteredResultId = {filtered_result_id}")
        
        if filtered_result_id:
            filtered_result = store.get_result(filtered_result_id)
            if filtered_result:
                logger.info(f"PEAK DETECTION: Using FILTERED data ({len(filtered_result['amplitude'])} points)")
            else:
                logger.warning(f"PEAK DETECTION: filteredResultId provided but result not found")
        else:
            logger.info(f"PEAK DETECTION: Using RAW data")
    
    # Log array size to confirm full-data processing
    logger.info(f"PEAK DETECTION: Starting with {len(base['time'])} data points (FULL ARRAY)")
    logger.info(f"PEAK DETECTION: Parameters - threshold={payload.prominence_threshold}, distance={payload.distance}, rel_height={payload.rel_height}, width_ms={payload.width_ms}, prominence_ratio={payload.prominence_ratio}")
    
    with tracker.phase("validate_params"):
        ok, msg = validate_peak_params(
            payload.prominence_threshold,
            payload.distance,
            payload.rel_height,
            payload.width_ms,
            payload.time_resolution,
            payload.prominence_ratio,
        )
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
    
    signal = filtered_result["amplitude"] if filtered_result else base["amplitude"]
    time_values = base["time"]
    
    tracker.add_metadata("data_points", len(signal))
    tracker.add_metadata("using_filtered", filtered_result is not None)
    tracker.add_metadata("prominence_threshold", payload.prominence_threshold)
    tracker.add_metadata("distance", payload.distance)
    
    # Log signal stats to verify we're using the right data
    logger.info(f"PEAK DETECTION: Signal min={float(np.min(signal)):.2f}, max={float(np.max(signal)):.2f}, mean={float(np.mean(signal)):.2f}")
    
    # IMPORTANT: Verify threshold is appropriate for the signal being used
    # If using filtered data, the threshold should ideally be calculated from filtered data
    # Log a warning if threshold seems inappropriate (this is informational only)
    if filtered_result:
        # Calculate what auto threshold would be for filtered data
        with tracker.phase("auto_threshold_calc"):
            from core.peak_detection import calculate_auto_threshold
            auto_threshold_filtered = calculate_auto_threshold(signal, sigma_multiplier=5.0)
            threshold_ratio = payload.prominence_threshold / auto_threshold_filtered if auto_threshold_filtered > 0 else 1.0
            logger.info(f"PEAK DETECTION: Using FILTERED data for detection")
            logger.info(f"PEAK DETECTION: Auto threshold for filtered data: {auto_threshold_filtered:.2f}, provided threshold: {payload.prominence_threshold:.2f} (ratio: {threshold_ratio:.2f}x)")
            # Warn if threshold is significantly different from what would be calculated from filtered data
            if abs(threshold_ratio - 1.0) > 0.5:  # More than 50% difference
                logger.warning(f"PEAK DETECTION: WARNING - Threshold ({payload.prominence_threshold:.2f}) differs significantly from auto threshold for filtered data ({auto_threshold_filtered:.2f}). "
                              f"Consider recalculating threshold from filtered data using 'Auto Threshold' button.")
    else:
        logger.info(f"PEAK DETECTION: Using ORIGINAL data for detection")
    
    with tracker.phase("analyze_time_resolved"):
        result = analyze_time_resolved_pure(
            signal=signal,
            time_values=time_values,
            width_ms=payload.width_ms,
            time_resolution=payload.time_resolution,
            prominence_threshold=payload.prominence_threshold,
            distance=payload.distance,
            rel_height=payload.rel_height,
            prominence_ratio=payload.prominence_ratio,
        )
    
    if result is None:
        tracker.stop()
        tracker.add_metadata("peaks_found", 0)
        timing_summary = tracker.get_summary()
        store.save_timing(f"detect_{payload.resultId}", timing_summary)
        logger.info(f"Peak detection completed in {tracker.get_total_time():.3f}s (no peaks found)")
        tracker.log_summary(logger)
        return {"count": 0, "peaks": [], "stats": {"mean_height": 0.0, "mean_width": 0.0}}
    
    peaks, areas, intervals, widths_samples = result
    peaks_list = [int(i) for i in peaks]
    
    tracker.add_metadata("peaks_found", len(peaks_list))
    
    # Calculate stats and extract peak details
    # IMPORTANT: Use the same signal (filtered or original) that was used for detection
    with tracker.phase("calculate_stats"):
        if len(peaks_list) > 0:
            peak_heights = [float(signal[i]) for i in peaks_list]
            peak_times = [float(base["time"][i]) for i in peaks_list]  # Get actual time values
            mean_height = float(np.mean(peak_heights))
            # Convert widths from samples to milliseconds: widths_samples * time_resolution * 1000
            widths_ms = widths_samples * payload.time_resolution * 1000
            mean_width = float(np.mean(widths_ms)) if len(widths_ms) > 0 else 0.0
            logger.info(f"PEAK DETECTION: Found {len(peaks_list)} peaks, mean height={mean_height:.2f}, peak heights range=[{min(peak_heights):.2f}, {max(peak_heights):.2f}]")
        else:
            mean_height = 0.0
            mean_width = 0.0
            peak_times = []
            peak_heights = []
            widths_ms = []
            logger.info("PEAK DETECTION: No peaks found")
    
    # Compute detailed width properties for drawing (left/right bounds and width height)
    # Reuse the same parameters to get properties arrays
    with tracker.phase("compute_properties"):
        from core.peak_analysis_utils import find_peaks_with_window
        # Normalize width_ms to sample tuple
        if isinstance(payload.width_ms, str):
            width_values = payload.width_ms.strip().split(',')
        else:
            width_values = [str(v) for v in payload.width_ms]
        sampling_rate = 1.0 / payload.time_resolution
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        peaks2, properties = find_peaks_with_window(
            signal,
            width=width_p,
            prominence=payload.prominence_threshold,
            distance=payload.distance,
            rel_height=payload.rel_height,
            prominence_ratio=payload.prominence_ratio,
        )
        # Extract arrays (may be empty)
        prop_prominences = properties.get("prominences", np.array([]))
        prop_widths = properties.get("widths", np.array([]))
        prop_left_ips = properties.get("left_ips", np.array([]))
        prop_right_ips = properties.get("right_ips", np.array([]))
        prop_width_heights = properties.get("width_heights", np.array([]))

    tracker.stop()
    
    # Store timing data
    timing_summary = tracker.get_summary()
    store.save_timing(f"detect_{payload.resultId}", timing_summary)
    
    # Log summary
    logger.info(f"Peak detection completed in {tracker.get_total_time():.3f}s")
    tracker.log_summary(logger)

    # Convert intervals from seconds to milliseconds for consistency
    intervals_ms = [float(i * 1000) for i in intervals] if len(intervals) > 0 else []

    # Pre-calculate histograms to avoid round-trip
    from service.plotting import generate_peak_histogram_data
    histograms = generate_peak_histogram_data(
        peak_heights,
        widths_ms,
        intervals_ms,
        config={
            "metrics": {
                "amplitude": {"xScale": "log"},
                "width": {"xScale": "log"},
                "interval": {"xScale": "log"}
            },
            "bin_count": 60
        }
    )

    return {
        "count": len(peaks_list),
        "peaks": peaks_list,  # Indices (for reference)
        "peak_times": peak_times,  # Actual time values (for plotting)
        "peak_amplitudes": peak_heights,  # Actual amplitudes (for plotting)
        "peak_intervals": intervals_ms,  # Intervals between peaks in milliseconds
        "stats": {
            "mean_height": mean_height,
            "mean_width": mean_width,
            "std_height": float(np.std(peak_heights)) if len(peaks_list) > 0 else 0.0,
            "std_width": float(np.std(widths_ms)) if len(peaks_list) > 0 else 0.0,
        },
        "properties": {
            "prominences": prop_prominences.tolist(),
            "widths": prop_widths.tolist(),
            "left_ips": prop_left_ips.tolist(),
            "right_ips": prop_right_ips.tolist(),
            "width_heights": prop_width_heights.tolist(),
        },
        "histograms": histograms
    }


@router.post("/preprocess/auto-cutoff")
def auto_cutoff(payload: Dict[str, str], store: InMemoryStore = Depends(get_store)) -> Dict[str, float]:
    """
    Auto-calculate cutoff frequency based on signal characteristics (CACHED).
    Uses 70% of max signal as threshold to detect peaks and measure their widths.
    
    OPTIMIZATION: Results are cached per result_id for instant retrieval.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    result_id = payload.get('resultId')
    if not result_id:
        raise HTTPException(status_code=400, detail="resultId required")
    
    # Check cache first
    cached_cutoff = cache_module.get_auto_cutoff_result(result_id)
    if cached_cutoff is not None:
        logger.info(f"Auto-cutoff: using cached result={cached_cutoff} Hz")
        return {"cutoff_freq": int(round(cached_cutoff))}
    
    base = store.get_result(result_id)
    if not base:
        raise HTTPException(status_code=404, detail="Result not found")
    
    signal = base["amplitude"]
    time_resolution = base["meta"].get("time_resolution", 1e-4)
    
    # Calculate sampling rate
    fs = 1.0 / time_resolution
    logger.info(f"Auto-cutoff: fs={fs} Hz, time_res={time_resolution}")
    
    # Find max signal and use 70% as threshold
    signal_max = float(np.max(signal))
    threshold = signal_max * 0.7
    logger.info(f"Auto-cutoff: max={signal_max}, threshold (70%)={threshold}")
    
    # estimate_peak_widths now uses full signal and fastest 10% of peaks
    avg_width_sec = estimate_peak_widths(signal, fs, prominence_threshold=threshold, time_resolution=time_resolution)
    logger.info(f"Auto-cutoff: avg_width={avg_width_sec} seconds")
    
    # Calculate cutoff frequency: f_c = 1 / avg_width
    if avg_width_sec <= 1e-9:
        cutoff_hz = fs / 4.0  # Fallback
    else:
        cutoff_hz = 1.0 / avg_width_sec
    
    # Ensure within Nyquist limit
    nyquist_hz = fs / 2.0
    cutoff_hz = min(cutoff_hz, nyquist_hz * 0.95)
    cutoff_hz = max(cutoff_hz, 0.01)
    
    # Round to integer
    cutoff_hz_int = int(round(cutoff_hz))
    
    logger.info(f"Auto-cutoff: calculated cutoff={cutoff_hz} Hz, rounded to {cutoff_hz_int} Hz")
    
    # Cache the result (store as integer)
    cache_module.cache_auto_cutoff_result(result_id, float(cutoff_hz_int))
    
    return {"cutoff_freq": cutoff_hz_int}


@router.post("/detect/auto-threshold")
def auto_threshold(payload: Dict[str, Any], store: InMemoryStore = Depends(get_store)) -> Dict[str, float]:
    """
    Auto-calculate prominence threshold based on signal statistics.
    Uses sigma_multiplier * std_dev of the signal.
    
    IMPORTANT: Always uses filtered data if filteredResultId is provided,
    as the threshold should be calculated from the same data that will be used for peak detection.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    result_id = payload.get('resultId')
    filtered_result_id = payload.get('filteredResultId')
    sigma_multiplier = payload.get('sigma_multiplier', 5.0)
    
    if not result_id:
        raise HTTPException(status_code=400, detail="resultId required")
    
    base = store.get_result(result_id)
    if not base:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # CRITICAL: Always prefer filtered data if provided, as threshold must match the data used for detection
    signal = base["amplitude"]
    data_source = "original"
    if filtered_result_id:
        filtered_result = store.get_result(filtered_result_id)
        if filtered_result:
            signal = filtered_result["amplitude"]
            data_source = "filtered"
            logger.info(f"Auto-threshold: Using FILTERED data (filteredResultId={filtered_result_id}, {len(signal)} points)")
            logger.info(f"Auto-threshold: Filtered signal stats - min={float(np.min(signal)):.2f}, max={float(np.max(signal)):.2f}, mean={float(np.mean(signal)):.2f}")
        else:
            logger.warning(f"Auto-threshold: filteredResultId provided but result not found; using original data")
    else:
        logger.info(f"Auto-threshold: Using ORIGINAL data ({len(signal)} points)")
        logger.info(f"Auto-threshold: Original signal stats - min={float(np.min(signal)):.2f}, max={float(np.max(signal)):.2f}, mean={float(np.mean(signal)):.2f}")
    
    # Calculate threshold from the selected signal
    threshold = calculate_auto_threshold(signal, sigma_multiplier=sigma_multiplier)
    # Round to integer as requested
    threshold_int = int(round(threshold))
    logger.info(f"Auto-threshold: Calculated from {data_source} data - sigma={sigma_multiplier}, threshold={threshold:.2f}, rounded to integer: {threshold_int}")
    
    return {"prominence_threshold": float(threshold_int)}


@router.post("/detect/inspect-peaks")
def inspect_peaks(
    payload: Dict[str, Any],
    store: InMemoryStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Generate peak inspection plot showing individual peaks in 2x5 grid (CACHED).
    Requires both original and filtered data, plus detection parameters.
    
    OPTIMIZATION: Images are cached per (result_id, params, offset) for instant navigation.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Extract parameters
    result_id = payload.get('resultId')
    filtered_result_id = payload.get('filteredResultId')
    offset = payload.get('offset', 0)
    
    # Detection parameters
    prominence_threshold = payload.get('prominence_threshold', 20.0)
    distance = payload.get('distance', 30)
    rel_height = payload.get('rel_height', 0.8)
    width_ms = payload.get('width_ms', '0.1,200')
    prominence_ratio = payload.get('prominence_ratio', 0.8)
    time_resolution = payload.get('time_resolution', 1e-4)
    
    if not result_id:
        raise HTTPException(status_code=400, detail="resultId required")
    
    # Get filtered data (needed for checking if it exists)
    if not filtered_result_id:
        raise HTTPException(status_code=400, detail="filteredResultId required for peak inspection")
    
    # Check image cache first
    params_dict = {
        'prominence_threshold': prominence_threshold,
        'distance': distance,
        'rel_height': rel_height,
        'width_ms': width_ms,
        'prominence_ratio': prominence_ratio,
        'time_resolution': time_resolution
    }
    
    cached_image = cache_module.get_peak_inspection_image(result_id, filtered_result_id, offset, params_dict)
    if cached_image is not None:
        # Also need to get peak count for total_peaks (could cache this too, but it's fast)
        cached_detection = cache_module.get_peak_detection_result(filtered_result_id, params_dict)
        if cached_detection is not None:
            peaks, _ = cached_detection
            logger.info(f"Peak inspection: using cached image (offset={offset}, total_peaks={len(peaks)})")
            return {
                "image": cached_image,
                "format": "png",
                "encoding": "base64",
                "total_peaks": len(peaks),
                "showing_offset": offset
            }
    
    # Get original data
    base = store.get_result(result_id)
    if not base:
        raise HTTPException(status_code=404, detail="Result not found")
    
    filtered_result = store.get_result(filtered_result_id)
    if not filtered_result:
        raise HTTPException(status_code=404, detail="Filtered result not found")
    
    # Run peak detection to get peaks and properties
    time_values = base["time"]
    amplitude_raw = base["amplitude"]
    amplitude_filtered = filtered_result["amplitude"]
    
    # Check detection cache first
    cached_detection = cache_module.get_peak_detection_result(filtered_result_id, params_dict)
    
    if cached_detection is not None:
        peaks, properties = cached_detection
        logger.info(f"Peak inspection: using cached detection result ({len(peaks)} peaks)")
    else:
        # Convert width from ms to samples
        if isinstance(width_ms, str):
            width_values = width_ms.strip().split(',')
        else:
            width_values = [str(v) for v in width_ms]
        
        sampling_rate = 1.0 / time_resolution
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        
        # Detect peaks
        peaks, properties = find_peaks_with_window(
            amplitude_filtered,
            width=width_p,
            prominence=prominence_threshold,
            distance=distance,
            rel_height=rel_height,
            prominence_ratio=prominence_ratio,
        )
        
        # Cache the detection result
        cache_module.cache_peak_detection_result(filtered_result_id, params_dict, peaks, properties)
    
    if len(peaks) == 0:
        return {
            "image": None,
            "format": "png",
            "encoding": "base64",
            "total_peaks": 0,
            "message": "No peaks detected with current parameters"
        }
    
    logger.info(f"Peak inspection: {len(peaks)} peaks detected, offset={offset}")
    
    # Generate plot
    img_base64 = generate_peak_regions_plot(
        time_values,
        amplitude_raw,
        amplitude_filtered,
        peaks,
        properties,
        offset=offset
    )
    
    # Cache the image
    cache_module.cache_peak_inspection_image(result_id, filtered_result_id, offset, params_dict, img_base64)
    
    return {
        "image": img_base64,
        "format": "png",
        "encoding": "base64",
        "total_peaks": len(peaks),
        "showing_offset": offset
    }


@router.post("/detect/histograms")
def generate_histograms(
    payload: Dict[str, Any],
    store: InMemoryStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Calculate histogram data for peak statistics.
    
    Returns histogram bins and counts for uPlot rendering.
    Requires peak detection data (amplitudes, widths, intervals).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    peak_amplitudes = payload.get('peak_amplitudes', [])
    peak_widths_ms = payload.get('peak_widths_ms', [])
    peak_intervals_ms = payload.get('peak_intervals_ms', [])
    config = payload.get('config') or {}
    
    import time
    start_time = time.time()
    logger.info(f"Generating histograms for {len(peak_amplitudes)} peaks")
    
    if len(peak_amplitudes) == 0:
        return {
            "amplitude": {"bins": [], "counts": []},
            "width": {"bins": [], "counts": []},
            "interval": {"bins": [], "counts": []},
            "message": "No peak data provided"
        }
    
    try:
        from service.plotting import generate_peak_histogram_data
        hist_data = generate_peak_histogram_data(
            peak_amplitudes,
            peak_widths_ms,
            peak_intervals_ms,
            config=config
        )
        
        if hist_data is None:
            return {
                "amplitude": {"bins": [], "counts": []},
                "width": {"bins": [], "counts": []},
                "interval": {"bins": [], "counts": []}
            }
        
        elapsed = time.time() - start_time
        logger.info(f"Histogram generation completed in {elapsed:.4f}s")
        
        return hist_data
    except Exception as e:
        logger.error(f"Error calculating histogram data: {e}")
        raise HTTPException(status_code=500, detail=f"Error calculating histogram data: {str(e)}")


@router.post("/double/histograms")
def generate_double_peak_histograms(
    payload: Dict[str, Any],
    store: InMemoryStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Calculate histogram data for double peak pair metrics.
    
    Returns histogram bins and counts for uPlot rendering.
    Accepts distance_ms, pair_prom_ratio, pair_width_ratio, prom_over_amp arrays.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    distance_ms = payload.get('distance_ms', [])
    pair_prom_ratio = payload.get('pair_prom_ratio', [])
    pair_width_ratio = payload.get('pair_width_ratio', [])
    prom_over_amp = payload.get('prom_over_amp', [])
    config = payload.get('config') or {}
    
    if len(distance_ms) == 0 and len(prom_over_amp) == 0:
        return {
            "distance": {"bins": [], "counts": []},
            "pairPromRatio": {"bins": [], "counts": []},
            "pairWidthRatio": {"bins": [], "counts": []},
            "promOverAmp": {"bins": [], "counts": []},
            "message": "No pair data provided"
        }
    
    try:
        from service.plotting import calculate_histogram_bins
        
        metrics_config: Dict[str, Any] = config.get("metrics", {})
        range_overrides = config.get("range_overrides", {}) or {}
        bin_count = config.get("bin_count")
        
        def metric_hist(values, metric_key: str):
            try:
                numeric_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
                if len(numeric_values) == 0:
                    logger.debug(f"No valid numeric values for {metric_key} histogram")
                    return {"bins": [], "counts": []}
                
                metric_cfg = metrics_config.get(metric_key, {})
                x_scale = metric_cfg.get("xScale") or metric_cfg.get("x_scale") or "linear"
                range_override = range_overrides.get(metric_key) or metric_cfg.get("range") or metric_cfg.get("range_override")
                
                logger.debug(f"Calculating {metric_key} histogram: {len(numeric_values)} values, scale={x_scale}, bins={bin_count}")
                
                return calculate_histogram_bins(
                    numeric_values,
                    bin_count=bin_count,
                    range_override=range_override,
                    log_scale=x_scale == "log",
                )
            except Exception as e:
                logger.error(f"Error calculating {metric_key} histogram: {e}", exc_info=True)
                return {"bins": [], "counts": []}
        
        distance_hist = metric_hist(distance_ms, "distance")
        pair_prom_hist = metric_hist(pair_prom_ratio, "pairPromRatio")
        pair_width_hist = metric_hist(pair_width_ratio, "pairWidthRatio")
        prom_amp_hist = metric_hist(prom_over_amp, "promOverAmp")
        
        logger.info(f"Generated double peak histograms: distance={len(distance_hist['bins'])} bins, "
                   f"pairProm={len(pair_prom_hist['bins'])} bins, pairWidth={len(pair_width_hist['bins'])} bins, "
                   f"promOverAmp={len(prom_amp_hist['bins'])} bins")
        
        return {
            "distance": distance_hist,
            "pairPromRatio": pair_prom_hist,
            "pairWidthRatio": pair_width_hist,
            "promOverAmp": prom_amp_hist,
        }
        
    except Exception as e:
        logger.error(f"Error calculating double peak histogram data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calculating histogram data: {str(e)}")


@router.get("/cache/stats")
def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring performance improvements."""
    return cache_module.get_cache_stats()


@router.delete("/cache/clear")
def clear_caches() -> Dict[str, str]:
    """Clear all caches (useful for testing or memory management)."""
    cache_module.clear_all_caches()
    return {"message": "All caches cleared successfully"}


@router.get("/params", response_model=Params)
def get_params(request: Request) -> Params:
    return getattr(request.app.state, "params_default", Params())


@router.put("/params", response_model=Params)
def put_params(new_params: Params, request: Request) -> Params:
    request.app.state.params_default = new_params
    return new_params


@router.post("/process/run")
def process_run(
    payload: ProcessRunRequest,
    app=Depends(),
    store: InMemoryStore = Depends(get_store),
    tm: TaskManager = Depends(get_task_manager),
) -> Dict[str, str]:
    base = store.get_result(payload.resultId)
    if not base:
        raise HTTPException(status_code=404, detail="Base result not found")

    def work(progress_callback, *, base_result, params: Params):
        # Minimal example: run peak analysis on existing result
        progress_callback({"progress": 0.05, "message": "Starting analysis"})
        peaks_tuple = analyze_time_resolved_pure(
            signal=base_result["amplitude"],
            time_values=base_result["time"],
            width_ms=params.widthMs,
            time_resolution=params.timeResolution,
            prominence_threshold=params.prominenceThreshold,
            distance=params.distance,
            rel_height=params.relHeight,
            prominence_ratio=params.prominenceRatio,
        )
        progress_callback({"progress": 0.9, "message": "Finalizing"})
        return {"peaks": None if peaks_tuple is None else [int(i) for i in peaks_tuple[0]]}

    def on_progress(task_id: str, update: dict):
        # Broadcast via app helper
        try:
            app.broadcast_progress(task_id, update)  # type: ignore[attr-defined]
        except Exception:
            pass

    task_id = tm.submit(work, on_progress=on_progress, base_result=base, params=payload.params)
    return {"taskId": task_id}


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str, tm: TaskManager = Depends(get_task_manager)) -> TaskStatusResponse:
    info = tm.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(
        status=info.get("status") or TaskStatus.PENDING,
        progress=info.get("progress"),
        error=info.get("error"),
        result=info.get("result"),
    )


@router.post("/double-peak/analyze")
def analyze_double_peaks(
    payload: Dict[str, Any],
    store: InMemoryStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Analyze double peaks based on detected peaks and constraints.
    
    Expects:
    - resultId: ID of the original data
    - peaksResultId: ID of detection result with peaks info (or use detection directly)
    - Alternatively: peaks, properties arrays directly
    - min_distance, max_distance (seconds)
    - min_amp_ratio, max_amp_ratio
    - min_width_ratio, max_width_ratio
    - time_resolution
    """
    import logging
    logger = logging.getLogger(__name__)
    
    result_id = payload.get('resultId')
    time_resolution = payload.get('time_resolution', 1e-4)
    
    # Get constraints
    min_distance = payload.get('min_distance', 0.001)  # 1ms default
    max_distance = payload.get('max_distance', 0.100)  # 100ms default
    min_amp_ratio = payload.get('min_amp_ratio', 0.1)
    max_amp_ratio = payload.get('max_amp_ratio', 10.0)
    min_width_ratio = payload.get('min_width_ratio', 0.1)
    max_width_ratio = payload.get('max_width_ratio', 10.0)
    
    # Get peaks and properties from payload
    peaks = payload.get('peaks')
    properties = payload.get('properties')
    
    if peaks is None or properties is None:
        # Try to get from detection request
        prominence_threshold = payload.get('prominence_threshold', 20.0)
        distance = payload.get('distance', 30)
        rel_height = payload.get('rel_height', 0.8)
        width_ms = payload.get('width_ms', '0.1,200')
        prominence_ratio = payload.get('prominence_ratio', 0.8)
        
        # Get filtered result
        filtered_result_id = payload.get('filteredResultId')
        if not filtered_result_id:
            raise HTTPException(status_code=400, detail="filteredResultId required")
        
        filtered_result = store.get_result(filtered_result_id)
        if not filtered_result:
            raise HTTPException(status_code=404, detail="Filtered result not found")
        
        amplitude_filtered = filtered_result["amplitude"]
        
        # Convert width from ms to samples
        if isinstance(width_ms, str):
            width_values = width_ms.strip().split(',')
        else:
            width_values = [str(v) for v in width_ms]
        
        sampling_rate = 1.0 / time_resolution
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        
        # Detect peaks
        peaks, properties = find_peaks_with_window(
            amplitude_filtered,
            width=width_p,
            prominence=prominence_threshold,
            distance=distance,
            rel_height=rel_height,
            prominence_ratio=prominence_ratio,
        )
    else:
        # Convert lists to numpy arrays if needed
        peaks = np.array(peaks) if not isinstance(peaks, np.ndarray) else peaks
        for key in properties:
            if not isinstance(properties[key], np.ndarray):
                properties[key] = np.array(properties[key])
    
    # Run double peak analysis
    result = analyze_double_peaks_pure(
        peaks=peaks,
        properties=properties,
        time_resolution=time_resolution,
        min_distance=min_distance,
        max_distance=max_distance,
        min_amp_ratio=min_amp_ratio,
        max_amp_ratio=max_amp_ratio,
        min_width_ratio=min_width_ratio,
        max_width_ratio=max_width_ratio,
    )
    
    logger.info(f"Double peak analysis: {result['double_peak_count']} double peaks found out of {result['total_pairs']} pairs")
    
    return result


@router.get("/export/peaks/csv")
def export_peaks_csv(
    resultId: str,
    filteredResultId: str = None,
    prominence_threshold: float = 20.0,
    distance: int = 30,
    rel_height: float = 0.8,
    width_ms: str = '0.1,200',
    prominence_ratio: float = 0.8,
    time_resolution: float = 1e-4,
    store: InMemoryStore = Depends(get_store)
):
    """Export detected peaks to CSV format."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Get original data
    base = store.get_result(resultId)
    if not base:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Get filtered data if provided, otherwise use original
    if filteredResultId:
        filtered_result = store.get_result(filteredResultId)
        if not filtered_result:
            raise HTTPException(status_code=404, detail="Filtered result not found")
        amplitude = filtered_result["amplitude"]
    else:
        amplitude = base["amplitude"]
    
    time_values = base["time"]
    
    # Convert width from ms to samples
    if isinstance(width_ms, str):
        width_values = width_ms.strip().split(',')
    else:
        width_values = [str(v) for v in width_ms]
    
    sampling_rate = 1.0 / time_resolution
    width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
    
    # Detect peaks
    peaks, properties = find_peaks_with_window(
        amplitude,
        width=width_p,
        prominence=prominence_threshold,
        distance=distance,
        rel_height=rel_height,
        prominence_ratio=prominence_ratio,
    )
    
    # Create DataFrame
    df = export_peaks_to_csv_data(time_values, peaks, properties, time_resolution)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No peaks detected")
    
    # Convert to CSV
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=peaks.csv"}
    )


@router.get("/performance/timing/{result_id}")
def get_timing_report(
    result_id: str,
    store: InMemoryStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Get performance timing report for a specific result ID.
    
    Returns detailed timing breakdown for data loading, filtering, or peak detection.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Getting timing data for result_id: {result_id}")
    
    timing = store.get_timing(result_id)
    if not timing:
        all_timings = store.get_all_timings()
        logger.debug(f"Available timing IDs: {list(all_timings.keys())}")
        raise HTTPException(
            status_code=404,
            detail=f"Timing data not found for result ID: {result_id}. Available IDs: {list(all_timings.keys())[:5]}"
        )
    return timing


@router.get("/performance/timing")
def get_all_timing_reports(
    store: InMemoryStore = Depends(get_store)
) -> Dict[str, Dict[str, Any]]:
    """
    Get all performance timing reports.
    
    Returns a dictionary mapping result IDs to their timing summaries.
    """
    logger = logging.getLogger(__name__)
    all_timings = store.get_all_timings()
    logger.debug(f"Returning {len(all_timings)} timing reports: {list(all_timings.keys())}")
    return all_timings


@router.get("/performance/debug")
def debug_timing(
    store: InMemoryStore = Depends(get_store)
) -> Dict[str, Any]:
    """
    Debug endpoint to check timing storage.
    """
    all_timings = store.get_all_timings()
    return {
        "total_timings": len(all_timings),
        "timing_ids": list(all_timings.keys()),
        "sample": all_timings.get(list(all_timings.keys())[0]) if all_timings else None
    }


@router.post("/export/double-peaks/csv")
def export_double_peaks_csv(
    payload: Dict[str, Any],
    store: InMemoryStore = Depends(get_store)
):
    """Export double peak analysis to CSV format."""
    
    # Get result ID and time values
    result_id = payload.get('resultId')
    if not result_id:
        raise HTTPException(status_code=400, detail="resultId required")
    
    base = store.get_result(result_id)
    if not base:
        raise HTTPException(status_code=404, detail="Result not found")
    
    time_values = base["time"]
    
    # Get double peak analysis (should be passed in payload)
    double_peak_analysis = payload.get('double_peak_analysis')
    if not double_peak_analysis:
        raise HTTPException(status_code=400, detail="double_peak_analysis required")
    
    # Create DataFrame
    df = export_double_peaks_to_csv_data(double_peak_analysis, time_values)
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No double peaks to export")
    
    # Convert to CSV
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=double_peaks.csv"}
    )


@router.get("/export/plot/image")
def export_plot_image(
    resultId: str,
    filteredResultId: str = None,
    format: str = 'png',
    dpi: int = 300,
    store: InMemoryStore = Depends(get_store)
):
    """Export current plot as an image."""
    
    base = store.get_result(resultId)
    if not base:
        raise HTTPException(status_code=404, detail="Result not found")
    
    time_data = base["time"]
    amp_data = base["amplitude"]
    
    # Get filtered data if provided
    filtered_amp = None
    if filteredResultId:
        filtered_result = store.get_result(filteredResultId)
        if filtered_result:
            filtered_amp = filtered_result["amplitude"]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original
    ax.plot(time_data, amp_data, label='Original', alpha=0.7)
    
    # Plot filtered if available
    if filtered_amp is not None:
        ax.plot(time_data, filtered_amp, label='Filtered', linewidth=1.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal Data')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save to bytes buffer
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # Determine media type
    media_types = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'svg': 'image/svg+xml',
        'pdf': 'application/pdf'
    }
    media_type = media_types.get(format.lower(), 'application/octet-stream')
    
    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename=plot.{format}"}
    )


@router.post("/export/unified")
def export_unified(
    payload: Dict[str, Any],
    store: InMemoryStore = Depends(get_store)
):
    """
    Unified export for peaks data with optional metadata and double peak/double pair sections.
    """
    import logging
    import pandas as pd
    from io import StringIO, BytesIO
    
    logger = logging.getLogger(__name__)
    
    result_id = payload.get('resultId')
    filtered_result_id = payload.get('filteredResultId')
    double_peak_analysis = payload.get('double_peak_analysis')
    metadata = payload.get('metadata', {})
    export_format = payload.get('format', 'csv').lower()
    include_metadata = payload.get('include_metadata', False)
    filter_double_peaks = payload.get('filter_double_peaks', False)
    export_mode = payload.get('export_mode', 'single')  # single | double | combined
    double_peak_indices = payload.get('double_peak_indices') or []
    if isinstance(double_peak_indices, list):
        double_peak_indices = [int(idx) for idx in double_peak_indices if isinstance(idx, (int, float))]
    double_peak_thresholds = payload.get('double_peak_thresholds') or {}
    selection_source = payload.get('selection_source') or ('lasso' if double_peak_indices else 'filters')
    
    if not result_id:
        raise HTTPException(status_code=400, detail="resultId required")
        
    # Get base data
    base = store.get_result(result_id)
    if not base:
        raise HTTPException(status_code=404, detail="Result not found")
        
    time_values = base["time"]
    # Use time_resolution from payload if provided, otherwise fallback to base meta
    # This ensures consistency with frontend which uses params.time_resolution
    time_resolution = payload.get("time_resolution") or base["meta"].get("time_resolution", 1e-4)
    
    # Get peaks and properties
    # If passed in payload, use them. Otherwise detect.
    peaks = payload.get('peaks')
    properties = payload.get('properties')
    double_peak_params = payload.get('double_peak_params')
    
    if peaks is None:
        # Re-run detection
        prominence_threshold = payload.get('prominence_threshold', 20.0)
        distance = payload.get('distance', 30)
        rel_height = payload.get('rel_height', 0.8)
        width_ms = payload.get('width_ms', '0.1,200')
        prominence_ratio = payload.get('prominence_ratio', 0.8)
        
        # Get amplitude (filtered or original)
        if filtered_result_id:
            filtered_result = store.get_result(filtered_result_id)
            if filtered_result:
                amplitude = filtered_result["amplitude"]
            else:
                amplitude = base["amplitude"]
        else:
            amplitude = base["amplitude"]
            
        # Convert width
        if isinstance(width_ms, str):
            width_values = width_ms.strip().split(',')
        else:
            width_values = [str(v) for v in width_ms]
        sampling_rate = 1.0 / time_resolution
        width_p = [int(float(value.strip()) * sampling_rate / 1000) for value in width_values]
        
        peaks, properties = find_peaks_with_window(
            amplitude,
            width=width_p,
            prominence=prominence_threshold,
            distance=distance,
            rel_height=rel_height,
            prominence_ratio=prominence_ratio,
        )
    else:
        peaks = np.array(peaks)
        # Ensure properties are numpy arrays
        if properties:
            for key in properties:
                if not isinstance(properties[key], np.ndarray):
                    properties[key] = np.array(properties[key])
        else:
            properties = {}
            
    # If double peak analysis is missing but we have params (or an export mode that needs it), run it
    needs_double_section = export_mode in ("double", "combined")
    if needs_double_section and not double_peak_analysis:
        params_to_use = double_peak_params or {
            "min_distance": double_peak_thresholds.get("distance", [0.001, 0.100])[0] / 1000 if double_peak_thresholds else 0.001,
            "max_distance": double_peak_thresholds.get("distance", [0.001, 0.100])[1] / 1000 if double_peak_thresholds else 0.100,
            "min_amp_ratio": double_peak_thresholds.get("pairPromRatio", [0.1, 10.0])[0],
            "max_amp_ratio": double_peak_thresholds.get("pairPromRatio", [0.1, 10.0])[1],
            "min_width_ratio": double_peak_thresholds.get("pairWidthRatio", [0.1, 10.0])[0],
            "max_width_ratio": double_peak_thresholds.get("pairWidthRatio", [0.1, 10.0])[1],
        }
        double_peak_analysis = analyze_double_peaks_pure(
            peaks=peaks,
            properties=properties,
            time_resolution=time_resolution,
            min_distance=params_to_use.get('min_distance', 0.001),
            max_distance=params_to_use.get('max_distance', 0.100),
            min_amp_ratio=params_to_use.get('min_amp_ratio', 0.1),
            max_amp_ratio=params_to_use.get('max_amp_ratio', 10.0),
            min_width_ratio=params_to_use.get('min_width_ratio', 0.1),
            max_width_ratio=params_to_use.get('max_width_ratio', 10.0),
        )
        
    # Generate DataFrames
    frames = export_unified_peaks_data(
        time_values,
        peaks,
        properties,
        time_resolution,
        double_peak_analysis,
        filter_double_peaks if needs_double_section else False,
        pair_indices=double_peak_indices if needs_double_section else None,
        include_double_flags=(export_mode != "single")
    )

    peaks_df = frames["peaks_df"]
    double_df = frames["double_df"]

    # Remove mask column for double-only export (requested clean view)
    if export_mode == "double":
        if "Is Double Peak" in peaks_df.columns:
            peaks_df = peaks_df.drop(columns=["Is Double Peak"])
        if "Is Double Peak" in double_df.columns:
            double_df = double_df.drop(columns=["Is Double Peak"])

    # Build metadata blocks
    single_metadata_items = list(metadata.items()) if metadata else []
    double_metadata_items: List[tuple] = []
    if needs_double_section:
        # Use thresholds if present; otherwise fall back to params and analysis counts
        if double_peak_thresholds:
            dist = double_peak_thresholds.get("distance", [None, None])
            if dist and len(dist) == 2:
                double_metadata_items.append(("Distance Min (ms)", dist[0]))
                double_metadata_items.append(("Distance Max (ms)", dist[1]))
            prom_ratio = double_peak_thresholds.get("pairPromRatio")
            if prom_ratio and len(prom_ratio) == 2:
                double_metadata_items.append(("Prominence Ratio Min", prom_ratio[0]))
                double_metadata_items.append(("Prominence Ratio Max", prom_ratio[1]))
            width_ratio = double_peak_thresholds.get("pairWidthRatio")
            if width_ratio and len(width_ratio) == 2:
                double_metadata_items.append(("Width Ratio Min", width_ratio[0]))
                double_metadata_items.append(("Width Ratio Max", width_ratio[1]))
            prom_over_amp = double_peak_thresholds.get("promOverAmp")
            if prom_over_amp and len(prom_over_amp) == 2:
                double_metadata_items.append(("Prominence/Amplitude Min", prom_over_amp[0]))
                double_metadata_items.append(("Prominence/Amplitude Max", prom_over_amp[1]))

        if double_peak_analysis:
            double_metadata_items.append(("Total Pairs", double_peak_analysis.get("total_pairs", 0)))
            double_metadata_items.append(("Double Peak Count", double_peak_analysis.get("double_peak_count", 0)))

        selected_count = len(double_df) if not double_df.empty else (len(double_peak_indices) if double_peak_indices else 0)
        double_metadata_items.append(("Selected Pair Count", selected_count))
        double_metadata_items.append(("Selection Source", selection_source))

    # If everything is empty, return an empty response with headers
    if peaks_df.empty and double_df.empty:
        peaks_df = pd.DataFrame(columns=[
            'Time (s)', 'Amplitude', 'Width (ms)', 'Width (samples)', 'Interval (s)'
        ])

    # Handle Export Format
    if export_format == 'xlsx':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if include_metadata and (single_metadata_items or double_metadata_items):
                meta_rows = []
                for key, val in single_metadata_items:
                    meta_rows.append(("Single Peaks", key, val))
                for key, val in double_metadata_items:
                    meta_rows.append(("Double Peaks", key, val))
                meta_df = pd.DataFrame(meta_rows, columns=['Section', 'Parameter', 'Value'])
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)

            peaks_df.to_excel(writer, sheet_name='Single Peaks', index=False)

            if needs_double_section and not double_df.empty:
                double_df.to_excel(writer, sheet_name='Double Peak Pairs', index=False)
            
        output.seek(0)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        filename = f"{export_mode}_peaks_analysis.xlsx"
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    else: # CSV or TXT
        output = StringIO()
        
        if include_metadata and (single_metadata_items or double_metadata_items):
            output.write("# Metadata\n")
            for k, v in single_metadata_items:
                output.write(f"# {k}: {v}\n")
            if double_metadata_items:
                output.write("# Double Peak Metadata\n")
                for k, v in double_metadata_items:
                    output.write(f"# {k}: {v}\n")
            output.write("\n")

        sep = ',' if export_format == 'csv' else '\t'
        # Single peaks section
        output.write("# Single Peaks\n")
        peaks_df.to_csv(output, index=False, sep=sep)

        # Double peak section (optional)
        if needs_double_section and not double_df.empty:
            output.write("\n# Double Peak Pairs\n")
            double_df.to_csv(output, index=False, sep=sep)

        output.seek(0)
        
        media_type = "text/csv" if export_format == 'csv' else "text/plain"
        ext = "csv" if export_format == 'csv' else "txt"
        filename = f"{export_mode}_peaks_analysis.{ext}"
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
