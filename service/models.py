from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ProtocolInfo(BaseModel):
    """Experiment protocol metadata"""
    measurement_date: Optional[str] = None
    start_time: Optional[str] = None
    setup: Optional[str] = None
    sample_number: Optional[str] = None
    particle: Optional[str] = None
    concentration: Optional[str] = None
    buffer: Optional[str] = None
    buffer_concentration: Optional[str] = None
    nd_filter: Optional[str] = None
    laser_power: Optional[str] = None
    stamp: Optional[str] = None
    notes: Optional[str] = None


class LoadFilesRequest(BaseModel):
    paths: List[str] = Field(..., description="Absolute paths to data files")
    mode: Literal["single", "batch"] = "single"
    timestamps: Optional[List[str]] = None
    time_resolution: float = Field(1e-4, description="Seconds per sample")
    apply_dead_time_correction: bool = Field(False, description="Apply photon counter dead-time correction")
    dead_time_ns: float = Field(43.0, description="Dead time in nanoseconds")
    protocol: Optional[ProtocolInfo] = None


class FileMeta(BaseModel):
    total_files: int
    total_points: int
    time_range: List[float]
    files: List[str]
    protocol: Optional[ProtocolInfo] = None
    corrections: Optional[Dict[str, Any]] = None


class LoadFilesResponse(BaseModel):
    resultId: str
    meta: FileMeta
    preview: str


class Params(BaseModel):
    width_ms: str = "1,200"
    prominence_threshold: float = 20.0
    distance: int = 10
    rel_height: float = 0.5
    prominence_ratio: float = 0.8
    time_resolution: float = 1e-4
    
    # Filter parameters
    filter_enabled: bool = True
    filter_type: Literal["none", "butterworth", "savgol"] = "none"
    filter_cutoff_freq: float = 1000.0
    butter_order: int = 4
    savgol_window: int = 51
    savgol_polyorder: int = 3


class ProcessRunRequest(BaseModel):
    resultId: str
    params: Params


class DetectPeaksRequest(BaseModel):
    resultId: str
    filteredResultId: Optional[str] = None
    prominence_threshold: float
    distance: int
    rel_height: float
    width_ms: str
    prominence_ratio: float
    time_resolution: float


class TaskStatusResponse(BaseModel):
    status: str
    progress: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


