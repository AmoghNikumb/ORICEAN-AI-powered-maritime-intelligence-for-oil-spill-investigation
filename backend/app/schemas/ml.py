"""
Pydantic v2 schemas for the ML inference pipeline.

These models cover the full lifecycle of an oil spill detection request:

- ``MLInferenceRequest``  — payload for POST /api/ml/run-inference
- ``TilePrediction``      — per-tile classifier result (label + confidence)
- ``SpillDetection``      — merged spill polygon with aggregate confidence
- ``InferenceResult``     — complete inference run output (response model)
- ``MLHealthResponse``    — status of the loaded ML model
- ``ModelInfoResponse``   — architecture and metadata for the loaded model

All models are used by ``backend/app/api/ml.py`` and produced by
``backend/app/services/ml_service.py``.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class MLInferenceRequest(BaseModel):
    """Request payload for triggering ML inference on a SAR AOI."""

    model_config = ConfigDict(protected_namespaces=())

    investigation_id: str = Field(
        ...,
        description="UUID of the investigation this inference run is associated with.",
    )
    aoi_geojson: dict = Field(
        ...,
        description="GeoJSON Polygon geometry defining the Area of Interest.",
    )
    start_date: str = Field(
        ...,
        description="Start of the date range for Sentinel-1 imagery (ISO format, e.g. 2024-01-01).",
    )
    end_date: str = Field(
        ...,
        description="End of the date range for Sentinel-1 imagery (ISO format, e.g. 2024-01-15).",
    )
    model_variant: Literal["primary", "student"] = Field(
        default="primary",
        description=(
            "Which model weights to use: "
            "'primary' = sar_oil_slick_detection/model.pth (ResNet-50), "
            "'student' = best_student.pth."
        ),
    )
    confidence_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to classify a tile as an oil spill.",
    )
    stride: int | None = Field(
        default=None,
        ge=1,
        description="Sliding window stride in pixels. Defaults to tile_size // 2.",
    )


# ---------------------------------------------------------------------------
# Per-tile diagnostic helpers
# ---------------------------------------------------------------------------


class PixelStats(BaseModel):
    """Pixel-level statistics computed from the raw tile image before preprocessing."""

    min_value: float = Field(..., description="Minimum pixel value in the tile image.")
    max_value: float = Field(..., description="Maximum pixel value in the tile image.")
    mean_value: float = Field(..., description="Mean pixel value across the tile image.")
    std_value: float = Field(..., description="Standard deviation of pixel values in the tile image.")


class DebugInfo(BaseModel):
    """Preprocessing and model-input metadata for a single tile."""

    model_config = ConfigDict(protected_namespaces=())

    input_dimensions: tuple[int, int] = Field(
        ...,
        description="Original tile image dimensions as (height, width) before model preprocessing.",
    )
    preprocessing_steps: list[str] = Field(
        ...,
        description="Ordered list of preprocessing steps applied, e.g. ['resize_224x224', 'normalize'].",
    )
    model_input_dims: tuple[int, int, int] = Field(
        ...,
        description="Final tensor dimensions fed to the model as (channels, height, width), e.g. (3, 224, 224).",
    )


# ---------------------------------------------------------------------------
# Per-tile result
# ---------------------------------------------------------------------------


class TilePrediction(BaseModel):
    """
    Classification result for a single SAR tile.

    ``is_positive`` is derived by comparing ``confidence`` against the
    ``confidence_threshold`` supplied in the originating ``MLInferenceRequest``.
    """

    tile_id: str = Field(..., description="Unique identifier for this SAR tile.")
    label: Literal["No Oil", "Oil Spill"] = Field(
        ..., description="Predicted class label for this tile."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the tile contains an oil spill (p_oil from softmax).",
    )
    is_positive: bool = Field(
        ...,
        description="True when confidence >= confidence_threshold; tile is a spill candidate.",
    )
    bounds: tuple[float, float, float, float] = Field(
        ...,
        description="Geographic bounding box as (min_lon, min_lat, max_lon, max_lat).",
    )
    acquisition_date: str = Field(
        ...,
        description="ISO date string for when this Sentinel-1 image was acquired.",
    )
    mock: bool = Field(
        default=False,
        description="True when this prediction was produced in ML_MOCK_MODE (no real inference).",
    )
    image_url: str | None = Field(
        default=None,
        description="GEE thumbnail URL for the SAR tile image that was passed to the model.",
    )
    image_data_b64: str | None = Field(
        default=None,
        description="Base64-encoded PNG of the SAR tile; populated in mock/GEE_MOCK_MODE instead of image_url.",
    )
    pixel_stats: PixelStats | None = Field(
        default=None,
        description="Pixel-level statistics (min/max/mean/std) computed from the raw tile before preprocessing.",
    )
    debug_info: DebugInfo | None = Field(
        default=None,
        description="Preprocessing and model-input dimension metadata for this tile.",
    )
    gradcam_b64: str | None = Field(
        default=None,
        description="Base64-encoded Grad-CAM heatmap PNG; only present for OIL_DETECTED tiles.",
    )


# ---------------------------------------------------------------------------
# Aggregated spill detection
# ---------------------------------------------------------------------------


class SpillDetection(BaseModel):
    """
    A single merged spill region, constructed from the union of positive tile
    bounding boxes via Shapely.

    ``polygon_geojson`` holds a GeoJSON Polygon or MultiPolygon depending on
    whether adjacent tile boxes merge into one contiguous shape.
    """

    detection_id: str = Field(..., description="UUID for this spill detection event.")
    polygon_geojson: dict = Field(
        ...,
        description="GeoJSON Polygon or MultiPolygon geometry for the merged spill region.",
    )
    mean_confidence: float = Field(
        ...,
        description="Mean confidence across all positive tiles that contributed to this detection.",
    )
    max_confidence: float = Field(
        ...,
        description="Highest individual tile confidence within this detection.",
    )
    tile_count: int = Field(
        ...,
        description="Number of positive tiles that contributed to this merged spill region.",
    )
    area_km2: float | None = Field(
        default=None,
        description="Estimated surface area of the spill polygon in square kilometres (None if not computed).",
    )


# ---------------------------------------------------------------------------
# Full inference run result
# ---------------------------------------------------------------------------


class InferenceResult(BaseModel):
    """
    Complete output of a single ML inference run.

    Returned by POST /api/ml/run-inference and persisted to
    ``Investigation.ml_result`` as a JSONB document.
    """

    model_config = ConfigDict(protected_namespaces=())

    investigation_id: str = Field(
        ...,
        description="UUID of the investigation this result belongs to.",
    )
    model_version: str = Field(
        ...,
        description="Version string extracted from the loaded model (e.g. commit hash or filename).",
    )
    model_variant: str = Field(
        ...,
        description="Which model variant was used: 'primary' or 'student'.",
    )
    inference_timestamp: datetime.datetime = Field(
        ...,
        description="UTC datetime when the inference run was executed.",
    )
    total_tiles_processed: int = Field(
        ...,
        description="Total number of SAR tiles that were passed through the classifier.",
    )
    positive_tile_count: int = Field(
        ...,
        description="Number of tiles classified as oil spill (confidence >= threshold).",
    )
    spill_detections: list[SpillDetection] = Field(
        ...,
        description="Merged spill polygon detections derived from positive tiles.",
    )
    tile_results: list[TilePrediction] = Field(
        ...,
        description="Per-tile classification results for every tile that was processed.",
    )
    mock_mode: bool = Field(
        default=False,
        description="True when ML_MOCK_MODE=true was active during this inference run.",
    )
    warnings: list[str] = Field(
        default=[],
        description="Non-fatal warnings generated during the run (e.g. empty tile list, GEE partial data).",
    )
    sar_scene_url: str | None = Field(
        default=None,
        description="URL or data URI for the full SAR scene preview image used as the SAR layer on the map.",
    )


# ---------------------------------------------------------------------------
# Health and model metadata
# ---------------------------------------------------------------------------


class MLHealthResponse(BaseModel):
    """Response model for GET /api/ml/health."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ready", "not_loaded", "mock_mode"] = Field(
        ...,
        description=(
            "'ready' = model loaded and inference available; "
            "'not_loaded' = model file missing or failed to load; "
            "'mock_mode' = ML_MOCK_MODE=true, synthetic predictions only."
        ),
    )
    model_loaded: bool = Field(
        ...,
        description="True when model weights are resident in memory.",
    )
    model_version: str | None = Field(
        ...,
        description="Version identifier for the loaded model, or None if not loaded.",
    )
    device: str = Field(
        ...,
        description="Compute device in use, e.g. 'cpu', 'cuda', or 'mps'.",
    )


class ModelInfoResponse(BaseModel):
    """Response model for GET /api/ml/model-info."""

    model_config = ConfigDict(protected_namespaces=())

    architecture: str = Field(
        ...,
        description="Model architecture name, e.g. 'mobilenetv3_large_100'.",
    )
    num_classes: int = Field(
        ...,
        description="Number of output classes (2 for binary oil/no-oil classification).",
    )
    class_names: list[str] = Field(
        ...,
        description="Ordered list of class label strings, e.g. ['No Oil', 'Oil Spill'].",
    )
    input_size: list[int] = Field(
        ...,
        description="Expected input tensor size as [channels, height, width], e.g. [3, 224, 224].",
    )
    normalization: dict = Field(
        ...,
        description="Normalisation parameters used during training, e.g. {'mean': [...], 'std': [...]}.",
    )
    model_variant: str = Field(
        ...,
        description="Which variant of the model is currently loaded: 'primary' or 'student'.",
    )
