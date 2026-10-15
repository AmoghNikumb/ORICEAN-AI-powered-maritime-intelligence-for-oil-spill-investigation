"""
ML service for ORICEAN — orchestrates the full oil spill inference pipeline.

Responsibilities:
- Fetch SAR tiles from GEE via ``gee_service.get_sar_tiles()``
- Run batch inference via ``OilSpillPredictor.predict_batch()``
- Merge positive-tile bounding boxes into spill polygons via Shapely
- Return a structured ``InferenceResult`` Pydantic model

Three public symbols are exported:
- ``MLServiceError``          — custom exception (HTTP 503 by default)
- ``build_spill_polygons()``  — pure polygon construction from tile list
- ``run_ml_inference()``      — async end-to-end pipeline orchestrator
- ``get_predictor()``         — FastAPI ``Depends``-compatible dependency
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import io
import logging
import uuid
from typing import List

import numpy as np
import torch
import torch.nn as nn
from fastapi import Request
from PIL import Image
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from torchvision import transforms

from app.core.config import settings
from app.schemas.ml import (
    DebugInfo,
    InferenceResult,
    MLInferenceRequest,
    PixelStats,
    SpillDetection,
    TilePrediction,
)
from app.services import gee_service
from app.services.gee_service import GEEServiceError, SARScene

# Import the predictor's TilePrediction dataclass under an alias to avoid
# collision with the schema's TilePrediction Pydantic model.
from ml.inference.predictor import OilSpillPredictor
from ml.inference.predictor import TilePrediction as PredictorTilePrediction  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

_CROP_CAP = 50   # MVP cap — keeps GEE quota safe and inference fast
_DEG_TO_KM = 111.0  # 1 degree ≈ 111 km at the equator


class MLServiceError(Exception):
    """Raised when the ML service cannot fulfil an inference request.

    Attributes
    ----------
    message:
        Human-readable description of the error.
    status_code:
        Suggested HTTP status code (default 503 — Service Unavailable).
    """

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------------------
# 8.1 — build_spill_polygons
# ---------------------------------------------------------------------------


def build_spill_polygons(
    positive_tiles: List[TilePrediction],
    merge_overlapping: bool = True,
) -> List[SpillDetection]:
    """Convert positive-tile bounding boxes into merged GeoJSON spill polygons.

    Each positive tile's ``bounds`` tuple ``(min_lon, min_lat, max_lon, max_lat)``
    is converted to a Shapely ``box``.  When ``merge_overlapping=True`` (the
    default), all boxes are unioned via ``shapely.ops.unary_union`` so that
    spatially adjacent or overlapping tiles form a single contiguous region.

    For each resulting region the function finds the contributing tiles,
    computes aggregate confidence statistics, and estimates the surface area.

    Parameters
    ----------
    positive_tiles:
        List of ``TilePrediction`` objects with ``is_positive=True``.
    merge_overlapping:
        When ``True`` (default) spatially connected tile boxes are merged
        into one ``SpillDetection``.  When ``False`` each tile produces its
        own ``SpillDetection``.

    Returns
    -------
    list[SpillDetection]
        One entry per merged (or individual) spill region.
        Returns ``[]`` for an empty input without raising.
    """
    if not positive_tiles:
        return []

    # Build (shapely_box, confidence, tile_id) triples
    shapes: list[tuple] = []
    for tile in positive_tiles:
        min_lon, min_lat, max_lon, max_lat = tile.bounds
        geom = box(min_lon, min_lat, max_lon, max_lat)
        shapes.append((geom, tile.confidence, tile.tile_id))

    if not merge_overlapping:
        # Each tile becomes its own detection
        detections: List[SpillDetection] = []
        for geom, confidence, tile_id in shapes:
            area_km2 = geom.area * (_DEG_TO_KM ** 2)
            detections.append(
                SpillDetection(
                    detection_id=str(uuid.uuid4()),
                    polygon_geojson=dict(mapping(geom)),
                    mean_confidence=confidence,
                    max_confidence=confidence,
                    tile_count=1,
                    area_km2=area_km2,
                )
            )
        return detections

    # Merge all boxes into one geometry (may produce Polygon or MultiPolygon)
    merged = unary_union([s[0] for s in shapes])

    # Normalise to a flat list of individual Polygon regions
    if merged.geom_type == "Polygon":
        region_list = [merged]
    else:
        # MultiPolygon or GeometryCollection
        region_list = list(merged.geoms)

    detections = []
    for region in region_list:
        # Find which original tile boxes intersect this region
        contributing = [
            (geom, conf, tid)
            for (geom, conf, tid) in shapes
            if geom.intersects(region)
        ]

        if not contributing:
            continue  # defensive — should not happen in practice

        confidences = [conf for (_, conf, _) in contributing]
        mean_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
        area_km2 = region.area * (_DEG_TO_KM ** 2)

        detections.append(
            SpillDetection(
                detection_id=str(uuid.uuid4()),
                polygon_geojson=dict(mapping(region)),
                mean_confidence=mean_conf,
                max_confidence=max_conf,
                tile_count=len(contributing),
                area_km2=area_km2,
            )
        )

    return detections


# ---------------------------------------------------------------------------
# 3.1 — compute_pixel_stats
# ---------------------------------------------------------------------------


def _is_nodata_crop(image_bytes: bytes, zero_fraction: float = 0.98) -> bool:
    """Return ``True`` when a crop is essentially empty (GEE nodata).

    A crop counts as nodata when at least ``zero_fraction`` of its pixels are
    exactly 0 — the value GEE writes outside a scene's footprint. Undecodable
    bytes are treated as nodata so they are dropped rather than fed to the model.

    Parameters
    ----------
    image_bytes:
        Raw PNG bytes of a single 224x224 crop.
    zero_fraction:
        Minimum fraction of zero pixels to classify the crop as nodata.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.asarray(img)
        if arr.size == 0:
            return True
        return float(np.count_nonzero(arr == 0)) / arr.size >= zero_fraction
    except Exception:
        return True


def compute_pixel_stats(image_bytes: bytes) -> PixelStats | None:
    """Compute min, max, mean, and std pixel statistics from raw image bytes.

    Converts the image to a float32 numpy array and operates over the entire
    flattened pixel array (all channels, all pixels).

    Parameters
    ----------
    image_bytes:
        Raw bytes of any PIL-readable image (e.g. PNG, JPEG).

    Returns
    -------
    PixelStats | None
        A ``PixelStats`` instance on success, or ``None`` on any failure.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        arr = np.array(img).astype(np.float32).ravel()
        return PixelStats(
            min_value=float(arr.min()),
            max_value=float(arr.max()),
            mean_value=float(arr.mean()),
            std_value=float(arr.std()),
        )
    except Exception as e:
        logger.warning(f"compute_pixel_stats failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 3.2 — extract_sliding_window_tiles
# ---------------------------------------------------------------------------

_MAX_CROPS = 50  # per-scene cap — MVP


def extract_sliding_window_tiles(
    raster_bytes: bytes,
    scene_bounds: tuple[float, float, float, float],
    raster_width_px: int,
    raster_height_px: int,
    tile_size: int = 224,
    stride: int | None = None,
) -> tuple[list[tuple[bytes, int, int, tuple[float, float, float, float]]], bool]:
    """Slice a full-resolution raster into overlapping tiles using a sliding window.

    Opens ``raster_bytes`` with PIL and generates crops at every
    ``(col * stride, row * stride)`` offset. Each crop is resized to
    ``tile_size × tile_size`` with Lanczos resampling and serialised as PNG bytes.
    Per-crop geographic bounds are computed from the scene geotransform.

    The result is capped at _MAX_CROPS crops. If more crops would be produced
    naturally, only the first _MAX_CROPS are returned and ``truncated=True`` is
    signalled to the caller so it can append a warning.

    Parameters
    ----------
    raster_bytes:
        Raw bytes of any PIL-readable image (e.g. GEO_TIFF, PNG).
    scene_bounds:
        Geographic bounding box of the full scene as (min_lon, min_lat, max_lon, max_lat).
    raster_width_px:
        Width of the full raster in pixels (used for geotransform).
    raster_height_px:
        Height of the full raster in pixels (used for geotransform).
    tile_size:
        Side length (px) of each square output tile (default 224).
    stride:
        Step between successive crop origins. Defaults to
        ``tile_size // 2`` (50% overlap) when ``None``.

    Returns
    -------
    tuple[list[tuple[bytes, int, int, tuple[float, float, float, float]]], bool]
        ``(crops, truncated)`` where ``crops`` is a list of
        ``(crop_png_bytes, col_idx, row_idx, crop_bounds)`` four-tuples and
        ``truncated`` indicates whether the list was capped at _MAX_CROPS.
        ``crop_bounds`` is (crop_min_lon, crop_min_lat, crop_max_lon, crop_max_lat).
    """
    img = Image.open(io.BytesIO(raster_bytes))
    W, H = img.size

    if stride is None:
        stride = tile_size  # default: no overlap — keeps tile count low for MVP

    # Compute geotransform: pixel → geographic coordinate mapping
    min_lon, min_lat, max_lon, max_lat = scene_bounds
    lon_per_px = (max_lon - min_lon) / raster_width_px
    lat_per_px = (max_lat - min_lat) / raster_height_px

    crops: list[tuple[bytes, int, int, tuple[float, float, float, float]]] = []
    truncated = False

    row = 0
    while row * stride < H:
        col = 0
        while col * stride < W:
            left = col * stride
            upper = row * stride
            right = min(left + tile_size, W)
            lower = min(upper + tile_size, H)

            crop = img.crop((left, upper, right, lower)).resize(
                (tile_size, tile_size), Image.LANCZOS
            )

            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop_bytes = buf.getvalue()

            # Compute per-crop geographic bounds from geotransform
            crop_min_lon = min_lon + left * lon_per_px
            crop_max_lon = min_lon + min(left + tile_size, raster_width_px) * lon_per_px
            crop_max_lat = max_lat - upper * lat_per_px
            crop_min_lat = max_lat - min(upper + tile_size, raster_height_px) * lat_per_px
            crop_bounds = (crop_min_lon, crop_min_lat, crop_max_lon, crop_max_lat)

            crops.append((crop_bytes, col, row, crop_bounds))

            if len(crops) >= _MAX_CROPS:
                truncated = True
                return crops, truncated

            col += 1
        row += 1

    return crops, truncated


# ---------------------------------------------------------------------------
# 3.3 — generate_gradcam
# ---------------------------------------------------------------------------

# ImageNet normalisation constants used during training.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Pre-computed tensors for manual normalization (avoids torchvision ToTensor numpy issue)
_IMAGENET_MEAN_T = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
_IMAGENET_STD_T  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a normalized [3, 224, 224] float tensor.

    Bypasses torchvision.transforms.ToTensor() which breaks in asyncio
    thread contexts due to numpy C-API initialization issues.
    """
    img_resized = pil_img.resize((224, 224), Image.BILINEAR).convert("RGB")
    arr = np.array(img_resized, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)        # [3, H, W]
    return (tensor - _IMAGENET_MEAN_T) / _IMAGENET_STD_T


def generate_gradcam(
    model: nn.Module,
    image_bytes: bytes,
    device: str,
) -> str | None:
    """Generate a CAM heatmap for a tile using Class Activation Mapping.

    For MobileNetV3-style models (global avg pool → linear classifier),
    CAM is computed directly from conv_head activations weighted by the
    classifier's output weights — no backward pass needed, avoiding
    inplace-operation conflicts in PyTorch's autograd.

    Returns a base64-encoded PNG of the jet heatmap blended over the
    original tile image, or None on any failure.
    """
    try:
        # --- Find conv_head (last conv before GAP) and classifier ---
        conv_layer: nn.Module | None = None
        classifier: nn.Module | None = None

        # MobileNetV3 (timm): conv_head + classifier
        if hasattr(model, "conv_head"):
            conv_layer = model.conv_head
        if hasattr(model, "classifier"):
            classifier = model.classifier

        # Generic fallback: last Conv2d + last Linear
        if conv_layer is None:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    conv_layer = m
        if classifier is None:
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    classifier = m

        if conv_layer is None or classifier is None:
            logger.warning("generate_gradcam: could not find conv + classifier layers")
            return None

        # --- Preprocess image ---
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        rgb_arr = np.array(pil_img).astype(np.float32) / 255.0
        input_tensor = _pil_to_tensor(pil_img).unsqueeze(0).to(device)

        # --- Capture conv activations via forward hook ---
        activations: list[torch.Tensor] = []

        def fwd_hook(module: nn.Module, inp: tuple, output: torch.Tensor) -> None:
            activations.append(output.clone().detach())

        handle = conv_layer.register_forward_hook(fwd_hook)

        with torch.no_grad():
            _ = model(input_tensor)

        handle.remove()

        if not activations:
            return None

        act = activations[0].squeeze(0)  # [C, H, W]

        # --- Get classifier weights for the oil spill class (class 1) ---
        # For timm MobileNetV3, classifier is Sequential(Linear, ...) or just Linear
        fc_weight: torch.Tensor | None = None
        if isinstance(classifier, nn.Linear):
            fc_weight = classifier.weight.data  # [num_classes, C]
        elif isinstance(classifier, nn.Sequential):
            for m in classifier.modules():
                if isinstance(m, nn.Linear):
                    fc_weight = m.weight.data
                    break

        if fc_weight is not None and fc_weight.shape[1] == act.shape[0]:
            # CAM: weighted sum of activations using oil-class weights
            weights = fc_weight[1]  # class 1 = oil spill, shape [C]
            cam = torch.zeros(act.shape[1:], dtype=torch.float32)
            for i, w in enumerate(weights):
                cam += w.item() * act[i]
        else:
            # Fallback: mean activation across channels
            cam = act.mean(dim=0)

        # ReLU and normalize
        cam = torch.clamp(cam, min=0)
        cam_np = cam.cpu().numpy()

        cam_resized = np.array(
            Image.fromarray(cam_np).resize((224, 224), Image.BILINEAR)
        )
        if cam_resized.max() > cam_resized.min():
            cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
        else:
            cam_norm = cam_resized

        # --- Apply jet colormap and blend ---
        def jet_colormap(t: np.ndarray) -> np.ndarray:
            r = np.clip(1.5 - np.abs(t * 4.0 - 3.0), 0, 1)
            g = np.clip(1.5 - np.abs(t * 4.0 - 2.0), 0, 1)
            b = np.clip(1.5 - np.abs(t * 4.0 - 1.0), 0, 1)
            return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

        heatmap_rgb = jet_colormap(cam_norm).astype(np.float32) / 255.0
        alpha = 0.5
        blended = (rgb_arr * (1 - alpha) + heatmap_rgb * alpha).clip(0, 1)
        blended_uint8 = (blended * 255).astype(np.uint8)

        out_img = Image.fromarray(blended_uint8)
        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        logger.warning(f"generate_gradcam failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 8.2 — run_ml_inference
# ---------------------------------------------------------------------------


async def run_ml_inference(
    request: MLInferenceRequest,
    predictor: OilSpillPredictor,
) -> InferenceResult:
    """Orchestrate the full SAR → ML → polygon inference pipeline.

    Steps
    -----
    1. Validate that the predictor is ready (or that mock mode is active).
    2. Fetch SAR scenes from GEE via ``gee_service.get_sar_tiles()``.
    3. Run sliding window extraction on each scene's full-resolution raster,
       collecting per-crop geographic bounds from the geotransform.
    4. Apply total crop cap (_CROP_CAP) across all scenes; append warning if hit.
    5. Run ``predictor.predict_batch()`` in a thread pool (non-blocking).
    6. Zip crops + predictions into schema ``TilePrediction`` objects,
       applying ``request.confidence_threshold`` and per-crop bounds.
    7. Build merged spill polygons from positive tiles.
    8. Assemble and return ``InferenceResult``.

    Parameters
    ----------
    request:
        Validated ``MLInferenceRequest`` from the API layer.
    predictor:
        App-level ``OilSpillPredictor`` instance (injected via dependency).

    Returns
    -------
    InferenceResult
        Always returned — zero detections on GEE failure (with warning).

    Raises
    ------
    MLServiceError
        When the predictor is not loaded and ``ML_MOCK_MODE`` is ``False``.
    """
    warnings: List[str] = []

    # --- Guard: model must be loaded unless we're in mock mode ---
    if not predictor.is_loaded and not settings.ML_MOCK_MODE:
        raise MLServiceError(
            "ML model is not loaded. Check model files in backend/ml/models/",
            status_code=503,
        )

    # --- Step 1: Fetch SAR scenes from GEE (returns list[SARScene]) ---
    scenes: list[SARScene] = []
    try:
        scenes = await gee_service.get_sar_tiles(
            aoi_geojson=request.aoi_geojson,
            start_date=request.start_date,
            end_date=request.end_date,
        )
    except GEEServiceError as exc:
        logger.warning("GEE scene fetch failed: %s", exc)
        warnings.append(f"GEE tile fetch failed: {exc}")
        return _zero_detection_result(request, predictor, warnings)

    if not scenes:
        warnings.append("No SAR tiles found for the given AOI and date range.")
        return _zero_detection_result(request, predictor, warnings)

    # --- Step 2: Sliding window extraction on full-resolution rasters ---
    # all_crops: list of (crop_bytes, col, row, crop_bounds)
    all_crops: list[tuple[bytes, int, int, tuple[float, float, float, float]]] = []
    crop_parents: list[SARScene] = []  # parallel list of SARScene for each crop

    # Nodata crops (the Sentinel-1 acquisition only clips part of the AOI, so
    # those crops are pure fill) are dropped *here*, before they count toward
    # _CROP_CAP. Otherwise the cap could be spent entirely on blank tiles from
    # a partially-covered scene and the real imagery would never be classified.
    nodata_dropped = 0
    for scene in scenes:
        crops, truncated = extract_sliding_window_tiles(
            scene.raster_bytes,
            scene.bounds,
            scene.raster_width_px,
            scene.raster_height_px,
            stride=request.stride,
        )
        if truncated:
            warnings.append(
                f"Sliding window capped at {_MAX_CROPS} crops for scene {scene.scene_id}"
            )
        for crop_bytes, col, row, crop_bounds in crops:
            if _is_nodata_crop(crop_bytes):
                nodata_dropped += 1
                continue

            all_crops.append((crop_bytes, col, row, crop_bounds))
            crop_parents.append(scene)

            # Apply total (usable) crop cap across all scenes
            if len(all_crops) >= _CROP_CAP:
                warnings.append(
                    f"Total crop count capped at {_CROP_CAP} across all scenes"
                )
                break
        else:
            continue  # inner loop completed normally — continue outer loop
        break  # inner loop hit _CROP_CAP — stop processing scenes too

    if nodata_dropped:
        warnings.append(
            f"Skipped {nodata_dropped} empty (nodata) tile(s) — the Sentinel-1 "
            f"scene does not fully cover this AOI on the selected dates"
        )

    if not all_crops:
        warnings.append(
            "No usable SAR pixels for this AOI and date range — try a wider date "
            "window or an AOI with full Sentinel-1 coverage"
        )
        return _zero_detection_result(request, predictor, warnings)

    # --- Step 3: Run batch inference on all crops (off the event loop) ---
    predictor_results: list[PredictorTilePrediction] = await asyncio.to_thread(
        predictor.predict_batch, [c[0] for c in all_crops]
    )

    # Debug: log confidence distribution to diagnose model output
    if predictor_results:
        confs = [p.confidence for p in predictor_results]
        logger.info(
            "Confidence stats over %d tiles — min=%.4f max=%.4f mean=%.4f",
            len(confs), min(confs), max(confs), sum(confs) / len(confs)
        )

    # --- Step 4: Assemble TilePrediction objects, one per crop ---
    # predict_batch may return fewer items if some crops fail decoding;
    # zip stops at the shorter sequence — skipped crops are simply omitted.
    tile_results: List[TilePrediction] = []
    for (crop_bytes, col, row, crop_bounds), scene, pred in zip(all_crops, crop_parents, predictor_results):
        is_positive = pred.confidence >= request.confidence_threshold
        label = "Oil Spill" if is_positive else "No Oil"

        # Unique tile ID per crop
        crop_tile_id = f"{scene.scene_id}_r{row}_c{col}"

        # Pixel stats for debug
        pixel_stats = compute_pixel_stats(crop_bytes)

        # Debug info: crop is always 224×224 after extract_sliding_window_tiles
        debug_info = DebugInfo(
            input_dimensions=(224, 224),
            preprocessing_steps=["resize_224x224", "normalize"],
            model_input_dims=(3, 224, 224),
        )

        # Image encoding: base64 PNG for both mock and real scenes
        image_url: str | None = None
        image_data_b64: str | None = None
        image_data_b64 = base64.b64encode(crop_bytes).decode("utf-8")

        # Grad-CAM for positive tiles only
        gradcam_b64: str | None = None
        if is_positive and predictor.is_loaded:
            gradcam_b64 = generate_gradcam(predictor.model, crop_bytes, predictor.device)

        tile_results.append(
            TilePrediction(
                tile_id=crop_tile_id,
                label=label,
                confidence=pred.confidence,
                is_positive=is_positive,
                bounds=crop_bounds,                  # per-crop geographic bounds
                acquisition_date=scene.acquisition_date,
                mock=scene.mock,
                image_url=image_url,
                image_data_b64=image_data_b64,
                pixel_stats=pixel_stats,
                debug_info=debug_info,
                gradcam_b64=gradcam_b64,
            )
        )

    # --- Step 5: Build spill polygons from positive tiles ---
    positive_tiles = [t for t in tile_results if t.is_positive]
    spill_detections = build_spill_polygons(positive_tiles)

    # --- Step 6: Determine mock_mode flag ---
    any_mock_scene = any(s.mock for s in crop_parents)
    mock_mode = settings.ML_MOCK_MODE or any_mock_scene

    # --- Step 7: Build sar_scene_url — fetch a proper AOI-clipped preview thumbnail ---
    sar_scene_url: str | None = None
    try:
        preview_bytes = await gee_service.get_sar_preview(
            aoi_geojson=request.aoi_geojson,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        if preview_bytes:
            preview_b64 = base64.b64encode(preview_bytes).decode("utf-8")
            sar_scene_url = f"data:image/png;base64,{preview_b64}"
    except Exception as preview_exc:
        logger.warning("SAR preview generation failed: %s", preview_exc)

    return InferenceResult(
        investigation_id=request.investigation_id,
        model_version=predictor.model_version,
        model_variant=request.model_variant,
        inference_timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
        total_tiles_processed=len(tile_results),
        positive_tile_count=len(positive_tiles),
        spill_detections=spill_detections,
        tile_results=tile_results,
        mock_mode=mock_mode,
        warnings=warnings,
        sar_scene_url=sar_scene_url,
    )


# ---------------------------------------------------------------------------
# 8.3 — get_predictor FastAPI dependency
# ---------------------------------------------------------------------------


def get_predictor(request: Request) -> OilSpillPredictor:
    """FastAPI dependency — return the app-level ``OilSpillPredictor`` instance.

    The predictor is stored on ``app.state.predictor`` during the FastAPI
    lifespan startup event.  Inject this function with ``Depends(get_predictor)``
    in any route that needs ML inference.

    Parameters
    ----------
    request:
        FastAPI ``Request`` object (injected automatically by the framework).

    Returns
    -------
    OilSpillPredictor
        The shared predictor instance for this application process.
    """
    return request.app.state.predictor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _zero_detection_result(
    request: MLInferenceRequest,
    predictor: OilSpillPredictor,
    warnings: List[str],
) -> InferenceResult:
    """Return an ``InferenceResult`` with zero tiles and zero detections.

    Used when GEE fails or returns no tiles so the caller always receives
    a well-formed response rather than an exception.
    """
    return InferenceResult(
        investigation_id=request.investigation_id,
        model_version=predictor.model_version,
        model_variant=request.model_variant,
        inference_timestamp=datetime.datetime.now(tz=datetime.timezone.utc),
        total_tiles_processed=0,
        positive_tile_count=0,
        spill_detections=[],
        tile_results=[],
        mock_mode=settings.ML_MOCK_MODE,
        warnings=warnings,
        sar_scene_url=None,
    )
