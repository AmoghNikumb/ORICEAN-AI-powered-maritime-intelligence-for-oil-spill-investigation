"""
Google Earth Engine service for ORICEAN.

Provides SAR tile delivery from the COPERNICUS/S1_GRD Sentinel-1 collection.
Exports each scene as a 224×224 PNG thumbnail and returns structured
``SARTile`` objects for downstream ML inference.

Mock path
---------
When ``GEE_MOCK_MODE=true`` no GEE calls are made.  Three synthetic
224×224 greyscale noise tiles are returned instead, each tagged
``mock=True``.  This mode is intended for development and demo runs where
a live GEE connection is unavailable.

Real path
---------
Requires a valid GEE service-account key at
``settings.GOOGLE_APPLICATION_CREDENTIALS`` (or application-default
credentials) and a project ID at ``settings.GEE_PROJECT_ID``.
Authentication is attempted once; if it fails ``GEEServiceError`` is raised
before any imagery request is made.
"""

from __future__ import annotations

import asyncio
import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

# A mosaicked acquisition must cover at least this fraction of the AOI to be
# worth running inference on — below it the raster is mostly nodata fill.
_MIN_SCENE_VALID_FRACTION: float = 0.10


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class GEEServiceError(Exception):
    """Raised when Google Earth Engine authentication or API calls fail.

    NOT raised for empty result sets — an empty ``list[SARTile]`` is
    returned when no Sentinel-1 imagery exists for the given AOI / date range.
    """


# ---------------------------------------------------------------------------
# SARScene dataclass
# ---------------------------------------------------------------------------


@dataclass
class SARScene:
    """A full-resolution Sentinel-1 SAR scene downloaded from GEE.

    Attributes
    ----------
    scene_id:
        GEE image ID or synthetic identifier.
    raster_bytes:
        Raw GEO_TIFF bytes of the VV band, dB-converted and clipped to [-25, 0] dB.
    acquisition_date:
        ISO-8601 date string of the Sentinel-1 acquisition (YYYY-MM-DD).
    orbit_pass:
        "ASCENDING" or "DESCENDING".
    bounds:
        Geographic bounding box as (min_lon, min_lat, max_lon, max_lat).
    raster_width_px:
        Width of the raster in pixels.
    raster_height_px:
        Height of the raster in pixels.
    mock:
        True when this scene was synthetically generated (mock mode).
    scale_m:
        GEE export scale used in metres per pixel.
    valid_fraction:
        Fraction of the raster (AOI bbox) that carries real Sentinel-1 data
        rather than nodata fill. 1.0 for mock scenes and fully-covered real
        scenes; lower when the acquisition only clips part of the AOI.
    """
    scene_id: str
    raster_bytes: bytes        # uint8 PNG bytes of VV band, dB-converted and normalized to [0,255]
    acquisition_date: str      # ISO date YYYY-MM-DD
    orbit_pass: str            # "ASCENDING" | "DESCENDING"
    bounds: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    raster_width_px: int
    raster_height_px: int
    mock: bool = False
    scale_m: int = 40
    valid_fraction: float = 1.0


# ---------------------------------------------------------------------------
# SARTile dataclass
# ---------------------------------------------------------------------------


@dataclass
class SARTile:
    """A single Sentinel-1 SAR tile ready for ML inference.

    Attributes
    ----------
    tile_id:
        Unique identifier for this tile (GEE image ID or ``mock_tile_{i}``).
    image_bytes:
        Raw PNG bytes of the 224×224 single-band (VV) tile.
    bounds:
        Geographic bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
    acquisition_date:
        ISO-8601 date string of the Sentinel-1 acquisition (``YYYY-MM-DD``).
    orbit_pass:
        ``"ASCENDING"`` or ``"DESCENDING"``.
    mock:
        ``True`` when this tile was synthetically generated (mock mode).
    thumb_url:
        Thumbnail URL returned by GEE ``getThumbURL``.  ``None`` in mock mode
        or when the fallback ``getDownloadURL`` path was used.
    """

    tile_id: str
    image_bytes: bytes
    bounds: tuple[float, float, float, float]
    acquisition_date: str
    orbit_pass: str
    mock: bool = False
    thumb_url: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_aoi_bbox(aoi_geojson: dict) -> tuple[float, float, float, float]:
    """Return ``(min_lon, min_lat, max_lon, max_lat)`` from a GeoJSON Polygon.

    Parameters
    ----------
    aoi_geojson:
        A GeoJSON ``Polygon`` feature or geometry dict.

    Returns
    -------
    tuple[float, float, float, float]
        Bounding box as ``(min_lon, min_lat, max_lon, max_lat)``.
    """
    # Handle both Feature and Geometry inputs
    geom = aoi_geojson.get("geometry", aoi_geojson)
    coords = geom["coordinates"][0]  # outer ring

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def _fetch_url_bytes(url: str) -> bytes:
    """Synchronously download bytes from a URL (runs in a thread pool)."""
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def _make_noise_png(size: int = 224) -> bytes:
    """Generate a synthetic greyscale noise PNG of ``size × size`` pixels."""
    arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ms_to_iso_date(ms: int) -> str:
    """Convert a GEE millisecond epoch timestamp to an ISO date string."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _unzip_gee_response(raw_bytes: bytes) -> bytes:
    """Extract the TIFF from a GEE getDownloadURL response.

    GEE's ``getDownloadURL`` wraps the GEO_TIFF in a ZIP archive.
    This helper detects the content type and unpacks accordingly.

    Parameters
    ----------
    raw_bytes:
        Raw bytes downloaded from the GEE download URL.

    Returns
    -------
    bytes
        Raw TIFF bytes ready to be opened by PIL.

    Raises
    ------
    ValueError
        If the bytes are an HTML error page, an empty ZIP, or an
        unrecognised format.
    """
    if not raw_bytes:
        raise ValueError("GEE download returned empty response")

    # Detect HTML error pages (GEE returns these with HTTP 200 on some errors)
    if raw_bytes[:15].lstrip().startswith(b"<") or b"<html" in raw_bytes[:512].lower():
        # Try to extract a useful message from the HTML
        snippet = raw_bytes[:512].decode("utf-8", errors="replace")
        raise ValueError(f"GEE returned an HTML error page instead of a TIFF: {snippet[:200]}")

    # ZIP magic bytes: GEE wraps GEO_TIFF in a ZIP
    if raw_bytes[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                raise ValueError(
                    f"GEE ZIP contained no TIFF files; contents: {zf.namelist()}"
                )
            tif_bytes = zf.read(tif_names[0])
            logger.debug("Unzipped GEE response: extracted %s (%d bytes)", tif_names[0], len(tif_bytes))
            return tif_bytes

    # TIFF magic bytes: II (little-endian) or MM (big-endian)
    if raw_bytes[:2] in (b"II", b"MM"):
        return raw_bytes

    # Unrecognised format — log first bytes for diagnosis
    raise ValueError(
        f"GEE response has unrecognised format. "
        f"First 16 bytes: {raw_bytes[:16]!r}"
    )


# ---------------------------------------------------------------------------
# Mock path
# ---------------------------------------------------------------------------

MOCK_RASTER_SIZE = 512


def _make_noise_tif(size: int) -> bytes:
    """Generate a synthetic grayscale noise TIFF of ``size × size`` pixels.

    Returns raw TIFF bytes that PIL can open as a valid image.
    No GeoTIFF metadata is required — raster dimensions are passed
    explicitly on SARScene.

    Parameters
    ----------
    size:
        Side length in pixels of the square output image.

    Returns
    -------
    bytes
        Raw TIFF image bytes.
    """
    arr = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    return buf.getvalue()


def _get_mock_sar_tiles(
    aoi_geojson: dict,
) -> List[SARScene]:
    """Return a single synthetic 512×512 SAR scene for demo / development mode.

    Returns a list containing one SARScene with a synthetic noise raster of
    MOCK_RASTER_SIZE × MOCK_RASTER_SIZE pixels, producing ~16 tiles at
    default stride=112 (satisfying the > 4 tiles requirement).

    Parameters
    ----------
    aoi_geojson:
        GeoJSON Polygon describing the area of interest.

    Returns
    -------
    list[SARScene]
        One SARScene with mock=True and a 512×512 synthetic TIFF raster.
    """
    min_lon, min_lat, max_lon, max_lat = _extract_aoi_bbox(aoi_geojson)
    noise_raster = _make_noise_tif(MOCK_RASTER_SIZE)

    logger.info("Mock mode: returning 1 synthetic SARScene (%dx%d px)", MOCK_RASTER_SIZE, MOCK_RASTER_SIZE)
    return [SARScene(
        scene_id="mock_scene_0",
        raster_bytes=noise_raster,
        acquisition_date="2024-01-01",
        orbit_pass="ASCENDING",
        bounds=(min_lon, min_lat, max_lon, max_lat),
        raster_width_px=MOCK_RASTER_SIZE,
        raster_height_px=MOCK_RASTER_SIZE,
        mock=True,
        scale_m=40,
    )]


# ---------------------------------------------------------------------------
# Real GEE path (sync worker — called inside asyncio.to_thread)
# ---------------------------------------------------------------------------


def _authenticate_gee() -> None:
    """Initialise the GEE Python client.

    Uses a service-account key when ``GOOGLE_APPLICATION_CREDENTIALS`` is set,
    otherwise falls back to application-default credentials.

    Raises
    ------
    GEEServiceError
        If authentication or initialisation fails.
    """
    try:
        import ee  # deferred — not installed in every environment
    except ImportError as exc:
        raise GEEServiceError(
            "earthengine-api package is not installed. "
            "Run: pip install earthengine-api"
        ) from exc

    try:
        credentials_path = settings.GOOGLE_APPLICATION_CREDENTIALS.strip()
        project_id = settings.GEE_PROJECT_ID.strip() or None

        if credentials_path:
            import google.oauth2.service_account as sa  # type: ignore

            credentials = sa.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/earthengine"],
            )
            ee.Initialize(credentials=credentials, project=project_id)
            logger.info("GEE initialised with service-account credentials")
        else:
            ee.Initialize(project=project_id)
            logger.info("GEE initialised with application-default credentials")

    except Exception as exc:
        raise GEEServiceError(f"GEE authentication failed: {exc}") from exc


def _fetch_sar_tiles_sync(
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    tile_size_px: int,
) -> List[SARScene]:
    """Synchronous core of the GEE tile fetch (runs in a thread pool).

    Downloads each Sentinel-1 scene as a full-resolution GEO_TIFF at
    scale=40 m/px with server-side dB conversion and [-25, 0] dB clipping.
    Auto-retries at scale=80 then scale=160 if the AOI would produce a
    raster larger than 4096×4096 pixels at the requested scale.

    Parameters
    ----------
    aoi_geojson:
        GeoJSON Polygon for the AOI.
    start_date:
        Start of the date range (YYYY-MM-DD).
    end_date:
        End of the date range (YYYY-MM-DD).
    tile_size_px:
        Retained for API compatibility; not used in this implementation
        since full-resolution rasters are downloaded.

    Returns
    -------
    list[SARScene]
        May be empty if no Sentinel-1 scenes cover the AOI in the range.

    Raises
    ------
    GEEServiceError
        On GEE authentication or API errors.
    """
    try:
        import ee  # noqa: PLC0415
    except ImportError as exc:
        raise GEEServiceError("earthengine-api not installed") from exc

    _authenticate_gee()

    try:
        min_lon, min_lat, max_lon, max_lat = _extract_aoi_bbox(aoi_geojson)
        aoi_geometry = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi_geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        )

        # Check the filtered collection before calling toList. Earth Engine
        # rejects toList on an empty collection with "Empty date ranges not
        # supported for the current operation".
        n_images: int = collection.size().getInfo()
        logger.info("GEE returned %d Sentinel-1 scenes for AOI", n_images)

        if n_images == 0:
            return []

        image_list = collection.toList(100)  # cap at 100 scenes

        # Determine optimal initial scale based on AOI size
        # Target: keep raster under ~4096×4096 pixels
        lon_span = max_lon - min_lon
        lat_span = max_lat - min_lat
        # 1 degree ≈ 1/0.00036 pixels at scale=40m (0.00036 deg/px ≈ 40m at equator)
        deg_per_px_at_40m = 0.00036

        # Pick the coarsest scale that still fits in 4096 px
        initial_scale = 40
        if (lon_span / deg_per_px_at_40m > 4096) or (lat_span / deg_per_px_at_40m > 4096):
            initial_scale = 80
            if (lon_span / (deg_per_px_at_40m * 2) > 4096) or (lat_span / (deg_per_px_at_40m * 2) > 4096):
                initial_scale = 160

        # -------------------------------------------------------------------
        # Group scenes by (acquisition date, orbit pass) and mosaic each group.
        #
        # A single Sentinel-1 GRD product only clips a ~250 km frame of the
        # swath, so for a multi-degree AOI one scene often covers just a
        # fraction of it — the rest of the downloaded raster is nodata fill.
        # Previously the pipeline tiled whichever scene came first (sometimes
        # 1% AOI coverage) and the ML model saw 50 blank tiles → zero
        # detections. The consecutive frames of one pass tile the full swath,
        # so mosaicking same-date/same-orbit frames reconstructs full AOI
        # coverage without blending different acquisitions together.
        # -------------------------------------------------------------------
        groups: dict[tuple[str, str], list] = {}
        for i in range(n_images):
            try:
                image = ee.Image(image_list.get(i))
                info = image.getInfo()
                props = info.get("properties", {})
                time_start = props.get("system:time_start", 0)
                acq_date = _ms_to_iso_date(time_start) if time_start else start_date
                orbit_pass = (props.get("orbitProperties_pass", "ASCENDING") or "ASCENDING").upper()
                image_id = info.get("id", f"scene_{i}")
            except Exception as meta_exc:
                logger.warning("Skipping scene %d — metadata fetch failed: %s", i, meta_exc)
                continue
            groups.setdefault((acq_date, orbit_pass), []).append((image, image_id))

        logger.info(
            "GEE: %d scenes grouped into %d (date, orbit) acquisitions",
            n_images, len(groups),
        )

        scenes: List[SARScene] = []
        for (acq_date, orbit_pass), members in groups.items():
            try:
                imgs = [m[0] for m in members]
                first_id = members[0][1]
                scene_id = first_id if len(members) == 1 else f"{first_id}+{len(members) - 1}"

                # COPERNICUS/S1_GRD VV is ALREADY gamma-naught in decibels
                # (calm open ocean ≈ -18..-13 dB; oil slicks suppress backscatter
                # to roughly -22..-25 dB). It must NOT be passed through
                # 10*log10() again. Just clamp to the training dB window
                # (ml/preprocessing:preprocess.py uses [-25, 0]); the float32
                # GeoTIFF is renormalised to uint8 below.
                image_vv = (
                    ee.ImageCollection(imgs).mosaic()
                    .select("VV")
                    .rename("VV_db")
                    .clamp(-25.0, 0.0)
                )

                raw_bytes: bytes | None = None
                used_scale = initial_scale

                # Try downloading at initial_scale, retry at coarser scales if needed
                for scale in [initial_scale, 80, 160]:
                    if scale < initial_scale:
                        continue  # don't retry at finer scale than initial
                    try:
                        dl_url = image_vv.getDownloadURL({
                            "scale": scale,
                            "region": aoi_geometry,
                            "format": "GEO_TIFF",
                            "crs": "EPSG:4326",
                        })
                        raw_bytes = _fetch_url_bytes(dl_url)
                        raw_bytes = _unzip_gee_response(raw_bytes)
                        used_scale = scale
                        logger.info(
                            "getDownloadURL OK for %s (%s, %s) at scale=%dm (%d bytes)",
                            scene_id, acq_date, orbit_pass, scale, len(raw_bytes),
                        )
                        break
                    except Exception as dl_exc:
                        logger.warning(
                            "getDownloadURL failed for %s at scale=%dm: %s — trying coarser scale",
                            scene_id, scale, dl_exc,
                        )

                if raw_bytes is None:
                    logger.warning("All download attempts failed for %s — skipping.", scene_id)
                    continue

                # Read raster dimensions and convert float32 GeoTIFF → uint8 PNG.
                # GEE GeoTIFFs are float32 single-band; PIL can't open them directly.
                try:
                    import rasterio  # noqa: PLC0415
                    from rasterio.io import MemoryFile  # noqa: PLC0415

                    with MemoryFile(raw_bytes) as memfile:
                        with memfile.open() as dataset:
                            raster_width_px = dataset.width
                            raster_height_px = dataset.height
                            band = dataset.read(1).astype(np.float32)

                    # Pixels outside the acquisition footprint come back as an
                    # exact 0.0 (GEO_TIFF nodata fill). Real S1 GRD VV in dB over
                    # water is essentially never exactly 0, so treat 0/±inf/NaN as
                    # nodata: map it to 0 (black) so ml_service's nodata-tile
                    # filter drops it, instead of the old behaviour where the
                    # [-25,0] dB → [0,255] stretch turned nodata into 255 (white)
                    # and fed the model 50 blank tiles.
                    valid = np.isfinite(band) & (band != 0.0)
                    valid_fraction = float(valid.mean()) if band.size else 0.0

                    DB_MIN, DB_MAX = -25.0, 0.0
                    band_norm = ((band - DB_MIN) / (DB_MAX - DB_MIN) * 255.0)
                    band_norm = np.where(valid, band_norm, 0.0).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(band_norm, mode="L")
                    png_buf = io.BytesIO()
                    pil_img.save(png_buf, format="PNG")
                    scene_bytes = png_buf.getvalue()

                except Exception as raster_exc:
                    logger.warning(
                        "Failed to decode raster for %s: %s — skipping", scene_id, raster_exc
                    )
                    continue

                if valid_fraction < _MIN_SCENE_VALID_FRACTION:
                    logger.warning(
                        "Scene %s (%s, %s) covers only %.1f%% of the AOI — skipping "
                        "(below %.0f%% threshold).",
                        scene_id, acq_date, orbit_pass,
                        valid_fraction * 100, _MIN_SCENE_VALID_FRACTION * 100,
                    )
                    continue

                scenes.append(SARScene(
                    scene_id=scene_id,
                    raster_bytes=scene_bytes,
                    acquisition_date=acq_date,
                    orbit_pass=orbit_pass,
                    bounds=(min_lon, min_lat, max_lon, max_lat),
                    raster_width_px=raster_width_px,
                    raster_height_px=raster_height_px,
                    mock=False,
                    scale_m=used_scale,
                    valid_fraction=valid_fraction,
                ))

            except Exception as scene_exc:
                logger.warning(
                    "Skipping acquisition (%s, %s) due to error: %s",
                    acq_date, orbit_pass, scene_exc, exc_info=True,
                )
                continue

        # Best AOI coverage first, so ml_service tiles the fullest scene before
        # it hits its per-run crop cap.
        scenes.sort(key=lambda s: s.valid_fraction, reverse=True)
        logger.info(
            "Successfully fetched %d real SAR scene(s) from GEE (coverage: %s)",
            len(scenes),
            ", ".join(f"{s.acquisition_date} {s.valid_fraction * 100:.0f}%" for s in scenes) or "none",
        )
        return scenes

    except GEEServiceError:
        raise
    except Exception as exc:
        raise GEEServiceError(f"GEE API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Public async interface
# ---------------------------------------------------------------------------


async def get_sar_tiles(
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    tile_size_px: int = 224,
) -> List[SARScene]:
    """Export full-resolution Sentinel-1 GRD SAR scenes from GEE.

    Filters the COPERNICUS/S1_GRD collection by the supplied AOI,
    date range, VV polarisation, and IW instrument mode. Each matching
    scene is downloaded as a full-resolution GEO_TIFF at scale=40 m/px
    with server-side dB conversion and clipping to [-25, 0] dB.

    When GEE_MOCK_MODE=true, no GEE calls are made and a single synthetic
    512×512 noise scene is returned instead (useful for local development and CI).

    Parameters
    ----------
    aoi_geojson:
        A GeoJSON Polygon (geometry or feature) defining the area of interest.
    start_date:
        Start of the acquisition window in YYYY-MM-DD format.
    end_date:
        End of the acquisition window in YYYY-MM-DD format.
    tile_size_px:
        Retained for API compatibility; not used in the full-resolution path.

    Returns
    -------
    list[SARScene]
        Zero or more SARScene objects. An empty list is returned
        (not an error) when no Sentinel-1 scenes cover the AOI in the
        given date range.

    Raises
    ------
    GEEServiceError
        On GEE authentication failure or unrecoverable API errors.
        NOT raised for empty results.
    """
    if settings.GEE_MOCK_MODE:
        logger.info(
            "GEE_MOCK_MODE is enabled — skipping real GEE call for AOI, "
            "dates %s – %s",
            start_date,
            end_date,
        )
        return _get_mock_sar_tiles(aoi_geojson)

    logger.info(
        "Fetching SAR scenes from GEE for dates %s – %s", start_date, end_date
    )

    scenes = await asyncio.to_thread(
        _fetch_sar_tiles_sync,
        aoi_geojson,
        start_date,
        end_date,
        tile_size_px,
    )
    return scenes


# ---------------------------------------------------------------------------
# SAR preview: AOI-clipped thumbnail for map display
# ---------------------------------------------------------------------------


def _get_sar_preview_sync(
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    display_size: int = 512,
) -> bytes | None:
    """Fetch a display-quality SAR thumbnail clipped to the AOI (sync, runs in thread).

    Uses GEE ``getThumbURL`` to fetch a small PNG of the median-composite
    Sentinel-1 VV image over the AOI and date range.  This is cheap in quota
    terms (thumbnail, not full-res download) and is used only for map display.

    Returns raw PNG bytes, or None if no scenes exist or GEE fails.
    """
    try:
        import ee  # noqa: PLC0415
    except ImportError:
        return None

    try:
        _authenticate_gee()
        min_lon, min_lat, max_lon, max_lat = _extract_aoi_bbox(aoi_geojson)
        aoi_geometry = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)

        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi_geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select("VV")
        )

        # Median composite clipped to AOI — reduces stripe artefacts
        composite = collection.median().clip(aoi_geometry)

        thumb_url = composite.getThumbURL({
            "dimensions": display_size,
            "region": aoi_geometry,
            "format": "png",
            "min": -22,
            "max": -3,
        })

        png_bytes = _fetch_url_bytes(thumb_url)

        # Validate it's actually a PNG (not an HTML error)
        if not png_bytes or png_bytes[:4] != b"\x89PNG":
            logger.warning("SAR preview: received non-PNG response (%d bytes)", len(png_bytes) if png_bytes else 0)
            return None

        logger.info("SAR preview: downloaded %d-byte PNG thumbnail", len(png_bytes))
        return png_bytes

    except Exception as exc:
        logger.warning("SAR preview fetch failed: %s", exc)
        return None


async def get_sar_preview(
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    display_size: int = 512,
) -> bytes | None:
    """Return a display-quality SAR PNG thumbnail for map overlay.

    Fetches a median-composite Sentinel-1 VV thumbnail clipped to the AOI.
    Intended for the frontend SAR raster layer — much cheaper than the
    full-resolution download used for ML inference.

    Returns ``None`` in mock mode or if GEE fails (callers should handle gracefully).
    """
    if settings.GEE_MOCK_MODE:
        # Return the mock noise raster as the preview
        noise = _make_noise_tif(display_size)
        return noise  # it's a TIFF but PIL/browser can handle it

    return await asyncio.to_thread(
        _get_sar_preview_sync,
        aoi_geojson,
        start_date,
        end_date,
        display_size,
    )
