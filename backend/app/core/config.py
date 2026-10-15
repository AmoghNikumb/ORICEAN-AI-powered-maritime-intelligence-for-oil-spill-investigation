"""
Application configuration module.

Loads all settings from the backend/.env file using pydantic-settings.
Required variables must be present at startup or the process will exit.
"""

import logging
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Central configuration for the ORICEAN FastAPI backend.

    Required fields are validated at startup; missing values will cause
    the application to log a clear error and exit with a non-zero status
    code before accepting any HTTP connections.

    Optional fields carry documented defaults that reflect sensible
    development/CPU-only behaviour and can be overridden in backend/.env.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ------------------------------------------------------------------
    # Required — no defaults; must be present in the environment / .env
    # ------------------------------------------------------------------

    DATABASE_URL: str
    """PostgreSQL connection string (e.g. postgresql+asyncpg://...)."""

    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    """Absolute path to the GEE service-account JSON key file."""

    GEE_PROJECT_ID: str = ""
    """Google Cloud / Earth Engine project ID (e.g. 'oricean1')."""

    MODEL_PATH: str = ""
    """Absolute (or repo-relative) path to the .pth model weights file."""

    MODEL_CONFIG_PATH: str = ""
    """Absolute (or repo-relative) path to the model's config.json (architecture,
    channels, normalization, threshold). Empty = use built-in defaults."""

    HF_TOKEN: str = ""
    """Hugging Face access token used to (re)download the model weights when
    MODEL_PATH is missing. Empty = rely on the weights already present on disk."""

    GEOTIFF_EXPORT_DIR: str = ""
    """Directory where GeoTIFF exports from GEE are written."""

    # ------------------------------------------------------------------
    # Optional — documented defaults
    # ------------------------------------------------------------------

    TILE_STRIDE: int = 112
    """Pixel stride used when generating overlapping 224×224 tiles (default 112)."""

    INFERENCE_BATCH_SIZE: int = 16
    """Number of tiles per ML inference batch (default 16)."""

    GRADCAM_MIN_COMPONENT_AREA: int = 50
    """Minimum connected-component area (px²) to retain in Grad-CAM output (default 50)."""

    INFERENCE_DEVICE: str = "cpu"
    """PyTorch device string: 'cpu', 'cuda', or 'mps' (default 'cpu')."""

    ML_MOCK_MODE: bool = False
    """When True, skip real model inference and return synthetic predictions (default False)."""

    GEE_MOCK_MODE: bool = False
    """When True, skip real GEE calls and return synthetic SAR tiles (default False)."""

    FRONTEND_ORIGIN: str = "*"
    """CORS allowed origin for the React frontend (default '*')."""

    # ------------------------------------------------------------------
    # AIS provider — optional; empty string disables AIS queries
    # ------------------------------------------------------------------

    AIS_API_KEY: str = ""
    """API key for the external AIS data provider (default '' — disables AIS queries)."""

    AIS_PROVIDER: str = ""
    """AIS provider name: 'gfw' (Global Fishing Watch historical AIS), 'marinecadastre'
    (NOAA historical, US waters), 'aisstream', or 'spire' (default '' — disables AIS queries)."""

    # ------------------------------------------------------------------
    # Global Fishing Watch (GFW) — historical AIS vessel activity
    # ------------------------------------------------------------------

    GFW_API_TOKEN: str = ""
    """Global Fishing Watch API access token (Bearer). Get one at
    https://globalfishingwatch.org/our-apis/tokens. Empty disables the 'gfw' provider."""

    GFW_API_BASE_URL: str = "https://gateway.api.globalfishingwatch.org"
    """Base URL for the GFW API gateway (v3 endpoints are under /v3)."""

    GFW_ACTIVITY_DATASETS: str = "public-global-presence:latest"
    """Comma-separated GFW 4wings dataset ids queried for vessel activity in the
    spill region. 'public-global-presence:latest' covers all AIS-broadcasting
    vessels (tankers, cargo, fishing); 'public-global-fishing-effort:latest'
    restricts to apparent fishing effort."""

    GFW_TEMPORAL_RESOLUTION: str = "HOURLY"
    """GFW 4wings report temporal bin size: HOURLY | DAILY. HOURLY yields a finer
    pseudo-track for distance/trajectory scoring."""

    GFW_LOOKBACK_HOURS: int = 48
    """Hours to look back from the satellite detection time when searching GFW for
    candidate source vessels (configurable; default 48)."""

    GFW_MAX_VESSELS: int = 60
    """Maximum number of vessels to retain from a GFW activity report (highest
    activity first) before spatial/temporal filtering."""

    GFW_HTTP_TIMEOUT_S: float = 60.0
    """Per-request timeout (seconds) for GFW API calls."""

    AIS_SEARCH_PADDING_DAYS: int = 3
    """Days to extend the investigation's entered date range on each side when
    searching AIS for candidate vessels (default 3). The vessel search window is
    [start_date - N, end_date + N]."""

    AIS_CACHE_DIR: str = ""
    """Directory for cached AIS downloads (marinecadastre daily files). Empty =
    default to backend/.ais_cache."""

    AIS_MARINECADASTRE_MAX_DAYS: int = 3
    """Max number of daily NOAA AIS files to download per attribution run
    (each file is ~150-500 MB). The first N days of the search window are used."""

    # ------------------------------------------------------------------
    # Source reconstruction / drift backtracking
    # ------------------------------------------------------------------

    DRIFT_LOOKBACK_HOURS: int = 48
    """Number of hours to backtrack when computing the source corridor (default 48)."""

    DRIFT_SPEED_M_S: float = 0.3
    """Assumed drift speed in metres per second used for kinematic backtracking (default 0.3)."""

    # ------------------------------------------------------------------
    # Vessel attribution
    # ------------------------------------------------------------------

    AIS_BUFFER_KM: float = 20.0
    """Spatial buffer in kilometres applied around the search region when querying AIS (default 20.0)."""

    ATTRIBUTION_WEIGHTS: str = "0.35,0.30,0.25,0.10"
    """Comma-separated weights for the four rule-based candidate sub-scores, in order:
    spatial proximity, temporal consistency, trajectory overlap, time spent near region.
    Must parse to exactly four floats summing to 1.0 ±0.01; falls back to defaults on failure
    (default '0.35,0.30,0.25,0.10')."""

    # ------------------------------------------------------------------
    # Vessel attribution — candidate qualification gates
    # ------------------------------------------------------------------
    # GFW retrieval and the 4-factor explainable score are unchanged. These
    # gates decide which scored vessels are "qualified" (shown by default) vs
    # merely scored (visible under "show all"). A vessel is only hidden when it
    # fails a measurable gate — never to hit an arbitrary count.

    ATTRIBUTION_TOP_N: int = 8
    """Number of top-ranked qualified candidates surfaced by default in the UI."""

    ATTRIBUTION_MAX_DISTANCE_KM: float = 12.0
    """Gate: a candidate's closest approach to the source corridor must be within
    this many km (default 12.0)."""

    ATTRIBUTION_MAX_TIME_DIFF_HOURS: float = 96.0
    """Gate: the vessel's closest AIS fix must be within this many hours of the
    satellite detection time (default 96.0)."""

    ATTRIBUTION_MIN_TRAJECTORY_OVERLAP: float = 0.05
    """Gate: fraction of the vessel's AIS positions that fall inside the search
    buffer must be at least this — OR the track must actually intersect the
    corridor (closest approach <= 0.5 km) (default 0.05)."""

    ATTRIBUTION_MIN_AIS_POINTS_NEAR_SOURCE: int = 2
    """Gate: number of in-window AIS positions within the search buffer of the
    source corridor must be at least this (default 2)."""

    ATTRIBUTION_MIN_SCORE: float = 12.0
    """Gate: the composite attribution_score (0-100) must be at least this
    (default 12.0)."""

    # ------------------------------------------------------------------
    # Officer alerting — Twilio SMS
    # ------------------------------------------------------------------

    TWILIO_ACCOUNT_SID: str = ""
    """Twilio Account SID (AC…) or API Key SID (SK…). Empty = SMS delivery
    disabled (alerts still recorded)."""

    TWILIO_ACCOUNT_SID_AC: str = ""
    """Account SID (AC…) — only needed when TWILIO_ACCOUNT_SID holds an API Key
    (SK…) and TWILIO_AUTH_TOKEN holds that key's secret."""

    TWILIO_AUTH_TOKEN: str = ""
    """Twilio Auth Token, or the API Key secret when using an SK… key. Never
    exposed to the frontend."""

    TWILIO_FROM_NUMBER: str = ""
    """Twilio sender phone number in E.164 form (e.g. +14155238886)."""

    ALERT_RECIPIENTS: str = ""
    """Comma-separated officer phone numbers (E.164) seeded into the notification
    settings on first run. Editable at runtime via PUT /api/alerts/settings."""

    ALERT_ENABLED: bool = True
    """Master switch for detection alerting (default True). Seeds notification
    settings on first run."""

    ALERT_MIN_CONFIDENCE: float = 0.6
    """Minimum max-tile confidence for a detection to be 'significant' enough to
    alert on (default 0.6). Seeds notification settings on first run."""

    ALERT_MIN_POSITIVE_TILES: int = 3
    """Minimum positive tile count for a detection to trigger an alert
    (default 3). Seeds notification settings on first run."""

    PUBLIC_BASE_URL: str = "http://localhost:8080"
    """Base URL used to build the investigation link included in alert messages."""


try:
    settings = Settings()
except Exception as exc:
    logger.error(
        "Failed to load application settings. "
        "Ensure all required environment variables are set: "
        "DATABASE_URL, GOOGLE_APPLICATION_CREDENTIALS, MODEL_PATH, GEOTIFF_EXPORT_DIR. "
        "Error: %s",
        exc,
    )
    raise SystemExit(1) from exc


_DEFAULT_ATTRIBUTION_WEIGHTS: List[float] = [0.35, 0.30, 0.25, 0.10]


def _parse_attribution_weights(raw: str) -> List[float]:
    """Parse and validate the ATTRIBUTION_WEIGHTS configuration string.

    Expects a comma-separated string of exactly four floats whose values
    sum to 1.0 within a tolerance of ±0.01.  Returns the parsed weights
    on success.  Logs a warning and returns the hard-coded defaults when
    the string is absent, malformed, or fails the sum check.

    Args:
        raw: The raw comma-separated string from ``Settings.ATTRIBUTION_WEIGHTS``.

    Returns:
        A list of four floats ``[w_spatial, w_temporal, w_trajectory, w_ais_anomaly]``.
    """
    try:
        parts = [float(x.strip()) for x in raw.split(",")]
    except (ValueError, AttributeError):
        logger.warning(
            "ATTRIBUTION_WEIGHTS '%s' could not be parsed as floats. "
            "Reverting to defaults %s.",
            raw,
            _DEFAULT_ATTRIBUTION_WEIGHTS,
        )
        return list(_DEFAULT_ATTRIBUTION_WEIGHTS)

    if len(parts) != 4:
        logger.warning(
            "ATTRIBUTION_WEIGHTS '%s' must contain exactly 4 values (got %d). "
            "Reverting to defaults %s.",
            raw,
            len(parts),
            _DEFAULT_ATTRIBUTION_WEIGHTS,
        )
        return list(_DEFAULT_ATTRIBUTION_WEIGHTS)

    if abs(sum(parts) - 1.0) > 0.01:
        logger.warning(
            "ATTRIBUTION_WEIGHTS '%s' values sum to %.4f (expected 1.0 ±0.01). "
            "Reverting to defaults %s.",
            raw,
            sum(parts),
            _DEFAULT_ATTRIBUTION_WEIGHTS,
        )
        return list(_DEFAULT_ATTRIBUTION_WEIGHTS)

    return parts


# Parsed once at module load; import this wherever scored weights are needed.
attribution_weights: List[float] = _parse_attribution_weights(settings.ATTRIBUTION_WEIGHTS)
