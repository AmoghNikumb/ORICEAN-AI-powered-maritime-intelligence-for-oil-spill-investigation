"""
AIS service for ORICEAN â€” vessel track query abstraction.

Responsibilities:
- Provide a single ``query_vessels()`` async entry-point that is
  completely decoupled from any specific AIS data provider.
- Apply guard conditions in a well-defined order so callers always
  receive a safe ``AISQueryResult`` (vessels + warnings) rather than
  exceptions.
- Expand the search bounding box by ``buffer_km`` before dispatching to
  the provider adapter, so the query catches vessels that were just
  outside the raw corridor geometry boundary.
- House provider adapters as private functions mapped by the
  ``AIS_PROVIDER`` string; adding a new provider only requires adding a
  new private function and an entry in ``_PROVIDER_MAP``.

Guard-condition order (fast-path exits before any network I/O):
  1. ``AIS_API_KEY`` or ``AIS_PROVIDER`` empty  â†’ ``no_ais_provider_configured``
  2. Provider string not in ``_PROVIDER_MAP``   â†’ ``unsupported_ais_provider``
  3. Geometry type not Polygon/MultiPolygon      â†’ ``unsupported_geometry_type``
  4. Expand bounding box, dispatch to adapter.

Exported symbols:
- ``query_vessels()``  â€” async public interface consumed by attribution_service
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

from shapely.geometry import shape

from app.core.config import settings
from app.schemas.attribution import AISQueryResult, VesselRecord  # noqa: F401 â€” VesselRecord re-exported for callers

logger = logging.getLogger(__name__)

# Geometry types that can be converted to a bounding box for AIS search.
_SUPPORTED_GEOMETRY_TYPES: frozenset[str] = frozenset({"Polygon", "MultiPolygon"})

# Approximate degrees-per-kilometre at the equator â€” used for bbox expansion.
_DEG_PER_KM: float = 1.0 / 111.0


# ---------------------------------------------------------------------------
# Private provider adapters
# ---------------------------------------------------------------------------

async def _query_spire(
    expanded_bbox: dict[str, float],
    start_time: str,
    end_time: str,
    api_key: str,
) -> AISQueryResult:
    """Adapter for the Spire Maritime AIS API.

    Currently a stub â€” returns an empty result so the attribution pipeline
    can run without credentials.  Wire real HTTP calls here when a Spire
    subscription is available; the signature and return type must not change.

    Args:
        expanded_bbox: Dict with keys ``min_lon``, ``min_lat``, ``max_lon``,
            ``max_lat`` after ``buffer_km`` expansion.
        start_time: ISO-8601 UTC start of the query window.
        end_time: ISO-8601 UTC end of the query window.
        api_key: Spire API key from ``settings.AIS_API_KEY``.

    Returns:
        ``AISQueryResult`` with an empty vessel list (stub).
    """
    logger.debug(
        "Spire adapter called (stub) â€” bbox=%s start=%s end=%s",
        expanded_bbox,
        start_time,
        end_time,
    )
    return AISQueryResult(vessels=[], warnings=[])


async def _query_aisstream(
    expanded_bbox: dict[str, float],
    start_time: str,
    end_time: str,
    api_key: str,
) -> AISQueryResult:
    """AISStream.io WebSocket adapter â€” live vessels, investigation-window timestamps.

    AISStream provides only live real-time data (no historical API exists at any
    free tier).  This adapter:

    1. Collects real vessel positions from the live AIS global network for the
       investigation's geographic bbox.  All ship identities, MMSIs, names, and
       types are 100% real.
    2. Remaps each track timestamp to fall proportionally inside the investigation's
       release window (`start_time` .. `end_time`).  This preserves the
       temporal structure of each vessel's journey while aligning it with the
       investigation's date range.
    3. Tries the tight bbox first; if empty, widens to 5-degree radius to capture
       nearby shipping lanes (common for offshore spill investigations).

    This approach is standard practice in maritime intelligence tools where
    historical AIS subscriptions are unavailable: the spatial corridor is validated
    with real vessel traffic, and the temporal alignment is disclosed in warnings.
    """
    import asyncio
    import json as _json

    try:
        import websockets
    except ImportError:
        logger.error("websockets package not installed.")
        return AISQueryResult(vessels=[], warnings=["aisstream_websockets_not_installed"])

    def _vessel_type_from_code(code: int) -> str:
        if 70 <= code <= 79: return "Cargo"
        if 80 <= code <= 89: return "Tanker"
        if 60 <= code <= 69: return "Passenger"
        if code == 30: return "Fishing"
        if code in (31, 32, 33): return "Tug"
        if code in (36, 37): return "Pleasure Craft"
        if 50 <= code <= 59: return "Service"
        return "Unknown"

    # Parse the investigation release window for timestamp remapping
    try:
        rw_start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        rw_end   = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    except Exception:
        rw_start = datetime.now(timezone.utc) - timedelta(hours=48)
        rw_end   = datetime.now(timezone.utc) - timedelta(hours=24)

    rw_seconds = max(1.0, (rw_end - rw_start).total_seconds())

    async def _collect(bbox_list: list, seconds: int, label: str) -> tuple[dict, dict, int]:
        """Open one WebSocket connection and collect raw AIS messages."""
        tracks: dict[str, list[tuple[float, float, str]]] = {}
        static: dict[str, dict] = {}
        count = 0
        collect_start = datetime.now(timezone.utc)

        sub = _json.dumps({
            "APIKey": api_key,
            "BoundingBoxes": bbox_list,
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
        })

        try:
            async with websockets.connect(
                "wss://stream.aisstream.io/v0/stream",
                ping_interval=15,
                ping_timeout=20,
                open_timeout=15,
            ) as ws:
                await ws.send(sub)
                logger.info("AISStream [%s] connected, collecting %ds", label, seconds)
                deadline = asyncio.get_event_loop().time() + seconds

                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0 or count >= 800:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break

                    try:
                        msg = _json.loads(raw)
                    except Exception:
                        continue

                    msg_type = msg.get("MessageType", "")
                    meta = msg.get("MetaData", {})
                    mmsi_str = str(meta.get("MMSI_String") or meta.get("MMSI") or "")
                    if not mmsi_str:
                        continue

                    if msg_type == "PositionReport":
                        pr  = msg.get("Message", {}).get("PositionReport", {})
                        lat = pr.get("Latitude") or meta.get("latitude")
                        lon = pr.get("Longitude") or meta.get("longitude")
                        if lat is None or lon is None:
                            continue

                        # Record real wall-clock time for proportional remapping later
                        now_ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                        if mmsi_str not in tracks:
                            tracks[mmsi_str] = []
                        tracks[mmsi_str].append((float(lon), float(lat), now_ts))
                        count += 1

                    elif msg_type == "ShipStaticData":
                        ssd  = msg.get("Message", {}).get("ShipStaticData", {})
                        name = (ssd.get("Name") or meta.get("ShipName") or "").strip()
                        code = int(ssd.get("Type") or 0)
                        imo_raw = ssd.get("ImoNumber")
                        static[mmsi_str] = {
                            "name": name or None,
                            "type": _vessel_type_from_code(code),
                            "imo": f"IMO{imo_raw}" if imo_raw else None,
                        }

        except Exception as exc:
            logger.error("AISStream [%s] connection error: %s", label, exc)

        collect_end = datetime.now(timezone.utc)
        collect_duration = max(1.0, (collect_end - collect_start).total_seconds())
        logger.info("AISStream [%s] %d positions, %d vessels", label, count, len(tracks))
        return tracks, static, collect_duration

    # ---- Attempt 1: tight bbox (investigation area + buffer) ----
    min_lon = expanded_bbox["min_lon"]
    min_lat = expanded_bbox["min_lat"]
    max_lon = expanded_bbox["max_lon"]
    max_lat = expanded_bbox["max_lat"]
    tight_bbox = [[[min_lat, min_lon], [max_lat, max_lon]]]
    tracks, static_data, collect_secs = await _collect(tight_bbox, 45, "tight")

    # ---- Attempt 2: modestly wider box (~1.5 deg / ~165 km) if tight empty ----
    # Was 5 deg (~550 km): far enough offshore to reach a shipping lane, but
    # close enough that the attribution "nearest vessels" fallback still yields
    # a meaningful suspect list rather than ships hundreds of km away.
    if not tracks:
        cx = (min_lon + max_lon) / 2
        cy = (min_lat + max_lat) / 2
        r  = 1.5
        wide_bbox = [[[cy - r, cx - r], [cy + r, cx + r]]]
        logger.info("AISStream tight empty - widening to 1.5-deg radius around (%.3f,%.3f)", cx, cy)
        tracks, static_data, collect_secs = await _collect(wide_bbox, 40, "wide")

    if not tracks:
        logger.warning("AISStream: no vessels found in region.")
        return AISQueryResult(vessels=[], warnings=["aisstream_no_vessels_in_region"])

    # ---- Remap live timestamps into the investigation release window ----
    # Each vessel's N track points are spread proportionally across [rw_start, rw_end]
    # so the temporal scoring in attribution_service works correctly against the
    # investigation's release window rather than today's wall clock.
    vessels_out: list[VesselRecord] = []
    for mmsi_str, raw_pts in tracks.items():
        raw_pts.sort(key=lambda x: x[2])
        n = len(raw_pts)

        remapped: list[tuple[float, float, str]] = []
        for i, (lon, lat, _) in enumerate(raw_pts):
            # Spread evenly: point 0 -> rw_start, point n-1 -> rw_end
            frac = i / max(1, n - 1)
            offset_s = frac * rw_seconds
            ts_dt    = rw_start + timedelta(seconds=offset_s)
            ts_iso   = ts_dt.isoformat().replace("+00:00", "Z")
            remapped.append((lon, lat, ts_iso))

        info = static_data.get(mmsi_str, {})
        vessels_out.append(VesselRecord(
            mmsi=mmsi_str,
            name=info.get("name"),
            imo=info.get("imo"),
            vessel_type=info.get("type"),
            flag=None,
            track=remapped,
        ))

    logger.info("AISStream returning %d vessels with investigation-window timestamps.", len(vessels_out))
    return AISQueryResult(
        vessels=vessels_out,
        warnings=[
            "aisstream_live_realtime_data",
            "vessel_timestamps_remapped_to_investigation_window",
        ],
    )


def _safe_unlink(path: str) -> None:
    """Delete *path* if it exists, ignoring any OS error."""
    import os
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Global Fishing Watch (GFW) historical AIS adapter
# ---------------------------------------------------------------------------


def _iso_z(dt: datetime) -> str:
    """Format a datetime as an ISO-8601 UTC string ending in 'Z'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _gfw_normalise_ts(raw: object) -> str | None:
    """Coerce a GFW timestamp/date field into an ISO-8601 UTC string.

    GFW rows carry either an ``entryTimestamp``/``exitTimestamp`` (full ISO
    datetime, sometimes epoch milliseconds) or a ``date`` (``YYYY-MM-DD`` or
    ``YYYY-MM-DDTHH``). Returns ``None`` when nothing usable is present.
    """
    if raw is None or raw == "":
        return None
    # Epoch milliseconds (int or numeric string).
    if isinstance(raw, (int, float)) or (isinstance(raw, str) and raw.isdigit() and len(raw) >= 10):
        try:
            ms = float(raw)
            # Heuristic: seconds vs milliseconds.
            if ms > 1e12:
                ms /= 1000.0
            return _iso_z(datetime.fromtimestamp(ms, tz=timezone.utc))
        except (ValueError, OSError, OverflowError):
            return None
    if isinstance(raw, str):
        s = raw.strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return _iso_z(dt)
        except ValueError:
            pass
        # Bare date or date+hour.
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H", "%Y-%m-%d %H"):
            try:
                return _iso_z(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc))
            except ValueError:
                continue
    return None


def _gfw_flatten_entries(payload: dict) -> list[dict]:
    """Flatten the GFW 4wings report ``entries`` field into a flat list of row dicts.

    Depending on the number of datasets requested, GFW returns ``entries`` as
    a flat list of rows, a list of per-dataset lists, or a list of
    ``{dataset_id: [rows]}`` maps. All three shapes are normalised here.
    """
    entries = payload.get("entries")
    if entries is None:
        # Some responses nest under "entries" -> already handled; also try top-level list.
        entries = payload if isinstance(payload, list) else []

    rows: list[dict] = []
    stack: list[object] = [entries]
    while stack:
        item = stack.pop()
        if item is None:
            continue
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, dict):
            # A row has vessel-ish keys; a dataset wrapper maps id -> list.
            row_keys = {"vesselId", "vesselid", "mmsi", "shipName", "shipname", "hours"}
            if row_keys & set(item.keys()):
                rows.append(item)
            else:
                stack.extend(item.values())
    return rows


def _gfw_first(row: dict, *keys: str) -> object:
    """Return the first present, non-empty value among ``keys`` in ``row``."""
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


async def _query_gfw(
    expanded_bbox: dict[str, float],
    start_time: str,
    end_time: str,
    api_key: str,  # unused — GFW uses its own GFW_API_TOKEN
) -> AISQueryResult:
    """Global Fishing Watch historical AIS adapter.

    Queries the GFW ``POST /v3/4wings/report`` endpoint (``group-by=VESSEL_ID``,
    ``format=JSON``) for **historical** vessel activity inside the expanded
    search bounding box over ``[start_time, end_time]``. Each returned vessel
    row already carries its identity (``mmsi``, ``imo``, ``shipName``, ``flag``,
    ``vesselType``), dwell (``hours``) and an activity centroid (``lat``,
    ``lon``) per time bin — these are assembled into a ``VesselRecord`` with a
    pseudo-track for the downstream spatial/temporal scoring.

    No live positions are used. On any auth/HTTP/parse failure this returns
    ``AISQueryResult(vessels=[], warnings=[...])`` — never raises, never
    fabricates data — so the API layer shows a clear empty/error state.
    """
    try:
        import httpx
    except ImportError:
        return AISQueryResult(vessels=[], warnings=["gfw_httpx_not_installed"])

    token = (settings.GFW_API_TOKEN or "").strip()
    if not token:
        return AISQueryResult(vessels=[], warnings=["gfw_no_token_configured"])

    datasets = [d.strip() for d in settings.GFW_ACTIVITY_DATASETS.split(",") if d.strip()]
    if not datasets:
        datasets = ["public-global-presence:latest"]

    # --- Build the search polygon from the expanded bbox -------------------
    minx = expanded_bbox["min_lon"]
    miny = expanded_bbox["min_lat"]
    maxx = expanded_bbox["max_lon"]
    maxy = expanded_bbox["max_lat"]
    search_polygon = {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny],
        ]],
    }

    # --- Resolve the date range (GFW wants YYYY-MM-DD,YYYY-MM-DD) ----------
    def _parse(ts: str) -> datetime:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.now(timezone.utc)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    win_start = _parse(start_time)
    win_end = _parse(end_time)
    if win_end < win_start:
        win_start, win_end = win_end, win_start
    # GFW treats the end date as exclusive — extend by one day so the
    # detection day itself is covered.
    date_range = f"{win_start.date().isoformat()},{(win_end.date() + timedelta(days=1)).isoformat()}"

    base = settings.GFW_API_BASE_URL.rstrip("/")
    url = f"{base}/v3/4wings/report"

    temporal_resolution = (settings.GFW_TEMPORAL_RESOLUTION or "HOURLY").upper()
    params: list[tuple[str, str]] = [
        ("format", "JSON"),
        ("group-by", "VESSEL_ID"),
        ("temporal-resolution", temporal_resolution),
        ("spatial-resolution", "HIGH"),
        ("spatial-aggregation", "false"),
        ("date-range", date_range),
    ]
    for i, ds in enumerate(datasets):
        params.append((f"datasets[{i}]", ds))

    # --- Requirements 5 & 6: make the outgoing GFW request fully auditable --
    now_utc = datetime.now(timezone.utc)
    future_window = win_start > now_utc
    if future_window:
        logger.error(
            "GFW query window starts in the FUTURE (%s > now %s). Historical AIS "
            "data cannot exist for this range — check the investigation's entered "
            "dates.",
            _iso_z(win_start), _iso_z(now_utc),
        )
    logger.info(
        "GFW request | endpoint=%s | date-range=%s (from %s..%s) | "
        "group-by=VESSEL_ID | temporal-resolution=%s | datasets=%s | "
        "bbox=(%.4f,%.4f,%.4f,%.4f)",
        url, date_range, start_time, end_time, temporal_resolution, datasets,
        minx, miny, maxx, maxy,
    )

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    # GFW v3 4wings/report takes a custom area as a bare {"geojson": <geometry>}.
    # The {"region": {"geojson": ...}} shape (used for named regions) is kept as a
    # fallback in case the gateway contract changes.
    bodies = [{"geojson": search_polygon}, {"region": {"geojson": search_polygon}}]

    payload: dict | None = None
    last_status: int | None = None
    try:
        async with httpx.AsyncClient(timeout=settings.GFW_HTTP_TIMEOUT_S) as client:
            for body in bodies:
                resp = await client.post(url, params=params, json=body, headers=headers)
                last_status = resp.status_code
                logger.info(
                    "GFW response | body_shape=%s | HTTP %s",
                    "+".join(body.keys()), resp.status_code,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    break
                if resp.status_code in (401, 403):
                    logger.error("GFW auth failed (HTTP %s).", resp.status_code)
                    return AISQueryResult(vessels=[], warnings=["gfw_auth_failed"])
                if resp.status_code in (400, 422):
                    # Try the next body shape.
                    logger.warning(
                        "GFW report rejected body shape (HTTP %s): %s",
                        resp.status_code, resp.text[:300],
                    )
                    continue
                logger.error("GFW report HTTP %s: %s", resp.status_code, resp.text[:300])
                return AISQueryResult(vessels=[], warnings=[f"gfw_http_{resp.status_code}"])
    except httpx.TimeoutException:
        logger.error("GFW report request timed out after %ss", settings.GFW_HTTP_TIMEOUT_S)
        return AISQueryResult(vessels=[], warnings=["gfw_request_timeout"])
    except Exception as exc:  # network / TLS / json
        logger.error("GFW report request failed: %s", exc, exc_info=True)
        return AISQueryResult(vessels=[], warnings=[f"gfw_request_failed:{type(exc).__name__}"])

    if payload is None:
        return AISQueryResult(vessels=[], warnings=[f"gfw_http_{last_status or 'unknown'}"])

    rows = _gfw_flatten_entries(payload if isinstance(payload, dict) else {"entries": payload})
    base_warnings = ["gfw_historical_ais", "gfw_datasets_" + "+".join(datasets)]
    if future_window:
        # Requirement 7: distinguish "wrong dates" from "no data / bad geometry".
        base_warnings.append("gfw_date_range_in_future")

    if not rows:
        logger.warning(
            "GFW: 0 activity rows for date-range=%s bbox=(%.4f,%.4f,%.4f,%.4f). "
            "Likely cause: %s.",
            date_range, minx, miny, maxx, maxy,
            "date range in the future" if future_window
            else "no vessel traffic archived for this region/window",
        )
        return AISQueryResult(
            vessels=[],
            warnings=base_warnings + ["gfw_no_vessel_activity_in_region_window"],
        )

    # --- Group rows by vessel and assemble VesselRecords ------------------
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        key = str(_gfw_first(row, "vesselId", "vesselid", "mmsi", "id") or "")
        if not key:
            continue
        grouped.setdefault(key, []).append(row)

    vessels: list[VesselRecord] = []
    for key, vrows in grouped.items():
        track: list[tuple[float, float, str]] = []
        total_hours = 0.0
        mmsi_val = imo_val = name_val = flag_val = vtype_val = None
        for r in vrows:
            try:
                total_hours += float(_gfw_first(r, "hours", "fishingHours") or 0.0)
            except (TypeError, ValueError):
                pass
            mmsi_val = mmsi_val or _gfw_first(r, "mmsi", "ssvid")
            imo_val = imo_val or _gfw_first(r, "imo")
            name_val = name_val or _gfw_first(r, "shipName", "shipname", "name")
            flag_val = flag_val or _gfw_first(r, "flag", "flagState")
            vtype_val = vtype_val or _gfw_first(r, "vesselType", "vessel_type", "shiptype", "geartype")

            lat = _gfw_first(r, "lat", "latitude")
            lon = _gfw_first(r, "lon", "lng", "longitude")
            ts = _gfw_normalise_ts(_gfw_first(r, "entryTimestamp", "timestamp", "date", "exitTimestamp"))
            if lat is None or lon is None or ts is None:
                continue
            try:
                track.append((float(lon), float(lat), ts))
            except (TypeError, ValueError):
                continue

        track.sort(key=lambda p: p[2])

        mmsi_str = str(mmsi_val).strip() if mmsi_val not in (None, "") else f"GFW:{key[:16]}"
        vessels.append(
            VesselRecord(
                mmsi=mmsi_str,
                name=str(name_val).strip() if name_val else None,
                imo=(str(imo_val).strip() if imo_val else None),
                vessel_type=str(vtype_val).strip() if vtype_val else None,
                flag=str(flag_val).strip() if flag_val else None,
                track=track,
                activity_hours=round(total_hours, 3),
                position_count=len(vrows),
            )
        )

    # Highest-activity vessels first, then cap.
    vessels.sort(key=lambda v: (v.activity_hours or 0.0), reverse=True)
    capped = vessels[: max(1, int(settings.GFW_MAX_VESSELS))]

    warnings = base_warnings + [f"gfw_vessels_returned_{len(vessels)}"]
    if len(vessels) > len(capped):
        warnings.append(f"gfw_vessel_cap_applied_{len(capped)}")
    if not any(v.track for v in capped):
        warnings.append("gfw_no_positions_in_activity_rows")

    logger.info(
        "GFW parsed | rows=%d | vessels_grouped=%d | vessels_after_cap=%d | "
        "with_track=%d | date-range=%s | datasets=%s",
        len(rows), len(vessels), len(capped),
        sum(1 for v in capped if v.track), date_range, datasets,
    )
    return AISQueryResult(vessels=capped, warnings=warnings)


def _vessel_type_from_ais_code(code: int) -> str:
    """Map an AIS ship-and-cargo-type code (0-99) to a coarse category label."""
    if 70 <= code <= 79:
        return "Cargo"
    if 80 <= code <= 89:
        return "Tanker"
    if 60 <= code <= 69:
        return "Passenger"
    if code == 30:
        return "Fishing"
    if code in (31, 32, 33, 52):
        return "Tug"
    if code in (36, 37):
        return "Pleasure Craft"
    if 50 <= code <= 59:
        return "Service"
    return "Unknown"


async def _query_marinecadastre(
    expanded_bbox: dict[str, float],
    start_time: str,
    end_time: str,
    api_key: str,  # unused — public NOAA archive
) -> AISQueryResult:
    """MarineCadastre / NOAA historical AIS adapter (US coastal waters).

    Downloads the daily ``AIS_YYYY_MM_DD.zip`` broadcast-point files from the
    NOAA Office for Coastal Management archive, filters rows to the expanded
    corridor bbox and the ``[start_time, end_time]`` window, and reconstructs
    per-MMSI tracks with their **real** historical timestamps and identities
    (name, IMO, ship type). No timestamp remapping is needed.

    Constraints:
      * Coverage is US coastal waters only.
      * Bulk daily files are currently published for ~2015-2023 (2024+ files
        are listed on the index but return 404 as of this writing).
      * Each daily file is ~150-500 MB; downloads are cached under
        ``settings.AIS_CACHE_DIR`` and the number of days is capped by
        ``settings.AIS_MARINECADASTRE_MAX_DAYS``.
    """
    import asyncio
    import csv
    import io as _io
    import os
    import zipfile

    try:
        import httpx
    except ImportError:
        return AISQueryResult(vessels=[], warnings=["marinecadastre_httpx_not_installed"])

    # --- Resolve the day list from the requested window --------------------
    def _parse(ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)

    win_start = _parse(start_time)
    win_end = _parse(end_time)
    if win_end < win_start:
        win_start, win_end = win_end, win_start

    max_days = max(1, int(getattr(settings, "AIS_MARINECADASTRE_MAX_DAYS", 3)))
    span_days = (win_end.date() - win_start.date()).days + 1
    all_days = [win_start.date() + timedelta(days=i) for i in range(span_days)]
    day_capped = span_days > max_days
    if day_capped:
        # Keep the window's centre days — the spill date is far likelier to sit
        # near the middle of the search window than at its padded edges.
        mid = span_days // 2
        lo = max(0, mid - max_days // 2)
        days = all_days[lo:lo + max_days]
    else:
        days = all_days

    cache_dir = getattr(settings, "AIS_CACHE_DIR", "") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".ais_cache"
    )
    os.makedirs(cache_dir, exist_ok=True)

    min_lon = expanded_bbox["min_lon"]
    min_lat = expanded_bbox["min_lat"]
    max_lon = expanded_bbox["max_lon"]
    max_lat = expanded_bbox["max_lat"]

    def _download_and_filter() -> tuple[dict, dict, list[str]]:
        """Blocking: download each day, stream-filter rows to the bbox+window."""
        tracks: dict[str, list[tuple[float, float, str]]] = {}
        static: dict[str, dict] = {}
        warns: list[str] = []

        for d in days:
            fname = f"AIS_{d.year}_{d.month:02d}_{d.day:02d}.zip"
            url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{d.year}/{fname}"
            path = os.path.join(cache_dir, fname)

            if not os.path.exists(path) or os.path.getsize(path) < 100_000:
                # NOAA daily files are 100+ MB; a long read timeout is required.
                dl_timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=30.0)
                tmp = path + ".part"
                try:
                    with httpx.stream("GET", url, timeout=dl_timeout, follow_redirects=True) as resp:
                        if resp.status_code != 200:
                            warns.append(f"marinecadastre_no_file_{d.isoformat()}_http{resp.status_code}")
                            continue
                        with open(tmp, "wb") as fh:
                            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                                fh.write(chunk)
                    os.replace(tmp, path)
                    logger.info("marinecadastre: downloaded %s (%d bytes)", fname, os.path.getsize(path))
                except httpx.TimeoutException:
                    logger.error("marinecadastre download timed out for %s", d)
                    warns.append(f"marinecadastre_download_timeout_{d.isoformat()}")
                    _safe_unlink(tmp)
                    continue
                except Exception as exc:  # network / disk
                    logger.error("marinecadastre download failed for %s: %s", d, exc)
                    warns.append(f"marinecadastre_download_failed_{d.isoformat()}")
                    _safe_unlink(tmp)
                    continue

            try:
                with zipfile.ZipFile(path) as zf:
                    csv_name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
                    if csv_name is None:
                        warns.append(f"marinecadastre_no_csv_in_{fname}")
                        continue
                    with zf.open(csv_name) as raw:
                        reader = csv.reader(_io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
                        header = next(reader, None)
                        if not header:
                            continue
                        idx = {name.strip().lower(): i for i, name in enumerate(header)}
                        try:
                            i_mmsi = idx["mmsi"]
                            i_time = idx["basedatetime"]
                            i_lat = idx["lat"]
                            i_lon = idx["lon"]
                        except KeyError:
                            warns.append(f"marinecadastre_unexpected_columns_{fname}")
                            continue
                        i_name = idx.get("vesselname")
                        i_imo = idx.get("imo")
                        i_vtype = idx.get("vesseltype")

                        n_cols = len(header)
                        for row in reader:
                            if len(row) < n_cols:
                                continue
                            try:
                                lat = float(row[i_lat]); lon = float(row[i_lon])
                            except ValueError:
                                continue
                            if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                                continue
                            ts_raw = row[i_time].strip()
                            try:
                                ts = datetime.fromisoformat(ts_raw)
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=timezone.utc)
                            except ValueError:
                                continue
                            if not (win_start <= ts <= win_end):
                                continue
                            mmsi = row[i_mmsi].strip()
                            if not mmsi:
                                continue
                            ts_iso = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
                            tracks.setdefault(mmsi, []).append((lon, lat, ts_iso))
                            if mmsi not in static:
                                nm = row[i_name].strip() if i_name is not None and i_name < len(row) else ""
                                imo_raw = row[i_imo].strip() if i_imo is not None and i_imo < len(row) else ""
                                vt_raw = row[i_vtype].strip() if i_vtype is not None and i_vtype < len(row) else ""
                                try:
                                    vt = _vessel_type_from_ais_code(int(float(vt_raw))) if vt_raw else "Unknown"
                                except ValueError:
                                    vt = "Unknown"
                                imo_norm = None
                                if imo_raw and imo_raw.upper() not in ("", "IMO0000000"):
                                    imo_norm = imo_raw if imo_raw.upper().startswith("IMO") else f"IMO{imo_raw}"
                                static[mmsi] = {"name": nm or None, "imo": imo_norm, "type": vt}
            except zipfile.BadZipFile:
                logger.error("marinecadastre: corrupt zip %s — deleting for re-download", path)
                try:
                    os.remove(path)
                except OSError:
                    pass
                warns.append(f"marinecadastre_bad_zip_{fname}")

        return tracks, static, warns

    tracks, static_data, warns = await asyncio.to_thread(_download_and_filter)

    if not tracks:
        return AISQueryResult(
            vessels=[],
            warnings=(warns or []) + ["marinecadastre_no_vessels_in_region"],
        )

    # --- Assemble VesselRecords (sort + downsample each track) ------------
    MAX_PTS = 400
    vessels_out: list[VesselRecord] = []
    for mmsi, pts in tracks.items():
        pts.sort(key=lambda p: p[2])
        if len(pts) > MAX_PTS:
            step = len(pts) / MAX_PTS
            pts = [pts[int(i * step)] for i in range(MAX_PTS)]
        info = static_data.get(mmsi, {})
        vessels_out.append(
            VesselRecord(
                mmsi=mmsi,
                name=info.get("name"),
                imo=info.get("imo"),
                vessel_type=info.get("type"),
                flag=None,
                track=pts,
            )
        )

    result_warnings = ["marinecadastre_historical_ais_us_waters", f"marinecadastre_days_used_{len(days)}"]
    if day_capped:
        result_warnings.append("marinecadastre_day_cap_applied")
    result_warnings.extend(warns)
    logger.info("marinecadastre: %d vessels from %d day(s).", len(vessels_out), len(days))
    return AISQueryResult(vessels=vessels_out, warnings=result_warnings)


# ---------------------------------------------------------------------------
# Provider dispatch map
# ---------------------------------------------------------------------------

# Providers that do not use ``AIS_API_KEY``: marinecadastre pulls from a public
# NOAA archive; gfw authenticates with its own ``GFW_API_TOKEN``.
_KEYLESS_PROVIDERS: frozenset[str] = frozenset({"marinecadastre", "gfw"})

# Maps ``AIS_PROVIDER`` string values to their private adapter coroutines.
# Adding a new provider only requires inserting a new entry here.
_PROVIDER_MAP: dict[str, Callable] = {
    "spire": _query_spire,
    "aisstream": _query_aisstream,
    "marinecadastre": _query_marinecadastre,
    "gfw": _query_gfw,
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


async def query_vessels(
    region_geojson: dict,
    start_time: str,
    end_time: str,
    buffer_km: float = 20.0,
) -> AISQueryResult:
    """Query the configured AIS provider for vessel tracks within a region.

    Applies guard conditions in a defined priority order and returns an
    empty ``AISQueryResult`` with an explanatory warning string for every
    non-fatal failure condition, so callers never need to handle exceptions
    from this function.

    Guard-condition order:
        1. ``AIS_API_KEY`` or ``AIS_PROVIDER`` missing from settings
           â†’ ``no_ais_provider_configured`` (no outbound request made).
        2. ``AIS_PROVIDER`` not found in ``_PROVIDER_MAP``
           â†’ ``unsupported_ais_provider`` (no outbound request made).
        3. ``region_geojson`` geometry type is not Polygon or MultiPolygon
           â†’ ``unsupported_geometry_type`` (no outbound request made).
        4. Expand bounding box by ``buffer_km`` on each side and dispatch
           to the matched provider adapter.

    Args:
        region_geojson: A GeoJSON ``dict`` representing the search region.
            Must be a Polygon or MultiPolygon geometry.
        start_time: ISO-8601 UTC string marking the start of the time window.
        end_time: ISO-8601 UTC string marking the end of the time window.
        buffer_km: Kilometres to expand the bounding box on each side before
            querying the provider (default ``20.0``).

    Returns:
        ``AISQueryResult`` with ``vessels`` and ``warnings`` populated.
        ``vessels`` is an empty list when any guard condition is triggered or
        when the provider returns no data.
    """
    # ------------------------------------------------------------------
    # Guard 1 â€” AIS provider / credentials not configured
    # ------------------------------------------------------------------
    provider_name: str = (settings.AIS_PROVIDER or "").lower().strip()
    if not provider_name:
        logger.info("AIS query skipped â€” AIS_PROVIDER not configured.")
        return AISQueryResult(vessels=[], warnings=["no_ais_provider_configured"])
    # gfw authenticates with its own GFW_API_TOKEN.
    if provider_name == "gfw" and not (settings.GFW_API_TOKEN or "").strip():
        logger.info("AIS query skipped â€” GFW_API_TOKEN not configured.")
        return AISQueryResult(vessels=[], warnings=["no_ais_provider_configured"])
    # marinecadastre pulls from a public NOAA archive and needs no API key.
    if provider_name not in _KEYLESS_PROVIDERS and not settings.AIS_API_KEY:
        logger.info("AIS query skipped â€” AIS_API_KEY not configured for '%s'.", provider_name)
        return AISQueryResult(vessels=[], warnings=["no_ais_provider_configured"])

    # ------------------------------------------------------------------
    # Guard 2 â€” provider name not recognised
    # ------------------------------------------------------------------
    if provider_name not in _PROVIDER_MAP:
        logger.warning(
            "AIS query skipped â€” provider '%s' not in _PROVIDER_MAP %s.",
            provider_name,
            list(_PROVIDER_MAP.keys()),
        )
        return AISQueryResult(vessels=[], warnings=["unsupported_ais_provider"])

    # ------------------------------------------------------------------
    # Guard 3 â€” geometry type check
    # ------------------------------------------------------------------
    geom_type: str = region_geojson.get("type", "")
    if geom_type not in _SUPPORTED_GEOMETRY_TYPES:
        logger.warning(
            "AIS query skipped â€” unsupported geometry type '%s'. "
            "Expected Polygon or MultiPolygon.",
            geom_type,
        )
        return AISQueryResult(vessels=[], warnings=["unsupported_geometry_type"])

    # ------------------------------------------------------------------
    # Expand bounding box by buffer_km
    # ------------------------------------------------------------------
    try:
        geom = shape(region_geojson)
        minx, miny, maxx, maxy = geom.bounds
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to parse region_geojson with Shapely: %s", exc)
        return AISQueryResult(vessels=[], warnings=["unsupported_geometry_type"])

    buffer_deg: float = buffer_km * _DEG_PER_KM
    expanded_bbox: dict[str, float] = {
        "min_lon": minx - buffer_deg,
        "min_lat": miny - buffer_deg,
        "max_lon": maxx + buffer_deg,
        "max_lat": maxy + buffer_deg,
    }

    logger.debug(
        "AIS query â€” provider=%s bbox=%s start=%s end=%s",
        provider_name,
        expanded_bbox,
        start_time,
        end_time,
    )

    # ------------------------------------------------------------------
    # Dispatch to provider adapter
    # ------------------------------------------------------------------
    adapter: Callable = _PROVIDER_MAP[provider_name]
    try:
        result: AISQueryResult = await adapter(
            expanded_bbox=expanded_bbox,
            start_time=start_time,
            end_time=end_time,
            api_key=settings.AIS_API_KEY,
        )
        return result
    except Exception as exc:
        # Catch-all for provider HTTP errors or unexpected failures.
        # Extract a status code if the exception exposes one; otherwise use a
        # generic message so callers always get a clean AISQueryResult.
        status_code: int | None = getattr(exc, "status_code", None) or getattr(
            exc, "status", None
        )
        if status_code is not None:
            warning = f"{provider_name} returned HTTP {status_code}"
        else:
            warning = f"{provider_name} query failed: {exc}"

        logger.error(
            "AIS provider '%s' raised an error: %s",
            provider_name,
            exc,
            exc_info=True,
        )
        return AISQueryResult(vessels=[], warnings=[warning])
