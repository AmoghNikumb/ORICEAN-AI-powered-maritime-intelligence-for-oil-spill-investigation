/**
 * OriceanMapLibre.tsx
 *
 * Core MapLibre GL JS map component for ORICEAN.
 * Replaces the SVG MapView with a full interactive map supporting:
 *   - Dark / light / satellite basemaps
 *   - AOI polygon layer
 *   - SAR imagery raster layer
 *   - Tile grid GeoJSON layer with hover tooltip
 *   - Spill detection polygon layer
 *   - Grad-CAM overlay raster layer (per positive tile)
 *   - Tile click → onTileClick callback
 *   - Coordinate overlay, zoom controls, basemap switcher
 *
 * Requirements: 1.1–1.11, 3.1, 3.6, 4.1–4.7, 5.1, 10.4–10.6, 13.6
 */

import "maplibre-gl/dist/maplibre-gl.css";
import maplibregl from "maplibre-gl";
import type { Map as MapLibreMap, MapMouseEvent } from "maplibre-gl";
import { useEffect, useRef, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import type {
  GeoJSONPolygon,
  InferenceResult,
  TilePrediction,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface LayerVisibility {
  sar: boolean;
  tileGrid: boolean;
  detections: boolean;
  gradcam: boolean;
  ais: boolean;
  drift?: boolean;
}

export interface ReplayFrame {
  t: string;
  lon: number;
  lat: number;
}

export interface OriceanMapLibreProps {
  aoi: GeoJSONPolygon | null;
  inferenceResult?: InferenceResult | null;
  sourceCorridorGeojson?: object | null;
  aisVesselTracks?: Array<{ mmsi: string; name?: string | null; track: Array<[number, number]> }> | null;
  /** MMSI of the #1 ranked candidate — its track is drawn prominently and labelled. */
  highlightMmsi?: string | null;
  /** Simulated spill-drift replay: dashed path + a hollow "SIMULATED" marker at `frames[activeIndex]`. */
  driftReplay?: { frames: ReplayFrame[]; activeIndex: number; direction?: "forward" | "backward" } | null;
  /** Ship-route replay: the #1 vessel's real AIS path + a solid red marker at `frames[activeIndex]`. */
  shipReplay?: { frames: ReplayFrame[]; activeIndex: number; label?: string } | null;
  mode: "aoi-select" | "results" | "simple";
  onTileClick?: (tile: TilePrediction) => void;
  displayThreshold?: number;
  visibleLayers?: LayerVisibility;
  className?: string;
  /** When true (with mode="aoi-select"), click the map to draw an AOI polygon. */
  drawEnabled?: boolean;
  /** Fires with the drawn polygon on Finish, or null on Clear. */
  onPolygonDrawn?: (polygon: GeoJSONPolygon | null) => void;
}

export type TileState =
  | "PENDING"
  | "PROCESSING"
  | "ANALYSED_NO_OIL"
  | "OIL_DETECTED"
  | "ERROR";

// ---------------------------------------------------------------------------
// Basemap URLs
// ---------------------------------------------------------------------------

const BASEMAPS = {
  dark: "https://tiles.openfreemap.org/styles/dark",
  light: "https://tiles.openfreemap.org/styles/bright",
  satellite: "https://tiles.openfreemap.org/styles/positron",
} as const;

type BasemapKey = keyof typeof BASEMAPS;

// ---------------------------------------------------------------------------
// Pure helper functions (exported for unit tests — Task 10)
// ---------------------------------------------------------------------------

/**
 * Derive the TileState for a tile given the current display threshold.
 *
 * Priority order (as per spec):
 *  1. OIL_DETECTED  — is_positive AND confidence >= threshold
 *  2. ANALYSED_NO_OIL — is_positive is false
 *  3. PENDING         — acquisition_date is absent (null / undefined / "")
 *  4. ERROR           — all other states
 *
 * Validates: Requirements 4.2
 */
export function getTileState(tile: TilePrediction, threshold: number): TileState {
  if (tile.is_positive && tile.confidence >= threshold) return "OIL_DETECTED";
  if (!tile.is_positive) return "ANALYSED_NO_OIL";
  if (!tile.acquisition_date) return "PENDING";
  return "ERROR";
}

/**
 * Map each TileState to a CSS colour string.
 * Every branch returns a non-empty string — no state is unhandled.
 * Validates: Requirements 4.2, Property 3
 */
export function tileStateToColor(state: TileState): string {
  switch (state) {
    case "PENDING":
      return "rgba(128,128,128,0.3)";
    case "PROCESSING":
      return "var(--satellite, #3b82f6)";
    case "ANALYSED_NO_OIL":
      return "var(--resolved, #22c55e)";
    case "OIL_DETECTED":
      return "var(--spill, #f97316)";
    case "ERROR":
      return "var(--critical, #ef4444)";
    default: {
      // exhaustive — TypeScript will warn if a case is missed
      const _: never = state;
      void _;
      return "rgba(128,128,128,0.3)";
    }
  }
}

/** Map TileState to a concrete hex colour for MapLibre paint expressions. */
function tileStateToHex(state: TileState): string {
  switch (state) {
    case "PENDING":        return "#808080";
    case "PROCESSING":     return "#3b82f6";
    case "ANALYSED_NO_OIL":return "#22c55e";
    case "OIL_DETECTED":   return "#f97316";
    case "ERROR":          return "#ef4444";
  }
}

/** Opacity per TileState */
function tileStateToOpacity(state: TileState): number {
  switch (state) {
    case "PENDING":         return 0.3;
    case "PROCESSING":      return 0.45;
    case "ANALYSED_NO_OIL": return 0.35;
    case "OIL_DETECTED":    return 0.55;
    case "ERROR":           return 0.5;
  }
}

/**
 * Build a GeoJSON FeatureCollection from an array of TilePredictions.
 * Validates: Requirements 4.1, Property 2
 */
export function buildTileGeoJSON(
  tiles: TilePrediction[],
  threshold: number,
): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = tiles.map((tile) => {
    const [min_lon, min_lat, max_lon, max_lat] = tile.bounds;
    const state = getTileState(tile, threshold);
    return {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
          ],
        ],
      } as GeoJSON.Polygon,
      properties: {
        tile_id: tile.tile_id,
        state,
        label: tile.label,
        confidence: tile.confidence,
        is_positive: tile.is_positive,
        acquisition_date: tile.acquisition_date,
        fill_color: tileStateToColor(state),
        fill_opacity: state === "OIL_DETECTED" ? 0.55 : 0.3,
      },
    };
  });
  return { type: "FeatureCollection", features };
}

/**
 * Reclassify tiles against a new threshold without re-running inference.
 * Validates: Requirements 9.2, 9.3, 9.5, Property 10
 */
export function reclassifyTiles(
  tiles: TilePrediction[],
  threshold: number,
): TilePrediction[] {
  return tiles.map((tile) => ({
    ...tile,
    is_positive: tile.confidence >= threshold,
    label: tile.confidence >= threshold ? "Oil Spill" : "No Oil",
  }));
}

/**
 * Extract MapLibre LngLatBoundsLike from a GeoJSON Polygon AOI.
 * Validates: Requirements 1.7, 2.3, Property 1
 */
export function fitBoundsForAoi(aoi: GeoJSONPolygon): maplibregl.LngLatBoundsLike {
  const ring = aoi.coordinates[0];
  const lons = ring.map(([lon]) => lon);
  const lats = ring.map(([, lat]) => lat);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  return [[minLon, minLat], [maxLon, maxLat]];
}

// ---------------------------------------------------------------------------
// Tooltip state type
// ---------------------------------------------------------------------------

interface TooltipState {
  x: number;
  y: number;
  tileId: string;
  state: TileState;
  label: string;
  confidence: number;
}

// ---------------------------------------------------------------------------
// Helper: safely add a source (guards against React StrictMode double-invoke)
// ---------------------------------------------------------------------------

function safeAddSource(
  map: MapLibreMap,
  id: string,
  source: maplibregl.SourceSpecification,
) {
  if (!map.getSource(id)) {
    map.addSource(id, source);
  }
}

function safeAddLayer(
  map: MapLibreMap,
  layer: maplibregl.LayerSpecification,
  beforeId?: string,
) {
  if (!map.getLayer(layer.id)) {
    try {
      map.addLayer(layer, beforeId);
    } catch {
      // layer may have been added between the check and the call
    }
  }
}

// ---------------------------------------------------------------------------
// OriceanMapLibre component
// ---------------------------------------------------------------------------

const DEFAULT_VISIBLE_LAYERS: LayerVisibility = {
  sar: true,
  tileGrid: true,
  detections: true,
  gradcam: true,
  ais: true,
  drift: true,
};

/**
 * Validate a hand-drawn AOI polygon's vertex list (pure — exported for tests).
 * Mirrors the bounding-box rules used elsewhere in the app.
 * Returns an error string, or null when the polygon is acceptable.
 */
export function validateDrawnPolygon(points: Array<[number, number]>): string | null {
  const distinct = points.filter(
    (p, i) => i === 0 || p[0] !== points[i - 1][0] || p[1] !== points[i - 1][1],
  );
  if (distinct.length < 3) return "Draw at least 3 points to form an area.";
  const lons = distinct.map((p) => p[0]);
  const lats = distinct.map((p) => p[1]);
  const lonSpan = Math.max(...lons) - Math.min(...lons);
  const latSpan = Math.max(...lats) - Math.min(...lats);
  if (lonSpan < 1e-4 || latSpan < 1e-4) return "The drawn area is too small.";
  if (lonSpan > 10 || latSpan > 10) return "Area too large — keep it under 10° × 10°.";
  return null;
}

export function OriceanMapLibre({
  aoi,
  inferenceResult,
  sourceCorridorGeojson,
  aisVesselTracks,
  highlightMmsi,
  driftReplay,
  shipReplay,
  mode,
  onTileClick,
  displayThreshold = 0.5,
  visibleLayers = DEFAULT_VISIBLE_LAYERS,
  className,
  drawEnabled = false,
  onPolygonDrawn,
}: OriceanMapLibreProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);

  // UI state
  const [coords, setCoords] = useState<[number, number] | null>(null);
  const [basemap, setBasemap] = useState<BasemapKey>("dark");
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  // Draw-mode state
  const drawPtsRef = useRef<Array<[number, number]>>([]);
  const [drawCount, setDrawCount] = useState(0);
  const [drawFinished, setDrawFinished] = useState(false);

  // Track which gradcam layer IDs have been added so we can clean them up
  const gradcamLayerIds = useRef<string[]>([]);

  // -------------------------------------------------------------------------
  // 9.1 — Initialise the MapLibre map instance
  // -------------------------------------------------------------------------

  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapRef.current) return; // already initialised

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: BASEMAPS.dark,
      center: [20, 15],
      zoom: 2,
      attributionControl: false,
    });

    mapRef.current = map;

    map.on("load", () => setMapLoaded(true));

    // Keep the GL canvas sized to its container. MapLibre only measures the
    // container on init; when the map lives in a flex/grid cell whose height
    // resolves *after* mount, the canvas stays 0px and renders blank until a
    // resize is forced. A ResizeObserver covers every later layout change.
    const ro = new ResizeObserver(() => {
      mapRef.current?.resize();
    });
    ro.observe(mapContainerRef.current);

    return () => {
      ro.disconnect();
      map.remove();
      mapRef.current = null;
      setMapLoaded(false);
    };
    // Only run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // -------------------------------------------------------------------------
  // 9.2 — Basemap switcher
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    // setStyle will reload all sources/layers, so we reset mapLoaded briefly
    setMapLoaded(false);
    map.setStyle(BASEMAPS[basemap]);
    map.once("style.load", () => setMapLoaded(true));
  }, [basemap]); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // 9.2 — Coordinate overlay (mousemove)
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const onMouseMove = (e: MapMouseEvent) => {
      setCoords([
        Math.round(e.lngLat.lng * 100000) / 100000,
        Math.round(e.lngLat.lat * 100000) / 100000,
      ]);
    };
    const onMouseLeave = () => setCoords(null);

    map.on("mousemove", onMouseMove);
    map.on("mouseout", onMouseLeave);

    return () => {
      map.off("mousemove", onMouseMove);
      map.off("mouseout", onMouseLeave);
    };
  }, [mapLoaded]);

  // -------------------------------------------------------------------------
  // 9.3 — AOI polygon layer
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !aoi) return;

    const geojson: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: aoi, properties: {} }],
    };

    // Update or add the source
    const existing = map.getSource("aoi") as maplibregl.GeoJSONSource | undefined;
    if (existing) {
      existing.setData(geojson);
    } else {
      safeAddSource(map, "aoi", { type: "geojson", data: geojson });
    }

    safeAddLayer(map, {
      id: "aoi-fill",
      type: "fill",
      source: "aoi",
      paint: {
        "fill-color": "#38bdf8",
        "fill-opacity": 0.12,
      },
    });

    safeAddLayer(map, {
      id: "aoi-outline",
      type: "line",
      source: "aoi",
      paint: {
        "line-color": "#38bdf8",
        "line-width": 2,
        "line-opacity": 0.9,
      },
    });

    // Fit bounds to AOI
    try {
      map.fitBounds(fitBoundsForAoi(aoi), { padding: 60, duration: 800 });
    } catch {
      // ignore invalid bounds
    }
  }, [aoi, mapLoaded]);

  // -------------------------------------------------------------------------
  // Draw mode — click to place AOI polygon vertices
  // -------------------------------------------------------------------------

  const renderDraw = useCallback(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    const pts = drawPtsRef.current;
    const lineCoords = drawFinished && pts.length >= 3 ? [...pts, pts[0]] : pts;
    const setFC = (id: string, features: GeoJSON.Feature[]) => {
      const fc: GeoJSON.FeatureCollection = { type: "FeatureCollection", features };
      const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
      if (src) src.setData(fc);
      else safeAddSource(map, id, { type: "geojson", data: fc });
    };
    setFC("draw-line", lineCoords.length >= 2
      ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: lineCoords } }]
      : []);
    setFC("draw-fill", drawFinished && pts.length >= 3
      ? [{ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...pts, pts[0]]] } }]
      : []);
    setFC("draw-verts", pts.map((p, i) => ({
      type: "Feature", properties: { i }, geometry: { type: "Point", coordinates: p },
    })));
    safeAddLayer(map, { id: "draw-fill-layer", type: "fill", source: "draw-fill",
      paint: { "fill-color": "#38bdf8", "fill-opacity": 0.12 } });
    safeAddLayer(map, { id: "draw-line-layer", type: "line", source: "draw-line",
      paint: { "line-color": "#38bdf8", "line-width": 2, "line-dasharray": [2, 1] } });
    safeAddLayer(map, { id: "draw-verts-layer", type: "circle", source: "draw-verts",
      paint: { "circle-radius": 4, "circle-color": "#38bdf8",
        "circle-stroke-color": "#0f172a", "circle-stroke-width": 1.5 } });
  }, [drawFinished]);

  const emitPolygon = useCallback(() => {
    const pts = drawPtsRef.current;
    if (pts.length < 3) return;
    onPolygonDrawn?.({ type: "Polygon", coordinates: [[...pts, pts[0]]] });
  }, [onPolygonDrawn]);

  const clearDraw = useCallback(() => {
    drawPtsRef.current = [];
    setDrawCount(0);
    setDrawFinished(false);
    onPolygonDrawn?.(null);
    renderDraw();
  }, [onPolygonDrawn, renderDraw]);

  const finishDraw = useCallback(() => {
    if (drawPtsRef.current.length < 3) return;
    setDrawFinished(true);
    emitPolygon();
  }, [emitPolygon]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    if (!drawEnabled || mode !== "aoi-select") return;

    map.getCanvas().style.cursor = "crosshair";
    const onClick = (e: MapMouseEvent) => {
      if (drawFinished) {
        drawPtsRef.current = [];
        setDrawFinished(false);
      }
      drawPtsRef.current = [...drawPtsRef.current, [e.lngLat.lng, e.lngLat.lat]];
      setDrawCount(drawPtsRef.current.length);
      renderDraw();
    };
    const onDblClick = (e: MapMouseEvent) => {
      e.preventDefault();
      finishDraw();
    };
    map.on("click", onClick);
    map.on("dblclick", onDblClick);
    map.doubleClickZoom.disable();
    renderDraw();

    return () => {
      map.off("click", onClick);
      map.off("dblclick", onDblClick);
      map.doubleClickZoom.enable();
      map.getCanvas().style.cursor = "";
    };
  }, [drawEnabled, mode, mapLoaded, drawFinished, renderDraw, finishDraw]);

  // -------------------------------------------------------------------------
  // 9.4 — SAR imagery raster layer
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !inferenceResult?.sar_scene_url || !aoi) return;

    const bounds = fitBoundsForAoi(aoi) as [[number, number], [number, number]];
    const [sw, ne] = bounds;
    // MapLibre image source coordinates: [NW, NE, SE, SW]
    const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [
      [sw[0], ne[1]], // NW
      [ne[0], ne[1]], // NE
      [ne[0], sw[1]], // SE
      [sw[0], sw[1]], // SW
    ];

    const existing = map.getSource("sar-image");
    if (existing) {
      // ImageSource doesn't have setData — remove and re-add
      if (map.getLayer("sar-layer")) map.removeLayer("sar-layer");
      map.removeSource("sar-image");
    }

    safeAddSource(map, "sar-image", {
      type: "image",
      url: inferenceResult.sar_scene_url,
      coordinates,
    });

    // Add SAR layer below tile grid (use "aoi-fill" as beforeId if it exists)
    const beforeId = map.getLayer("tile-fill") ? "tile-fill" : map.getLayer("aoi-fill") ? "aoi-fill" : undefined;

    safeAddLayer(
      map,
      {
        id: "sar-layer",
        type: "raster",
        source: "sar-image",
        paint: { "raster-opacity": 0.8 },
      },
      beforeId,
    );
  }, [inferenceResult?.sar_scene_url, aoi, mapLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // Sync SAR layer visibility
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    if (map.getLayer("sar-layer")) {
      map.setLayoutProperty("sar-layer", "visibility", visibleLayers.sar ? "visible" : "none");
    }
  }, [visibleLayers.sar, mapLoaded]);

  // -------------------------------------------------------------------------
  // 9.5 — Tile grid GeoJSON layer
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    if (!inferenceResult?.tile_results?.length) return;

    const geojson = buildTileGeoJSON(inferenceResult.tile_results, displayThreshold);

    const existing = map.getSource("tile-grid") as maplibregl.GeoJSONSource | undefined;
    if (existing) {
      existing.setData(geojson);
    } else {
      safeAddSource(map, "tile-grid", { type: "geojson", data: geojson });
    }

    safeAddLayer(map, {
      id: "tile-fill",
      type: "fill",
      source: "tile-grid",
      paint: {
        "fill-color": ["get", "fill_color"],
        "fill-opacity": ["get", "fill_opacity"],
      },
    });

    safeAddLayer(map, {
      id: "tile-outline",
      type: "line",
      source: "tile-grid",
      filter: ["==", ["get", "state"], "OIL_DETECTED"],
      paint: {
        "line-color": "#f97316",
        "line-width": 1.5,
        "line-opacity": 0.9,
      },
    });
  }, [inferenceResult, displayThreshold, mapLoaded]);

  // Sync tile grid visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    const vis = visibleLayers.tileGrid ? "visible" : "none";
    if (map.getLayer("tile-fill")) map.setLayoutProperty("tile-fill", "visibility", vis);
    if (map.getLayer("tile-outline")) map.setLayoutProperty("tile-outline", "visibility", vis);
  }, [visibleLayers.tileGrid, mapLoaded]);

  // -------------------------------------------------------------------------
  // 9.5 — Hover tooltip on tile-fill layer
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const onEnter = (e: maplibregl.MapLayerMouseEvent) => {
      map.getCanvas().style.cursor = "pointer";
      const feature = e.features?.[0];
      if (!feature) return;
      const p = feature.properties as {
        tile_id: string;
        state: TileState;
        label: string;
        confidence: number;
      };
      setTooltip({
        x: e.point.x,
        y: e.point.y,
        tileId: p.tile_id,
        state: p.state,
        label: p.label,
        confidence: p.confidence,
      });
    };

    const onMove = (e: maplibregl.MapLayerMouseEvent) => {
      const feature = e.features?.[0];
      if (!feature) return;
      const p = feature.properties as {
        tile_id: string;
        state: TileState;
        label: string;
        confidence: number;
      };
      setTooltip({
        x: e.point.x,
        y: e.point.y,
        tileId: p.tile_id,
        state: p.state,
        label: p.label,
        confidence: p.confidence,
      });
    };

    const onLeave = () => {
      map.getCanvas().style.cursor = "";
      setTooltip(null);
    };

    map.on("mouseenter", "tile-fill", onEnter);
    map.on("mousemove", "tile-fill", onMove);
    map.on("mouseleave", "tile-fill", onLeave);

    return () => {
      map.off("mouseenter", "tile-fill", onEnter);
      map.off("mousemove", "tile-fill", onMove);
      map.off("mouseleave", "tile-fill", onLeave);
    };
  }, [mapLoaded]);

  // -------------------------------------------------------------------------
  // 9.6 — Spill detection polygon layer
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    if (!inferenceResult?.spill_detections?.length) return;

    const features: GeoJSON.Feature[] = inferenceResult.spill_detections.map((det) => ({
      type: "Feature",
      geometry: det.polygon_geojson as GeoJSON.Geometry,
      properties: {
        detection_id: det.detection_id,
        mean_confidence: det.mean_confidence,
      },
    }));

    const geojson: GeoJSON.FeatureCollection = { type: "FeatureCollection", features };

    const existing = map.getSource("spill-detections") as maplibregl.GeoJSONSource | undefined;
    if (existing) {
      existing.setData(geojson);
    } else {
      safeAddSource(map, "spill-detections", { type: "geojson", data: geojson });
    }

    safeAddLayer(map, {
      id: "spill-fill",
      type: "fill",
      source: "spill-detections",
      paint: {
        "fill-color": "#dc2626",
        "fill-opacity": 0.3,
      },
    });

    safeAddLayer(map, {
      id: "spill-outline",
      type: "line",
      source: "spill-detections",
      paint: {
        "line-color": "#dc2626",
        "line-width": 2,
        "line-opacity": 0.85,
      },
    });
  }, [inferenceResult?.spill_detections, mapLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync spill visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    const vis = visibleLayers.detections ? "visible" : "none";
    if (map.getLayer("spill-fill")) map.setLayoutProperty("spill-fill", "visibility", vis);
    if (map.getLayer("spill-outline")) map.setLayoutProperty("spill-outline", "visibility", vis);
  }, [visibleLayers.detections, mapLoaded]);

  // -------------------------------------------------------------------------
  // Source corridor + AIS tracks layers
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !sourceCorridorGeojson) return;
    const geojson: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: [{ type: "Feature", geometry: sourceCorridorGeojson as GeoJSON.Geometry, properties: {} }],
    };
    const existingSrc = map.getSource("source-corridor") as maplibregl.GeoJSONSource | undefined;
    if (existingSrc) { existingSrc.setData(geojson); }
    else { safeAddSource(map, "source-corridor", { type: "geojson", data: geojson }); }
    safeAddLayer(map, { id: "source-corridor-fill", type: "fill", source: "source-corridor",
      paint: { "fill-color": "#38bdf8", "fill-opacity": 0.15 } });
    safeAddLayer(map, { id: "source-corridor-outline", type: "line", source: "source-corridor",
      paint: { "line-color": "#38bdf8", "line-width": 2, "line-dasharray": [4, 3], "line-opacity": 0.8 } });
    try {
      const poly = sourceCorridorGeojson as { type: string; coordinates: number[][][] };
      if (poly.type === "Polygon" && poly.coordinates?.[0]?.length) {
        const c = poly.coordinates[0];
        map.fitBounds([[Math.min(...c.map(p=>p[0])), Math.min(...c.map(p=>p[1]))],
                       [Math.max(...c.map(p=>p[0])), Math.max(...c.map(p=>p[1]))]], { padding: 60, duration: 800 });
      }
    } catch { /* ignore */ }
  }, [sourceCorridorGeojson, mapLoaded]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const setFC = (id: string, features: GeoJSON.Feature[]) => {
      const fc: GeoJSON.FeatureCollection = { type: "FeatureCollection", features };
      const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
      if (src) src.setData(fc);
      else safeAddSource(map, id, { type: "geojson", data: fc });
    };

    const tracks = aisVesselTracks ?? [];
    const isTop = (mmsi: string) => !!highlightMmsi && mmsi === highlightMmsi;

    // --- Other candidates — de-emphasised grey lines + small end dots ---
    const others = tracks.filter((v) => !isTop(v.mmsi));
    setFC(
      "ais-tracks",
      others.filter((v) => v.track.length >= 2).map((v) => ({
        type: "Feature", properties: { mmsi: v.mmsi, name: v.name ?? v.mmsi },
        geometry: { type: "LineString", coordinates: v.track },
      })),
    );
    setFC(
      "ais-dots",
      others.filter((v) => v.track.length >= 1).map((v) => ({
        type: "Feature", properties: { mmsi: v.mmsi, name: v.name ?? v.mmsi },
        geometry: { type: "Point", coordinates: v.track[v.track.length - 1] },
      })),
    );
    safeAddLayer(map, { id: "ais-tracks-line", type: "line", source: "ais-tracks",
      paint: { "line-color": "#93c5fd", "line-width": 1.4, "line-opacity": 0.6 } });
    safeAddLayer(map, { id: "ais-dots-circle", type: "circle", source: "ais-dots",
      paint: { "circle-radius": 3.5, "circle-color": "#93c5fd", "circle-opacity": 0.8,
        "circle-stroke-color": "#0f172a", "circle-stroke-width": 1 } });

    // --- Top candidate — prominent RED path, endpoints, and label ---
    const top = tracks.find((v) => isTop(v.mmsi));
    const topLine: GeoJSON.Feature[] =
      top && top.track.length >= 2
        ? [{ type: "Feature", properties: { name: top.name ?? top.mmsi },
            geometry: { type: "LineString", coordinates: top.track } }]
        : [];
    const topPts: GeoJSON.Feature[] = top && top.track.length >= 1
      ? [
          { type: "Feature", properties: { role: "start" },
            geometry: { type: "Point", coordinates: top.track[0] } },
          { type: "Feature", properties: { role: "end", label: "Top Candidate Vessel" },
            geometry: { type: "Point", coordinates: top.track[top.track.length - 1] } },
        ]
      : [];
    setFC("ais-top-line", topLine);
    setFC("ais-top-pts", topPts);
    safeAddLayer(map, { id: "ais-top-casing", type: "line", source: "ais-top-line",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#0f172a", "line-width": 6, "line-opacity": 0.9 } });
    safeAddLayer(map, { id: "ais-top-line-layer", type: "line", source: "ais-top-line",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ef4444", "line-width": 4, "line-opacity": 1 } });
    safeAddLayer(map, { id: "ais-top-pts-circle", type: "circle", source: "ais-top-pts",
      paint: {
        "circle-radius": ["match", ["get", "role"], "end", 7, 4],
        "circle-color": "#ef4444",
        "circle-stroke-color": "#0f172a", "circle-stroke-width": 2,
      } });
    safeAddLayer(map, { id: "ais-top-label", type: "symbol", source: "ais-top-pts",
      filter: ["==", ["get", "role"], "end"],
      layout: {
        "text-field": ["get", "label"], "text-size": 11, "text-offset": [0, 1.4],
        "text-anchor": "top", "text-font": ["Noto Sans Bold", "Open Sans Bold"],
      },
      paint: { "text-color": "#fca5a5", "text-halo-color": "#0f172a", "text-halo-width": 1.6 } });
  }, [aisVesselTracks, highlightMmsi, mapLoaded]);

  useEffect(() => {
    const map = mapRef.current; if (!map || !mapLoaded) return;
    const vis = visibleLayers.ais ? "visible" : "none";
    for (const id of [
      "ais-tracks-line", "ais-dots-circle",
      "ais-top-casing", "ais-top-line-layer", "ais-top-pts-circle", "ais-top-label",
      "ship-replay-line", "ship-replay-marker", "ship-replay-label",
    ]) {
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis);
    }
  }, [visibleLayers.ais, mapLoaded]);

  // -------------------------------------------------------------------------
  // Drift replay (simulated) + Ship-route replay (real AIS) — time-slider markers
  // -------------------------------------------------------------------------
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const setFC = (id: string, features: GeoJSON.Feature[]) => {
      const fc: GeoJSON.FeatureCollection = { type: "FeatureCollection", features };
      const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
      if (src) src.setData(fc);
      else safeAddSource(map, id, { type: "geojson", data: fc });
    };
    const line = (coords: number[][]): GeoJSON.Feature[] =>
      coords.length >= 2
        ? [{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: coords } }]
        : [];

    // --- Drift replay: dashed amber path + hollow "SIMULATED" marker ---
    const df = driftReplay?.frames ?? [];
    setFC("drift-replay-line", line(df.map((f) => [f.lon, f.lat])));
    const dCur = df[Math.min(driftReplay?.activeIndex ?? 0, Math.max(0, df.length - 1))];
    setFC(
      "drift-replay-marker",
      dCur
        ? [{ type: "Feature", properties: { label: `SIMULATED · ${dCur.t.slice(0, 16).replace("T", " ")}` },
            geometry: { type: "Point", coordinates: [dCur.lon, dCur.lat] } }]
        : [],
    );
    safeAddLayer(map, { id: "drift-replay-line", type: "line", source: "drift-replay-line",
      paint: { "line-color": "#f59e0b", "line-width": 2, "line-dasharray": [2, 2], "line-opacity": 0.85 } });
    safeAddLayer(map, { id: "drift-replay-marker", type: "circle", source: "drift-replay-marker",
      paint: {
        "circle-radius": 9, "circle-color": "#f59e0b", "circle-opacity": 0.18,
        "circle-stroke-color": "#f59e0b", "circle-stroke-width": 2.5,
      } });
    safeAddLayer(map, { id: "drift-replay-label", type: "symbol", source: "drift-replay-marker",
      layout: {
        "text-field": ["get", "label"], "text-size": 10, "text-offset": [0, 1.5],
        "text-anchor": "top", "text-font": ["Noto Sans Bold", "Open Sans Bold"],
      },
      paint: { "text-color": "#fbbf24", "text-halo-color": "#0f172a", "text-halo-width": 1.6 } });

    // --- Ship replay: full red path + solid red marker at the active frame ---
    const sf = shipReplay?.frames ?? [];
    setFC("ship-replay-line", line(sf.map((f) => [f.lon, f.lat])));
    const sCur = sf[Math.min(shipReplay?.activeIndex ?? 0, Math.max(0, sf.length - 1))];
    setFC(
      "ship-replay-marker",
      sCur
        ? [{ type: "Feature",
            properties: { label: shipReplay?.label ?? "Top Potential Source Vessel" },
            geometry: { type: "Point", coordinates: [sCur.lon, sCur.lat] } }]
        : [],
    );
    safeAddLayer(map, { id: "ship-replay-line", type: "line", source: "ship-replay-line",
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": "#ef4444", "line-width": 2.5, "line-opacity": 0.7 } });
    safeAddLayer(map, { id: "ship-replay-marker", type: "circle", source: "ship-replay-marker",
      paint: {
        "circle-radius": 7, "circle-color": "#ef4444",
        "circle-stroke-color": "#0f172a", "circle-stroke-width": 2.5,
      } });
    safeAddLayer(map, { id: "ship-replay-label", type: "symbol", source: "ship-replay-marker",
      layout: {
        "text-field": ["get", "label"], "text-size": 10, "text-offset": [0, 1.4],
        "text-anchor": "top", "text-font": ["Noto Sans Bold", "Open Sans Bold"],
      },
      paint: { "text-color": "#fca5a5", "text-halo-color": "#0f172a", "text-halo-width": 1.6 } });
  }, [driftReplay, shipReplay, mapLoaded]);

  useEffect(() => {
    const map = mapRef.current; if (!map || !mapLoaded) return;
    const vis = (visibleLayers.drift ?? true) ? "visible" : "none";
    for (const id of ["drift-replay-line", "drift-replay-marker", "drift-replay-label"]) {
      if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", vis);
    }
  }, [visibleLayers.drift, mapLoaded]);

  // 9.7 — Grad-CAM overlay raster layers (one per positive tile)
  // -------------------------------------------------------------------------

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !inferenceResult?.tile_results || !aoi) return;

    const positiveTiles = inferenceResult.tile_results.filter(
      (t) => t.is_positive && t.gradcam_b64,
    );

    // Remove old gradcam layers/sources that are no longer in the result
    const newIds = new Set(positiveTiles.map((t) => t.tile_id));
    const toRemove = gradcamLayerIds.current.filter((id) => !newIds.has(id));
    for (const id of toRemove) {
      if (map.getLayer(`gradcam-layer-${id}`)) map.removeLayer(`gradcam-layer-${id}`);
      if (map.getSource(`gradcam-${id}`)) map.removeSource(`gradcam-${id}`);
    }

    const addedIds: string[] = [];

    for (const tile of positiveTiles) {
      if (!tile.gradcam_b64) continue;
      const [min_lon, min_lat, max_lon, max_lat] = tile.bounds;
      const coordinates: [[number, number], [number, number], [number, number], [number, number]] = [
        [min_lon, max_lat], // NW
        [max_lon, max_lat], // NE
        [max_lon, min_lat], // SE
        [min_lon, min_lat], // SW
      ];

      const srcId = `gradcam-${tile.tile_id}`;
      const lyrId = `gradcam-layer-${tile.tile_id}`;

      const dataUrl = tile.gradcam_b64.startsWith("data:")
        ? tile.gradcam_b64
        : `data:image/png;base64,${tile.gradcam_b64}`;

      if (map.getSource(srcId)) {
        // Already exists, skip
      } else {
        safeAddSource(map, srcId, {
          type: "image",
          url: dataUrl,
          coordinates,
        });
      }

      safeAddLayer(map, {
        id: lyrId,
        type: "raster",
        source: srcId,
        paint: { "raster-opacity": 0.65 },
      });

      addedIds.push(tile.tile_id);
    }

    gradcamLayerIds.current = addedIds;
  }, [inferenceResult?.tile_results, mapLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync Grad-CAM visibility
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;
    const vis = visibleLayers.gradcam ? "visible" : "none";
    for (const id of gradcamLayerIds.current) {
      const lyrId = `gradcam-layer-${id}`;
      if (map.getLayer(lyrId)) map.setLayoutProperty(lyrId, "visibility", vis);
    }
  }, [visibleLayers.gradcam, mapLoaded]);

  // -------------------------------------------------------------------------
  // 9.8 — onTileClick callback
  // -------------------------------------------------------------------------

  const onTileClickRef = useRef(onTileClick);
  onTileClickRef.current = onTileClick;

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded) return;

    const onClick = (e: maplibregl.MapLayerMouseEvent) => {
      const feature = e.features?.[0];
      if (!feature) return;
      const tileId = (feature.properties as { tile_id: string }).tile_id;
      const tile = inferenceResult?.tile_results?.find((t) => t.tile_id === tileId);
      if (tile && onTileClickRef.current) {
        onTileClickRef.current(tile);
      }
    };

    map.on("click", "tile-fill", onClick);
    return () => {
      map.off("click", "tile-fill", onClick);
    };
  }, [mapLoaded, inferenceResult]); // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // Zoom / Reset helpers
  // -------------------------------------------------------------------------

  const handleZoomIn = useCallback(() => mapRef.current?.zoomIn(), []);
  const handleZoomOut = useCallback(() => mapRef.current?.zoomOut(), []);
  const handleReset = useCallback(() => {
    const map = mapRef.current;
    if (!map || !aoi) return;
    try {
      map.fitBounds(fitBoundsForAoi(aoi), { padding: 60, duration: 800 });
    } catch {
      map.flyTo({ center: [20, 15], zoom: 2 });
    }
  }, [aoi]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className={cn("relative h-full w-full overflow-hidden", className)}>
      {/* Map container — explicit min-height so the GL canvas never collapses
          to 0px when a parent supplies only min-height (not a resolved height). */}
      <div ref={mapContainerRef} className="h-full w-full" style={{ minHeight: 260 }} />

      {/* Zoom + Reset controls */}
      <div className="absolute right-3 top-3 z-10 flex flex-col overflow-hidden rounded border border-white/15 bg-black/70 shadow-lg backdrop-blur-sm">
        <button
          type="button"
          aria-label="Zoom in"
          onClick={handleZoomIn}
          className="px-2.5 py-1.5 text-sm font-bold text-white/80 transition-colors hover:bg-white/10 hover:text-white"
        >
          +
        </button>
        <button
          type="button"
          aria-label="Zoom out"
          onClick={handleZoomOut}
          className="border-t border-white/10 px-2.5 py-1.5 text-sm font-bold text-white/80 transition-colors hover:bg-white/10 hover:text-white"
        >
          −
        </button>
        <button
          type="button"
          aria-label="Reset to AOI"
          onClick={handleReset}
          className="border-t border-white/10 px-2.5 py-1 text-[9px] tracking-wider text-white/50 transition-colors hover:bg-white/10 hover:text-white/80"
        >
          RST
        </button>
      </div>

      {/* Basemap switcher */}
      <div className="absolute left-3 top-3 z-10 flex flex-col gap-1">
        {(Object.keys(BASEMAPS) as BasemapKey[]).map((key) => (
          <button
            key={key}
            type="button"
            onClick={() => setBasemap(key)}
            className={cn(
              "rounded border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide backdrop-blur-sm transition-colors",
              basemap === key
                ? "border-white/40 bg-white/20 text-white"
                : "border-white/15 bg-black/50 text-white/50 hover:bg-white/10 hover:text-white/80",
            )}
          >
            {key}
          </button>
        ))}
      </div>

      {/* Coordinate overlay */}
      {coords && (
        <div className="pointer-events-none absolute bottom-8 left-3 z-10 rounded border border-white/15 bg-black/70 px-2 py-0.5 font-mono text-[10px] text-white/70 backdrop-blur-sm">
          {coords[0].toFixed(5)}° {coords[1].toFixed(5)}°
        </div>
      )}

      {/* Hover tooltip */}
      {tooltip && (
        <div
          className="pointer-events-none absolute z-20 min-w-[160px] rounded border border-white/15 bg-black/85 px-3 py-2 text-xs text-white/90 shadow-xl backdrop-blur-sm"
          style={{
            left: tooltip.x + 12,
            top: tooltip.y - 10,
          }}
        >
          <p className="font-mono text-[10px] text-white/50">{tooltip.tileId}</p>
          <p className="mt-0.5 font-semibold">{tooltip.label}</p>
          <p className="text-white/60">{tooltip.state}</p>
          <p className="mt-1 text-white/80">
            <span className="text-white/50">Prob: </span>
            {(tooltip.confidence * 100).toFixed(1)}%
          </p>
        </div>
      )}

      {/* "mode: results" indicator badge — subtle */}
      {mode === "aoi-select" && !drawEnabled && (
        <div className="pointer-events-none absolute bottom-3 right-3 z-10 rounded border border-satellite/30 bg-black/70 px-2 py-0.5 text-[10px] text-satellite/80 backdrop-blur-sm">
          AOI selection
        </div>
      )}

      {/* Draw-mode controls */}
      {drawEnabled && mode === "aoi-select" && (
        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-1.5 rounded border border-white/15 bg-black/75 px-2 py-1.5 text-[11px] text-white/80 backdrop-blur-sm">
          <span className="mr-1 font-medium">
            {drawFinished
              ? `Polygon · ${drawCount} pts`
              : drawCount === 0
                ? "Click the map to add points"
                : `${drawCount} pt${drawCount === 1 ? "" : "s"} — dbl-click to finish`}
          </span>
          <button
            type="button"
            onClick={finishDraw}
            disabled={drawCount < 3 || drawFinished}
            className="rounded border border-white/20 px-2 py-0.5 font-medium text-white transition-colors hover:bg-white/15 disabled:opacity-40"
          >
            Finish
          </button>
          <button
            type="button"
            onClick={clearDraw}
            disabled={drawCount === 0}
            className="rounded border border-white/20 px-2 py-0.5 font-medium text-white transition-colors hover:bg-white/15 disabled:opacity-40"
          >
            {drawFinished ? "Redraw" : "Clear"}
          </button>
        </div>
      )}
    </div>
  );
}

export default OriceanMapLibre;
