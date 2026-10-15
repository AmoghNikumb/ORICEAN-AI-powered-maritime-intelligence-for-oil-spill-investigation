/**
 * Time-slider replay control — shared by the drift replay (simulated spill
 * movement) and the ship-route replay (#1 candidate vessel's real AIS track).
 *
 * The bar is fully controlled: the parent owns `activeIndex` + `playing` and
 * reads `activeIndex` to position the map marker. When `playing`, the bar
 * advances the index every ~700 ms via `onIndex`.
 */

import { useEffect, type ReactNode } from "react";
import { ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { formatDateTime } from "@/lib/oricean-data";

export interface ReplayStep {
  t: string; // ISO-8601 UTC
  lon: number;
  lat: number;
}

// ---------------------------------------------------------------------------
// Pure helpers (also unit-testable)
// ---------------------------------------------------------------------------

/** Centroid [lon, lat] of a GeoJSON Polygon / MultiPolygon exterior ring(s). */
export function polygonCentroid(geom: unknown): [number, number] | null {
  try {
    const g = geom as { type?: string; coordinates?: number[][][] | number[][][][] };
    let ring: number[][] | undefined;
    if (g.type === "Polygon") ring = (g.coordinates as number[][][])[0];
    else if (g.type === "MultiPolygon") ring = (g.coordinates as number[][][][])[0][0];
    if (!ring || ring.length === 0) return null;
    const xs = ring.map((p) => p[0]);
    const ys = ring.map((p) => p[1]);
    return [xs.reduce((a, b) => a + b, 0) / xs.length, ys.reduce((a, b) => a + b, 0) / ys.length];
  } catch {
    return null;
  }
}

/**
 * Straight-line replay steps between two points over a time window.
 * `direction: "forward"` goes from → to; `"backward"` reverses the geometry
 * while keeping timestamps ascending across the window.
 */
export function buildReplaySteps(
  from: [number, number],
  to: [number, number],
  windowStartISO: string,
  windowEndISO: string,
  n = 24,
  direction: "forward" | "backward" = "forward",
): ReplayStep[] {
  const t0 = Date.parse(windowStartISO);
  const t1 = Date.parse(windowEndISO);
  const startMs = Number.isFinite(t0) ? t0 : Date.now() - 48 * 3600_000;
  const endMs = Number.isFinite(t1) && t1 > startMs ? t1 : startMs + 48 * 3600_000;
  const [a, b] = direction === "forward" ? [from, to] : [to, from];
  const steps: ReplayStep[] = [];
  for (let i = 0; i < n; i++) {
    const f = n === 1 ? 0 : i / (n - 1);
    steps.push({
      t: new Date(startMs + f * (endMs - startMs)).toISOString(),
      lon: a[0] + f * (b[0] - a[0]),
      lat: a[1] + f * (b[1] - a[1]),
    });
  }
  return steps;
}

/** Even-timestamped steps from a real (lon,lat,ts?) track, for the ship replay. */
export function stepsFromTrack(
  track: Array<[number, number] | [number, number, string]>,
  fallbackStartISO?: string | null,
  fallbackEndISO?: string | null,
): ReplayStep[] {
  if (!track.length) return [];
  const haveTs = track.every((p) => p.length > 2 && p[2]);
  if (haveTs) {
    return (track as Array<[number, number, string]>).map((p) => ({ lon: p[0], lat: p[1], t: p[2] }));
  }
  const s = Date.parse(fallbackStartISO ?? "");
  const e = Date.parse(fallbackEndISO ?? "");
  const startMs = Number.isFinite(s) ? s : Date.now() - track.length * 3600_000;
  const endMs = Number.isFinite(e) && e > startMs ? e : startMs + track.length * 3600_000;
  return track.map((p, i) => ({
    lon: p[0],
    lat: p[1],
    t: new Date(startMs + (track.length === 1 ? 0 : i / (track.length - 1)) * (endMs - startMs)).toISOString(),
  }));
}

/** Great-circle km from a point to a corridor (0 when the point is inside its centroid tolerance). */
export function distanceToCorridorKm(pt: [number, number], corridorGeom: unknown): number | null {
  const c = polygonCentroid(corridorGeom);
  if (!c) return null;
  const R = 6371;
  const dLat = ((pt[1] - c[1]) * Math.PI) / 180;
  const dLon = ((pt[0] - c[0]) * Math.PI) / 180;
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((c[1] * Math.PI) / 180) * Math.cos((pt[1] * Math.PI) / 180) * Math.sin(dLon / 2) ** 2;
  return Math.round(2 * R * Math.asin(Math.sqrt(s)) * 10) / 10;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DriftReplayBar({
  steps,
  activeIndex,
  onIndex,
  playing,
  onPlayToggle,
  direction,
  onDirection,
  stepMs = 700,
  extra,
  label,
}: {
  steps: ReplayStep[];
  activeIndex: number;
  onIndex: (i: number) => void;
  playing: boolean;
  onPlayToggle: (v: boolean) => void;
  direction?: "forward" | "backward";
  onDirection?: (d: "forward" | "backward") => void;
  stepMs?: number;
  extra?: ReactNode;
  label?: string;
}) {
  const n = steps.length;
  const cur = steps[Math.min(activeIndex, Math.max(0, n - 1))];

  useEffect(() => {
    if (!playing || n === 0) return;
    const id = window.setInterval(() => {
      onIndex(activeIndex >= n - 1 ? 0 : activeIndex + 1);
    }, stepMs);
    return () => window.clearInterval(id);
  }, [playing, activeIndex, n, stepMs, onIndex]);

  if (n === 0) return null;

  return (
    <div className="border-t border-border bg-muted/30 px-4 py-3">
      <div className="flex flex-wrap items-center gap-3">
        {onDirection && (
          <div className="inline-flex overflow-hidden rounded border border-border text-[11px]">
            {(["backward", "forward"] as const).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => onDirection(d)}
                className={`px-2 py-1 font-medium transition-colors ${
                  direction === d ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground hover:bg-muted"
                }`}
              >
                {d === "backward" ? "◀ Backward" : "Forward ▶"}
              </button>
            ))}
          </div>
        )}

        <div className="flex items-center gap-1">
          <button type="button" aria-label="First" onClick={() => { onPlayToggle(false); onIndex(0); }}
            className="flex h-7 w-7 items-center justify-center rounded border border-border hover:bg-muted">
            <SkipBack className="h-3.5 w-3.5" />
          </button>
          <button type="button" aria-label="Step back" onClick={() => { onPlayToggle(false); onIndex(Math.max(0, activeIndex - 1)); }}
            className="flex h-7 w-7 items-center justify-center rounded border border-border hover:bg-muted">
            <ChevronLeft className="h-3.5 w-3.5" />
          </button>
          <button type="button" aria-label={playing ? "Pause" : "Play"} onClick={() => onPlayToggle(!playing)}
            className="flex h-7 w-7 items-center justify-center rounded border border-border bg-background hover:bg-muted">
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </button>
          <button type="button" aria-label="Step forward" onClick={() => { onPlayToggle(false); onIndex(Math.min(n - 1, activeIndex + 1)); }}
            className="flex h-7 w-7 items-center justify-center rounded border border-border hover:bg-muted">
            <ChevronRight className="h-3.5 w-3.5" />
          </button>
          <button type="button" aria-label="Last" onClick={() => { onPlayToggle(false); onIndex(n - 1); }}
            className="flex h-7 w-7 items-center justify-center rounded border border-border hover:bg-muted">
            <SkipForward className="h-3.5 w-3.5" />
          </button>
        </div>

        <input
          type="range"
          min={0}
          max={n - 1}
          value={Math.min(activeIndex, n - 1)}
          onChange={(e) => { onPlayToggle(false); onIndex(parseInt(e.target.value, 10)); }}
          className="min-w-[140px] flex-1 accent-[var(--critical)]"
          aria-label="Replay scrubber"
        />

        <div className="mono-num text-right text-[11px] leading-tight text-muted-foreground">
          <div className="font-semibold text-foreground">{cur ? formatDateTime(cur.t) : "—"}</div>
          <div>
            step {Math.min(activeIndex, n - 1) + 1} / {n}
            {cur ? ` · ${cur.lat.toFixed(3)}°, ${cur.lon.toFixed(3)}°` : ""}
          </div>
          {extra && <div>{extra}</div>}
        </div>
      </div>
      {label && <p className="mt-1.5 text-[11px] leading-4 text-muted-foreground">{label}</p>}
    </div>
  );
}
