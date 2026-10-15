import { geoMercator, geoPath, geoGraticule10 } from "d3-geo";
import { feature } from "topojson-client";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import landTopo from "world-atlas/land-110m.json";
import { cn } from "@/lib/utils";

const W = 1000;
const H = 560;
const MIN_K = 0.6;
const MAX_K = 60;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const land = feature(landTopo as any, (landTopo as any).objects.land) as any;

export interface MapHelpers {
  /** projects [lon, lat] into map space */
  p: (c: [number, number]) => [number, number];
  /** path string for a lon/lat ring */
  poly: (ring: [number, number][]) => string;
  /** path string for a lon/lat line */
  line: (coords: [number, number][]) => string;
  /** current zoom factor — divide sizes by this to keep markers constant */
  k: number;
}

interface MapViewProps {
  center?: [number, number];
  /** degrees of longitude visible across the map width */
  span?: number;
  className?: string;
  children?: (h: MapHelpers) => ReactNode;
  overlay?: ReactNode;
  interactive?: boolean;
  variant?: "ocean" | "satellite";
}

export function MapView({
  center = [20, 15],
  span = 360,
  className,
  children,
  overlay,
  interactive = true,
  variant = "ocean",
}: MapViewProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({ k: 1, x: 0, y: 0 });
  const viewRef = useRef(view);
  viewRef.current = view;
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);

  const { projection, path } = useMemo(() => {
    const proj = geoMercator()
      .center(center)
      .scale((W / span) * (180 / Math.PI) * (span >= 300 ? 1 : 1))
      .translate([W / 2, H / 2]);
    return { projection: proj, path: geoPath(proj) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [center[0], center[1], span]);

  const helpers: MapHelpers = useMemo(() => {
    const p = (c: [number, number]) => (projection(c) ?? [0, 0]) as [number, number];
    return {
      p,
      k: view.k,
      poly: (ring) => ring.map((c, i) => `${i ? "L" : "M"}${p(c)[0].toFixed(2)},${p(c)[1].toFixed(2)}`).join(" ") + " Z",
      line: (coords) => coords.map((c, i) => `${i ? "L" : "M"}${p(c)[0].toFixed(2)},${p(c)[1].toFixed(2)}`).join(" "),
    };
  }, [projection, view.k]);

  useEffect(() => {
    if (!interactive) return;
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const sx = ((e.clientX - rect.left) / rect.width) * W;
      const sy = ((e.clientY - rect.top) / rect.height) * H;
      const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 100 : 1);
      setView((v) => {
        const next = Math.min(MAX_K, Math.max(MIN_K, v.k * Math.exp(-dy * 0.0018)));
        const r = next / v.k;
        return { k: next, x: sx - (sx - v.x) * r, y: sy - (sy - v.y) * r };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [interactive]);

  const zoomBy = (f: number) =>
    setView((v) => {
      const next = Math.min(MAX_K, Math.max(MIN_K, v.k * f));
      const r = next / v.k;
      return { k: next, x: W / 2 - (W / 2 - v.x) * r, y: H / 2 - (H / 2 - v.y) * r };
    });

  return (
    <div
      ref={wrapRef}
      className={cn(
        "relative overflow-hidden rounded-md border border-border bg-ocean-deep select-none",
        interactive && "cursor-grab active:cursor-grabbing",
        className,
      )}
      onPointerDown={(e) => {
        if (!interactive) return;
        (e.target as Element).setPointerCapture?.(e.pointerId);
        dragRef.current = { x: e.clientX, y: e.clientY, ox: viewRef.current.x, oy: viewRef.current.y };
      }}
      onPointerMove={(e) => {
        const d = dragRef.current;
        if (!d || !wrapRef.current) return;
        const rect = wrapRef.current.getBoundingClientRect();
        const sc = W / rect.width;
        setView((v) => ({ ...v, x: d.ox + (e.clientX - d.x) * sc, y: d.oy + (e.clientY - d.y) * sc }));
      }}
      onPointerUp={() => (dragRef.current = null)}
      onPointerLeave={() => (dragRef.current = null)}
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="block h-full w-full">
        <defs>
          <linearGradient id="oceanGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--ocean-deep)" />
            <stop offset="100%" stopColor="color-mix(in oklch, var(--ocean) 55%, var(--ocean-deep))" />
          </linearGradient>
          <linearGradient id="sarGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="oklch(0.32 0.01 250)" />
            <stop offset="50%" stopColor="oklch(0.42 0.012 240)" />
            <stop offset="100%" stopColor="oklch(0.3 0.01 250)" />
          </linearGradient>
          <pattern id="sarNoise" width="6" height="6" patternUnits="userSpaceOnUse">
            <rect width="6" height="6" fill="transparent" />
            <circle cx="1" cy="2" r="0.7" fill="oklch(1 0 0 / 0.06)" />
            <circle cx="4" cy="5" r="0.6" fill="oklch(1 0 0 / 0.05)" />
            <circle cx="5" cy="1" r="0.5" fill="oklch(0 0 0 / 0.07)" />
          </pattern>
        </defs>

        <rect width={W} height={H} fill={variant === "satellite" ? "url(#sarGrad)" : "url(#oceanGrad)"} />
        {variant === "satellite" && <rect width={W} height={H} fill="url(#sarNoise)" />}

        <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
          <path
            d={path(geoGraticule10()) ?? ""}
            fill="none"
            stroke="oklch(1 0 0 / 0.09)"
            strokeWidth={0.6}
            vectorEffect="non-scaling-stroke"
          />
          <path
            d={path(land) ?? ""}
            fill={variant === "satellite" ? "oklch(0.26 0.012 250)" : "var(--land)"}
            stroke="var(--land-border)"
            strokeWidth={0.7}
            vectorEffect="non-scaling-stroke"
          />
          {children?.(helpers)}
        </g>
      </svg>

      {interactive && (
        <div className="absolute right-3 top-3 flex flex-col overflow-hidden rounded-md border border-border bg-card shadow-panel">
          <button
            type="button"
            aria-label="Zoom in"
            className="px-2.5 py-1.5 text-sm text-card-foreground transition-colors hover:bg-muted"
            onClick={() => zoomBy(1.4)}
          >
            +
          </button>
          <button
            type="button"
            aria-label="Zoom out"
            className="border-t border-border px-2.5 py-1.5 text-sm text-card-foreground transition-colors hover:bg-muted"
            onClick={() => zoomBy(1 / 1.4)}
          >
            −
          </button>
          <button
            type="button"
            aria-label="Reset view"
            className="border-t border-border px-2.5 py-1 text-[10px] tracking-wider text-muted-foreground transition-colors hover:bg-muted"
            onClick={() => setView({ k: 1, x: 0, y: 0 })}
          >
            RST
          </button>
        </div>
      )}
      {overlay}
    </div>
  );
}

export const statusColor: Record<string, string> = {
  confirmed: "var(--critical)",
  investigating: "var(--warning)",
  potential: "var(--caution)",
  resolved: "var(--resolved)",
};
