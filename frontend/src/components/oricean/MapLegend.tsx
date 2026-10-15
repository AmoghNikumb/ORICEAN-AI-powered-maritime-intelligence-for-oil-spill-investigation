/**
 * MapLegend.tsx
 *
 * Small legend card showing colour swatches for each map layer state.
 * Colour tokens match those used in tileStateToColor() in OriceanMapLibre.tsx.
 *
 * Requirements: 10.2
 */

import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Legend entries — colours match tileStateToColor and spill/gradcam layers
// ---------------------------------------------------------------------------

const LEGEND_ITEMS = [
  {
    label: "AOI Boundary",
    // Inline style: matches the aoi-outline line colour in OriceanMapLibre
    color: "#38bdf8",
    style: "border" as const,
  },
  {
    label: "Analysed — No Oil",
    // var(--resolved) fallback
    color: "var(--resolved, #22c55e)",
    style: "fill" as const,
  },
  {
    label: "Oil Detected",
    // var(--spill) fallback
    color: "var(--spill, #f97316)",
    style: "fill" as const,
  },
  {
    label: "Spill Polygon",
    // Spill detection fill: dc2626 red (see OriceanMapLibre 9.6)
    color: "#dc2626",
    style: "fill" as const,
  },
  {
    label: "Grad-CAM Overlay",
    // Warm heatmap indicator
    color: "#facc15",
    style: "fill" as const,
  },
] as const;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface MapLegendProps {
  className?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function MapLegend({ className }: MapLegendProps) {
  return (
    <div
      className={cn(
        "rounded-md border border-white/15 bg-black/80 px-3 py-2 shadow-xl backdrop-blur-sm",
        className,
      )}
    >
      {/* Header */}
      <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-white/50">
        Legend
      </p>

      {/* Swatch rows */}
      <div className="space-y-1.5">
        {LEGEND_ITEMS.map(({ label, color, style }) => (
          <div key={label} className="flex items-center gap-2">
            {style === "border" ? (
              // AOI boundary — hollow square with coloured border
              <span
                className="h-3 w-3 shrink-0 rounded-sm border-2"
                style={{ borderColor: color, backgroundColor: "transparent" }}
                aria-hidden="true"
              />
            ) : (
              // Filled swatch with slight opacity to mimic map layers
              <span
                className="h-3 w-3 shrink-0 rounded-sm opacity-80"
                style={{ backgroundColor: color }}
                aria-hidden="true"
              />
            )}
            <span className="text-[11px] text-white/75 leading-none">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default MapLegend;
