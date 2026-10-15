/**
 * LayerSwitcherPanel.tsx
 *
 * Floating card component for toggling individual map layers on/off.
 * Renders a labelled Switch row for each LayerVisibility key.
 *
 * Requirements: 1.6, 3.4, 4.4, 10.3
 */

import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import type { LayerVisibility } from "@/components/oricean/OriceanMapLibre";

// ---------------------------------------------------------------------------
// Layer metadata — display labels for each LayerVisibility key
// ---------------------------------------------------------------------------

const LAYER_LABELS: { key: keyof LayerVisibility; label: string; description: string }[] = [
  { key: "sar",        label: "SENTINEL-1 SAR",  description: "SAR imagery raster" },
  { key: "tileGrid",   label: "Tile Grid",        description: "Analysed tile polygons" },
  { key: "detections", label: "Probable Spill Region", description: "Organic spill hotspot blob" },
  { key: "gradcam",    label: "Grad-CAM",         description: "Model attention heatmaps" },
  { key: "ais",        label: "AIS Tracks",       description: "Vessel track lines" },
];

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface LayerSwitcherPanelProps {
  visibleLayers: LayerVisibility;
  onToggle: (key: keyof LayerVisibility) => void;
  className?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function LayerSwitcherPanel({
  visibleLayers,
  onToggle,
  className,
}: LayerSwitcherPanelProps) {
  return (
    <div
      className={cn(
        "min-w-[200px] rounded-md border border-white/15 bg-black/80 shadow-xl backdrop-blur-sm",
        className,
      )}
    >
      {/* Header */}
      <div className="border-b border-white/10 px-3 py-2">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-white/50">
          Layers
        </p>
      </div>

      {/* Toggle rows */}
      <div className="divide-y divide-white/[0.06] px-3 py-1">
        {LAYER_LABELS.map(({ key, label, description }) => (
          <div key={key} className="flex items-center justify-between gap-3 py-2">
            <div className="min-w-0">
              <p className="text-xs font-medium text-white/85 leading-tight">{label}</p>
              <p className="text-[10px] text-white/40 leading-tight mt-0.5">{description}</p>
            </div>
            <Switch
              checked={visibleLayers[key]}
              onCheckedChange={() => onToggle(key)}
              aria-label={`Toggle ${label} layer`}
              className="shrink-0"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default LayerSwitcherPanel;
