/**
 * TileInspectorSheet.tsx
 *
 * Right-side Radix UI Sheet that opens when a map tile is clicked.
 * Displays full per-tile diagnostic information including SAR image patch,
 * Grad-CAM overlay, pixel stats table, and all classification metadata.
 *
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8
 */

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { TilePrediction } from "@/lib/api";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TileInspectorSheetProps {
  tile: TilePrediction | null;
  threshold: number;
  onClose: () => void;
}

// ---------------------------------------------------------------------------
// Small helper: label badge colour
// ---------------------------------------------------------------------------

function LabelBadge({ label }: { label: "Oil Spill" | "No Oil" }) {
  return (
    <Badge
      className={cn(
        "text-xs font-semibold",
        label === "Oil Spill"
          ? "border-orange-500/40 bg-orange-500/15 text-orange-400"
          : "border-green-500/40 bg-green-500/15 text-green-400",
      )}
      variant="outline"
    >
      {label}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Small helper: single stat row
// ---------------------------------------------------------------------------

function StatRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <tr className="border-b border-white/[0.06] last:border-0">
      <td className="py-1.5 pr-4 text-[11px] text-white/50">{label}</td>
      <td className="py-1.5 font-mono text-xs text-white/85">{value}</td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TileInspectorSheet({ tile, threshold, onClose }: TileInspectorSheetProps) {
  const isOpen = tile !== null;

  // Determine SAR image source (prefer base64, fall back to URL)
  const imageSrc = tile?.image_data_b64
    ? tile.image_data_b64.startsWith("data:")
      ? tile.image_data_b64
      : `data:image/png;base64,${tile.image_data_b64}`
    : (tile?.image_url ?? "");

  const hasGradcam = Boolean(tile?.gradcam_b64 && tile?.is_positive);
  const gradcamSrc = tile?.gradcam_b64
    ? tile.gradcam_b64.startsWith("data:")
      ? tile.gradcam_b64
      : `data:image/png;base64,${tile.gradcam_b64}`
    : "";

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-0 overflow-y-auto border-white/15 bg-[#0a0f1a] p-0 text-white sm:max-w-md"
      >
        {tile && (
          <>
            {/* Header */}
            <SheetHeader className="border-b border-white/10 px-5 py-4">
              <SheetTitle className="flex items-center gap-3 text-sm text-white">
                <span className="font-mono text-white/60 text-xs">{tile.tile_id}</span>
                <LabelBadge label={tile.label} />
              </SheetTitle>
              <SheetDescription className="text-[11px] text-white/40">
                Tile inspection — acquisition {tile.acquisition_date || "unknown"}
              </SheetDescription>
            </SheetHeader>

            <div className="flex flex-col gap-5 px-5 py-4">

              {/* ── SAR Image + Grad-CAM Overlay ─────────────────────────────── */}
              {imageSrc ? (
                <div>
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-white/40">
                    SAR Patch
                  </p>
                  <div className="relative overflow-hidden rounded border border-white/10">
                    {/* Base SAR image */}
                    <img
                      src={imageSrc}
                      alt={`SAR patch for tile ${tile.tile_id}`}
                      loading="lazy"
                      className="block h-auto w-full object-cover"
                      onError={(e) => {
                        (e.currentTarget as HTMLImageElement).style.display = "none";
                      }}
                    />

                    {/* Grad-CAM overlay — position: absolute, mix-blend-mode: multiply */}
                    {hasGradcam && gradcamSrc && (
                      <img
                        src={gradcamSrc}
                        alt="Grad-CAM heatmap overlay"
                        loading="lazy"
                        className="absolute inset-0 h-full w-full object-cover opacity-70"
                        style={{ mixBlendMode: "multiply" }}
                      />
                    )}
                  </div>

                  {/* Grad-CAM availability notice */}
                  {!hasGradcam && (
                    <p className="mt-1.5 text-[11px] text-white/35 italic">
                      {tile.is_positive
                        ? "(Grad-CAM not available)"
                        : "(Grad-CAM not generated for No Oil tiles)"}
                    </p>
                  )}
                </div>
              ) : (
                <div className="flex h-28 items-center justify-center rounded border border-white/10 bg-white/5">
                  <p className="text-xs text-white/30">No SAR image available</p>
                </div>
              )}

              {/* ── Classification details ────────────────────────────────────── */}
              <div>
                <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-white/40">
                  Classification
                </p>
                <table className="w-full">
                  <tbody>
                    <StatRow label="Label" value={<LabelBadge label={tile.label} />} />
                    <StatRow
                      label="Raw confidence"
                      value={
                        <span className="font-mono">
                          {tile.confidence.toFixed(6)}
                        </span>
                      }
                    />
                    <StatRow
                      label="Threshold used"
                      value={
                        <span className={tile.confidence >= threshold ? "text-orange-400" : "text-green-400"}>
                          {threshold.toFixed(2)}
                        </span>
                      }
                    />
                    <StatRow
                      label="Is positive"
                      value={
                        <span className={tile.is_positive ? "text-orange-400" : "text-green-400"}>
                          {tile.is_positive ? "Yes" : "No"}
                        </span>
                      }
                    />
                    <StatRow label="Acquisition date" value={tile.acquisition_date || "—"} />
                  </tbody>
                </table>
              </div>

              {/* ── Geographic bounds ─────────────────────────────────────────── */}
              <div>
                <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-white/40">
                  Bounds (WGS-84)
                </p>
                <table className="w-full">
                  <tbody>
                    <StatRow label="Min Lon (W)" value={`${tile.bounds[0].toFixed(6)}°`} />
                    <StatRow label="Min Lat (S)" value={`${tile.bounds[1].toFixed(6)}°`} />
                    <StatRow label="Max Lon (E)" value={`${tile.bounds[2].toFixed(6)}°`} />
                    <StatRow label="Max Lat (N)" value={`${tile.bounds[3].toFixed(6)}°`} />
                  </tbody>
                </table>
              </div>

              {/* ── Pixel stats ───────────────────────────────────────────────── */}
              {tile.pixel_stats ? (
                <div>
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-white/40">
                    Pixel Statistics
                  </p>
                  <table className="w-full">
                    <tbody>
                      <StatRow label="Min value" value={tile.pixel_stats.min_value.toFixed(4)} />
                      <StatRow label="Max value" value={tile.pixel_stats.max_value.toFixed(4)} />
                      <StatRow label="Mean value" value={tile.pixel_stats.mean_value.toFixed(4)} />
                      <StatRow label="Std deviation" value={tile.pixel_stats.std_value.toFixed(4)} />
                    </tbody>
                  </table>
                </div>
              ) : (
                <div>
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-widest text-white/40">
                    Pixel Statistics
                  </p>
                  <p className="text-xs text-white/30 italic">Pixel stats unavailable</p>
                </div>
              )}

              {/* ── Debug info ────────────────────────────────────────────────── */}
              {tile.debug_info && (
                <div>
                  <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-white/40">
                    Debug Info
                  </p>
                  <table className="w-full">
                    <tbody>
                      <StatRow
                        label="Input dims (H×W)"
                        value={`${tile.debug_info.input_dimensions[0]} × ${tile.debug_info.input_dimensions[1]}`}
                      />
                      <StatRow
                        label="Model input (C×H×W)"
                        value={`${tile.debug_info.model_input_dims[0]} × ${tile.debug_info.model_input_dims[1]} × ${tile.debug_info.model_input_dims[2]}`}
                      />
                      <StatRow
                        label="Preprocessing"
                        value={
                          <ul className="list-none space-y-0.5">
                            {tile.debug_info.preprocessing_steps.map((s, i) => (
                              <li key={i} className="text-white/70">{s}</li>
                            ))}
                          </ul>
                        }
                      />
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

export default TileInspectorSheet;
