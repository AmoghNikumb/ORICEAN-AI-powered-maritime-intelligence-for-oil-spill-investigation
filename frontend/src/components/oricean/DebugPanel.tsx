/**
 * DebugPanel.tsx
 *
 * Analyst/developer debug panel exposing raw per-tile inference diagnostics.
 * Sections:
 *  1. Threshold control — Radix UI Slider bound to displayThreshold
 *  2. Probability distribution — Recharts BarChart with 10 confidence buckets
 *  3. Top-10 tiles table — sorted by confidence descending
 *  4. Per-tile diagnostics table — full pixel_stats + debug_info per tile
 *
 * Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 9.1, 9.2, 9.4, 9.7
 */

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import type { InferenceResult, TilePrediction } from "@/lib/api";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_THRESHOLD = 0.5;
const NUM_BUCKETS = 10;

// ---------------------------------------------------------------------------
// Helper: build 10-bucket histogram from tile confidence values
// ---------------------------------------------------------------------------

function buildHistogram(
  tiles: TilePrediction[],
): { bucket: string; count: number; start: number }[] {
  const buckets = Array.from({ length: NUM_BUCKETS }, (_, i) => ({
    bucket: `${(i / NUM_BUCKETS).toFixed(1)}–${((i + 1) / NUM_BUCKETS).toFixed(1)}`,
    count: 0,
    start: i / NUM_BUCKETS,
  }));

  for (const tile of tiles) {
    // Clamp to [0, 1]
    const c = Math.min(1, Math.max(0, tile.confidence));
    // Map to bucket index (tile.confidence === 1.0 goes in the last bucket)
    const idx = Math.min(NUM_BUCKETS - 1, Math.floor(c * NUM_BUCKETS));
    buckets[idx].count += 1;
  }

  return buckets;
}

// ---------------------------------------------------------------------------
// Helper: get top N tiles by confidence descending
// ---------------------------------------------------------------------------

export function getTopTilesByProbability(
  tiles: TilePrediction[],
  n: number,
): TilePrediction[] {
  return [...tiles]
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, n);
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface DebugPanelProps {
  inferenceResult: InferenceResult | null;
  displayThreshold: number;
  onThresholdChange: (t: number) => void;
  className?: string;
}

// ---------------------------------------------------------------------------
// Section header
// ---------------------------------------------------------------------------

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-white/40">
      {children}
    </p>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function DebugPanel({
  inferenceResult,
  displayThreshold,
  onThresholdChange,
  className,
}: DebugPanelProps) {
  const tiles = inferenceResult?.tile_results ?? [];

  // Pre-compute derived data
  const histogram = useMemo(() => buildHistogram(tiles), [tiles]);
  const top10 = useMemo(() => getTopTilesByProbability(tiles, 10), [tiles]);

  // Recharts reference line position in bucket-index space
  // We need to convert threshold (0–1) to the matching bucket label
  const thresholdBucketLabel = useMemo(() => {
    const idx = Math.min(NUM_BUCKETS - 1, Math.floor(displayThreshold * NUM_BUCKETS));
    return histogram[idx]?.bucket ?? "";
  }, [displayThreshold, histogram]);

  if (!inferenceResult) {
    return (
      <div
        className={cn(
          "flex items-center justify-center rounded-md border border-white/10 bg-black/60 px-6 py-10 text-sm text-white/40",
          className,
        )}
      >
        Inference has not yet been run.
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex flex-col gap-6 rounded-md border border-white/15 bg-[#070d18] p-5 text-white",
        className,
      )}
    >
      {/* ── Section 1: Threshold control ─────────────────────────────────── */}
      <div>
        <SectionHeader>Classification Threshold</SectionHeader>
        <div className="flex items-center justify-between text-xs mb-3">
          <span className="text-white/50">
            Default: <span className="font-mono text-white/70">{DEFAULT_THRESHOLD.toFixed(2)}</span>
          </span>
          <span className="font-semibold text-white">
            Active:{" "}
            <span
              className={cn(
                "font-mono",
                displayThreshold !== DEFAULT_THRESHOLD
                  ? "text-orange-400"
                  : "text-white",
              )}
            >
              {displayThreshold.toFixed(2)}
            </span>
          </span>
        </div>

        <Slider
          min={0}
          max={1}
          step={0.01}
          value={[displayThreshold]}
          onValueChange={([v]) => onThresholdChange(v)}
          aria-label="Classification threshold"
          className="w-full"
        />

        <div className="mt-1.5 flex justify-between text-[10px] text-white/30">
          <span>0.00 (all positive)</span>
          <span>1.00 (none positive)</span>
        </div>

        {displayThreshold !== DEFAULT_THRESHOLD && (
          <button
            type="button"
            onClick={() => onThresholdChange(DEFAULT_THRESHOLD)}
            className="mt-2 text-[11px] text-white/40 underline underline-offset-2 hover:text-white/70 transition-colors"
          >
            Reset to default (0.50)
          </button>
        )}
      </div>

      {/* ── Section 2: Probability distribution ──────────────────────────── */}
      <div>
        <SectionHeader>
          Probability Distribution ({tiles.length} tile{tiles.length !== 1 ? "s" : ""})
        </SectionHeader>

        {tiles.length === 0 ? (
          <p className="text-xs text-white/30 italic">No tiles to display.</p>
        ) : (
          <div className="h-[160px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={histogram}
                margin={{ top: 4, right: 4, bottom: 4, left: -20 }}
                barCategoryGap="8%"
              >
                <XAxis
                  dataKey="bucket"
                  tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 9 }}
                  tickLine={false}
                  axisLine={false}
                  interval={1}
                />
                <YAxis
                  tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 9 }}
                  tickLine={false}
                  axisLine={false}
                  allowDecimals={false}
                />
                <RechartsTooltip
                  contentStyle={{
                    background: "rgba(0,0,0,0.85)",
                    border: "1px solid rgba(255,255,255,0.15)",
                    borderRadius: 4,
                    fontSize: 11,
                    color: "rgba(255,255,255,0.85)",
                  }}
                  itemStyle={{ color: "rgba(255,255,255,0.85)" }}
                  formatter={(value: number) => [`${value} tile${value !== 1 ? "s" : ""}`, "Count"]}
                />

                {/* Reference line at the active threshold bucket */}
                <ReferenceLine
                  x={thresholdBucketLabel}
                  stroke="#f97316"
                  strokeWidth={1.5}
                  strokeDasharray="4 3"
                  label={{
                    value: `t=${displayThreshold.toFixed(2)}`,
                    fill: "#f97316",
                    fontSize: 9,
                    position: "insideTopLeft",
                  }}
                />

                <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                  {histogram.map((entry) => (
                    <Cell
                      key={entry.bucket}
                      fill={
                        entry.start >= displayThreshold
                          ? "rgba(249,115,22,0.7)"   // orange — would be positive
                          : "rgba(34,197,94,0.5)"    // green — below threshold
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Detection count at current threshold */}
        {tiles.length > 0 && (() => {
          const positiveCount = tiles.filter((t) => t.confidence >= displayThreshold).length;
          return (
            <p className="mt-1 text-[11px] text-white/40">
              {positiveCount === 0
                ? "0 detections at this threshold"
                : `${positiveCount} tile${positiveCount !== 1 ? "s" : ""} above threshold`}
            </p>
          );
        })()}
      </div>

      {/* ── Section 3: Top-10 tiles ───────────────────────────────────────── */}
      <div>
        <SectionHeader>Top-10 Tiles by Confidence</SectionHeader>

        {top10.length === 0 ? (
          <p className="text-xs text-white/30 italic">No tile results available.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[280px] text-xs">
              <thead>
                <tr className="border-b border-white/10 text-[10px] uppercase tracking-wider text-white/35">
                  <th className="pb-1.5 pr-4 text-left font-semibold">#</th>
                  <th className="pb-1.5 pr-4 text-left font-semibold">Tile ID</th>
                  <th className="pb-1.5 text-right font-semibold">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.05]">
                {top10.map((tile, idx) => (
                  <tr key={tile.tile_id} className="text-white/75">
                    <td className="py-1.5 pr-4 font-mono text-white/35 text-[10px]">
                      {(idx + 1).toString().padStart(2, "0")}
                    </td>
                    <td className="py-1.5 pr-4 font-mono text-[11px] text-white/80 truncate max-w-[140px]">
                      {tile.tile_id}
                    </td>
                    <td className="py-1.5 text-right font-mono">
                      <span
                        className={cn(
                          "font-semibold",
                          tile.confidence >= displayThreshold
                            ? "text-orange-400"
                            : "text-green-400",
                        )}
                      >
                        {tile.confidence.toFixed(6)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Section 4: Per-tile diagnostics ──────────────────────────────── */}
      <div>
        <SectionHeader>Per-Tile Diagnostics ({tiles.length} total)</SectionHeader>

        {tiles.length === 0 ? (
          <p className="text-xs text-white/30 italic">No tiles to display.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-xs">
              <thead>
                <tr className="border-b border-white/10 text-[10px] uppercase tracking-wider text-white/35">
                  <th className="pb-1.5 pr-3 text-left font-semibold">Tile ID</th>
                  <th className="pb-1.5 pr-3 text-right font-semibold">Confidence</th>
                  <th className="pb-1.5 pr-3 text-right font-semibold">Min</th>
                  <th className="pb-1.5 pr-3 text-right font-semibold">Max</th>
                  <th className="pb-1.5 pr-3 text-right font-semibold">Mean</th>
                  <th className="pb-1.5 pr-3 text-right font-semibold">Std</th>
                  <th className="pb-1.5 text-right font-semibold">Input Dims</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {tiles.map((tile) => (
                  <tr key={tile.tile_id} className="text-white/65 hover:bg-white/[0.03]">
                    <td className="py-1.5 pr-3 font-mono text-[10px] text-white/70 max-w-[120px] truncate">
                      {tile.tile_id}
                    </td>
                    <td className="py-1.5 pr-3 text-right font-mono">
                      <span
                        className={
                          tile.confidence >= displayThreshold
                            ? "text-orange-400"
                            : "text-green-400"
                        }
                      >
                        {tile.confidence.toFixed(4)}
                      </span>
                    </td>

                    {/* Pixel stats — inline or "unavailable" message */}
                    {tile.pixel_stats ? (
                      <>
                        <td className="py-1.5 pr-3 text-right font-mono text-[11px]">
                          {tile.pixel_stats.min_value.toFixed(2)}
                        </td>
                        <td className="py-1.5 pr-3 text-right font-mono text-[11px]">
                          {tile.pixel_stats.max_value.toFixed(2)}
                        </td>
                        <td className="py-1.5 pr-3 text-right font-mono text-[11px]">
                          {tile.pixel_stats.mean_value.toFixed(2)}
                        </td>
                        <td className="py-1.5 pr-3 text-right font-mono text-[11px]">
                          {tile.pixel_stats.std_value.toFixed(2)}
                        </td>
                      </>
                    ) : (
                      <td
                        colSpan={4}
                        className="py-1.5 pr-3 text-center text-[10px] text-white/25 italic"
                      >
                        Pixel stats unavailable
                      </td>
                    )}

                    {/* Input dimensions from debug_info */}
                    <td className="py-1.5 text-right font-mono text-[10px] text-white/45">
                      {tile.debug_info
                        ? `${tile.debug_info.input_dimensions[0]}×${tile.debug_info.input_dimensions[1]}`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default DebugPanel;
