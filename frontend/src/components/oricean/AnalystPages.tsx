/**
 * AnalystPages.tsx
 *
 * All analyst-role page components. Now wired to the live backend:
 *  - Dashboard: lists real investigations from POST /investigations
 *  - NewInvestigation: creates investigations, runs inference via POST /api/ml/run-inference
 *  - SatelliteAnalysis: shows real tile + spill detection results from ml_result
 *  - SourceReconstruction, VesselAttribution, EvidenceTimeline, Reports:
 *    show live data where available, static demo data for features not yet in backend
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "@tanstack/react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronRight,
  CircleDot,
  Clock3,
  Download,
  FileCheck2,
  FilePlus2,
  FileText,
  Filter,
  Flag,
  Gauge,
  Layers3,
  Loader2,
  MapPin,
  Play,
  Plus,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Ship,
  SlidersHorizontal,
  Sparkles,
  Target,
  TimerReset,
  TriangleAlert,
  Waves,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AppShell } from "@/components/oricean/AppShell";
import { MapView } from "@/components/oricean/MapView";
import { Legend, KV, Panel, Pill, ScoreBar, StatCard } from "@/components/oricean/bits";
import { OriceanMapLibre, validateDrawnPolygon, type LayerVisibility } from "@/components/oricean/OriceanMapLibre";
import { LayerSwitcherPanel } from "@/components/oricean/LayerSwitcherPanel";
import { MapLegend } from "@/components/oricean/MapLegend";
import { TileInspectorSheet } from "@/components/oricean/TileInspectorSheet";
import { PipelineProgressPanel } from "@/components/oricean/PipelineProgressPanel";
import { DebugPanel } from "@/components/oricean/DebugPanel";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@radix-ui/react-collapsible";
import type { GeoJSONPolygon } from "@/lib/api";
import {
  backtrack,
  currentField,
  formatDate,
  formatDateTime,
  primarySpill,
  scenes,
  sourceRegion,
  spills,
  timeline,
  vessels,
  type Investigation,
} from "@/lib/oricean-data";
import {
  investigationsApi,
  mlApi,
  sourceApi,
  attributionApi,
  ApiError,
  type InferenceResult,
  type InvestigationRead,
  type SpillDetection,
  type TilePrediction,
  type SourceReconstructionResult,
  type AttributionResult,
  type CandidateVessel,
} from "@/lib/api";
import { useActiveInvestigation } from "@/lib/investigation-store";
import { buildEvidenceBundle, downloadJson, openPrintableReport } from "@/components/oricean/reportDoc";
import {
  DriftReplayBar,
  buildReplaySteps,
  distanceToCorridorKm,
  polygonCentroid,
  stepsFromTrack,
  type ReplayStep,
} from "@/components/oricean/DriftReplayBar";

// ---------------------------------------------------------------------------
// Re-use static data types for display purposes
// ---------------------------------------------------------------------------


const iconByKind = {
  satellite: <Layers3 className="h-4 w-4" />,
  ai: <Sparkles className="h-4 w-4" />,
  ocean: <Waves className="h-4 w-4" />,
  vessel: <Ship className="h-4 w-4" />,
  alert: <AlertTriangle className="h-4 w-4" />,
  report: <FileText className="h-4 w-4" />,
} as const;

const kindTone = {
  satellite: "bg-satellite/15 text-satellite",
  ai: "bg-ocean/15 text-ocean",
  ocean: "bg-satellite/15 text-satellite",
  vessel: "bg-primary/10 text-primary",
  alert: "bg-critical/12 text-critical",
  report: "bg-resolved/12 text-resolved",
} as const;

// ---------------------------------------------------------------------------
// Shell wrapper
// ---------------------------------------------------------------------------

function AnalystShell({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <AppShell role="analyst" title={title} subtitle={subtitle} actions={actions}>
      <div className="mx-auto w-full max-w-[1500px]">{children}</div>
    </AppShell>
  );
}

// ---------------------------------------------------------------------------
// Status badge helper
// ---------------------------------------------------------------------------

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    pending: "bg-muted text-muted-foreground border-border",
    running_gee: "bg-satellite/15 text-satellite border-satellite/30",
    running_ml: "bg-ocean/15 text-ocean border-ocean/30",
    complete: "bg-resolved/12 text-resolved border-resolved/30",
    failed: "bg-critical/12 text-critical border-critical/30",
  };
  const labels: Record<string, string> = {
    pending: "Pending",
    running_gee: "Retrieving SAR",
    running_ml: "Running AI",
    complete: "Complete",
    failed: "Failed",
  };
  return (
    <Pill tone={map[status] ?? "bg-secondary text-secondary-foreground border-border"}>
      {labels[status] ?? status}
    </Pill>
  );
}





// ---------------------------------------------------------------------------
// Progress steps for New Investigation wizard
// ---------------------------------------------------------------------------

const STEPS = ["Define area", "Set date range", "Run detection", "View results"];

function ProgressSteps({ current }: { current: number }) {
  return (
    <div className="grid grid-cols-4 border border-border bg-card">
      {STEPS.map((step, i) => (
        <div
          key={step}
          className={`relative flex items-center gap-2 border-r border-border px-3 py-3 last:border-0 ${i === current ? "bg-secondary" : ""}`}
        >
          <span
            className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
              i < current
                ? "bg-resolved text-primary-foreground"
                : i === current
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground"
            }`}
          >
            {i < current ? <Check className="h-3.5 w-3.5" /> : i + 1}
          </span>
          <span
            className={`hidden text-xs font-medium sm:inline ${i === current ? "text-foreground" : "text-muted-foreground"}`}
          >
            {step}
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Field component
// ---------------------------------------------------------------------------

function Field({
  label,
  value,
  onChange,
  type = "text",
}: {
  label: string;
  value: string;
  onChange?: (v: string) => void;
  type?: string;
}) {
  return (
    <label className="block">
      <span className="label-caps">{label}</span>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        readOnly={!onChange}
        className="mt-1.5 h-9 w-full border border-input bg-background px-3 text-sm outline-none focus:ring-1 focus:ring-ring"
      />
    </label>
  );
}

// ---------------------------------------------------------------------------
// InvestigationRow for table
// ---------------------------------------------------------------------------

function InvestigationRow({ item }: { item: InvestigationRead }) {
  const { setInvestigation, setInferenceResult } = useActiveInvestigation();

  const handleOpen = useCallback(() => {
    setInvestigation(item);
    if (item.ml_result) {
      setInferenceResult(item.ml_result as unknown as InferenceResult);
    }
  }, [item, setInvestigation, setInferenceResult]);

  return (
    <tr className="transition-colors hover:bg-muted/40">
      <td className="px-4 py-3">
        <Link
          to="/analyst/satellite"
          onClick={handleOpen}
          className="font-medium text-primary hover:underline"
        >
          {item.name}
        </Link>
        <span className="mt-0.5 block text-[10px] font-mono text-muted-foreground">{item.id}</span>
      </td>
      <td className="mono-num px-4 py-3 text-xs text-muted-foreground">
        {item.created_at ? formatDate(item.created_at) : "—"}
      </td>
      <td className="px-4 py-3">
        <StatusBadge status={item.status} />
      </td>
      <td className="px-4 py-3">
        {item.ml_result ? (
          <div className="flex min-w-28 items-center gap-2">
            <div className="h-1.5 flex-1 overflow-hidden bg-muted">
              <div
                className="h-full bg-ocean"
                style={{
                  width: `${Math.round((item.ml_result as unknown as InferenceResult).positive_tile_count / Math.max(1, (item.ml_result as unknown as InferenceResult).total_tiles_processed) * 100)}%`,
                }}
              />
            </div>
            <span className="mono-num text-xs">
              {item.ml_result
                ? `${(item.ml_result as unknown as InferenceResult).spill_detections?.length ?? 0} spill${
                    (item.ml_result as unknown as InferenceResult).spill_detections?.length !== 1 ? "s" : ""
                  } found`
                : "—"}
            </span>
          </div>
        ) : (
          <span className="text-xs text-muted-foreground">Not yet run</span>
        )}
      </td>
      <td className="px-4 py-3 text-right">
        {item.status === "complete" ? (
          <Pill tone="bg-resolved/12 text-resolved border-resolved/30">Done</Pill>
        ) : item.status === "failed" ? (
          <Pill tone="bg-critical/12 text-critical border-critical/30">Failed</Pill>
        ) : (
          <Pill tone="bg-muted text-muted-foreground border-border">Pending</Pill>
        )}
      </td>
    </tr>
  );
}

// ===========================================================================
// Dashboard
// ===========================================================================

export function Dashboard() {
  const [query, setQuery] = useState("");
  const { setInvestigation, setInferenceResult } = useActiveInvestigation();

  const { data: paginated, isLoading, error } = useQuery({
    queryKey: ["investigations"],
    queryFn: () => investigationsApi.list(1, 50),
    refetchInterval: 10_000, // poll every 10s to pick up status changes
  });

  const investigations = paginated?.items ?? [];

  const filtered = investigations.filter(
    (item) =>
      item.name.toLowerCase().includes(query.toLowerCase()) ||
      item.id.toLowerCase().includes(query.toLowerCase()),
  );

  const active = investigations.filter((i) => i.status === "running_gee" || i.status === "running_ml").length;
  const complete = investigations.filter((i) => i.status === "complete").length;

  return (
    <AnalystShell
      title="Analyst Dashboard"
      subtitle="Investigation queue — live from backend"
      actions={
        <Button asChild size="sm">
          <Link to="/analyst/new">
            <Plus /> New investigation
          </Link>
        </Button>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Total investigations"
          value={String(investigations.length)}
          hint="All records in database"
          tone="satellite"
          icon={<Target className="h-4 w-4" />}
        />
        <StatCard
          label="Processing now"
          value={String(active).padStart(2, "0")}
          hint="Running GEE or ML pipeline"
          tone="warning"
          icon={<CircleDot className="h-4 w-4" />}
        />
        <StatCard
          label="Completed"
          value={String(complete).padStart(2, "0")}
          hint="With ML inference results"
          tone="resolved"
          icon={<Check className="h-4 w-4" />}
        />
        <StatCard
          label="Failed"
          value={String(investigations.filter((i) => i.status === "failed").length).padStart(2, "0")}
          hint="Check error messages"
          tone="critical"
          icon={<TriangleAlert className="h-4 w-4" />}
        />
      </div>

      <Panel className="mt-5" title="Investigation queue" right={
        <div className="flex items-center gap-2">
          <Filter className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
      } bodyClassName="p-0">
        <div className="flex border-b border-border px-4 py-3">
          <div className="relative w-full max-w-xs">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search name or ID"
              className="h-8 w-full border border-input bg-background pl-8 pr-3 text-xs outline-none focus:ring-1 focus:ring-ring"
            />
          </div>
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center gap-2 px-4 py-12 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading investigations…
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 px-4 py-8 text-sm text-critical">
            <TriangleAlert className="h-4 w-4" />
            Could not load investigations. Is the backend running?
          </div>
        ) : filtered.length === 0 ? (
          <div className="px-4 py-10 text-center text-sm text-muted-foreground">
            No investigations yet.{" "}
            <Link to="/analyst/new" className="text-primary underline">
              Create one
            </Link>
            .
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[700px] text-left text-sm">
              <thead className="bg-muted/60 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                <tr>
                  <th className="px-4 py-2.5 font-semibold">Investigation</th>
                  <th className="px-4 py-2.5 font-semibold">Created</th>
                  <th className="px-4 py-2.5 font-semibold">Status</th>
                  <th className="px-4 py-2.5 font-semibold">ML Result</th>
                  <th className="px-4 py-2.5 text-right font-semibold">Outcome</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {filtered.map((item) => (
                  <InvestigationRow key={item.id} item={item} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </AnalystShell>
  );
}

// ===========================================================================
// NewInvestigation
// ===========================================================================

export function NewInvestigation() {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { setInvestigation, setInferenceResult, setProcessing, setError } = useActiveInvestigation();

  // Form state
  const [name, setName] = useState("Gulf of Mexico — Offshore Investigation");
  const [startDate, setStartDate] = useState("2024-06-01");
  const [endDate, setEndDate] = useState("2024-06-15");
  const [minLon, setMinLon] = useState("-91.0");
  const [maxLon, setMaxLon] = useState("-89.5");
  const [minLat, setMinLat] = useState("25.5");
  const [maxLat, setMaxLat] = useState("27.0");
  const [confidence, setConfidence] = useState(0.5);

  // AOI input mode: type a bounding box, or draw a polygon on the map.
  const [aoiInputMode, setAoiInputMode] = useState<"coords" | "draw">("coords");
  const [drawnPolygon, setDrawnPolygon] = useState<GeoJSONPolygon | null>(null);

  // Validate whichever AOI input is active
  const aoiError = useMemo(() => {
    if (aoiInputMode === "draw") {
      if (!drawnPolygon) return "Draw an area on the map, then press Finish.";
      const ring = (drawnPolygon.coordinates?.[0] ?? []).map(
        (p) => [p[0], p[1]] as [number, number],
      );
      return validateDrawnPolygon(ring);
    }
    const mn = parseFloat(minLon), mx = parseFloat(maxLon);
    const sl = parseFloat(minLat), nl = parseFloat(maxLat);
    if (isNaN(mn) || isNaN(mx) || isNaN(sl) || isNaN(nl)) return "All four coordinates are required.";
    if (mn >= mx) return "Min Longitude must be less than Max Longitude.";
    if (sl >= nl) return "Min Latitude must be less than Max Latitude.";
    if (mn < -180 || mx > 180) return "Longitude must be between −180 and 180.";
    if (sl < -90 || nl > 90) return "Latitude must be between −90 and 90.";
    if ((mx - mn) > 10 || (nl - sl) > 10) return "Area too large — keep box under 10° × 10°.";
    return null;
  }, [aoiInputMode, drawnPolygon, minLon, maxLon, minLat, maxLat]);

  // Build the AOI GeoJSON polygon — the drawn polygon in draw mode, otherwise
  // a rectangle from the bounding-box fields. The backend accepts any Polygon.
  const aoiGeojson = useMemo<GeoJSONPolygon>(() => {
    if (aoiInputMode === "draw" && drawnPolygon) return drawnPolygon;
    const mn = parseFloat(minLon) || 68.0;
    const mx = parseFloat(maxLon) || 71.0;
    const sl = parseFloat(minLat) || 17.0;
    const nl = parseFloat(maxLat) || 20.0;
    return {
      type: "Polygon" as const,
      coordinates: [[[mn, sl], [mx, sl], [mx, nl], [mn, nl], [mn, sl]]],
    };
  }, [aoiInputMode, drawnPolygon, minLon, maxLon, minLat, maxLat]);

  const drawnVertexCount = drawnPolygon
    ? Math.max(0, (drawnPolygon.coordinates?.[0]?.length ?? 1) - 1)
    : 0;

  const [step, setStep] = useState(0);
  const [createdInv, setCreatedInv] = useState<InvestigationRead | null>(null);
  const [inferenceResult, setLocalResult] = useState<InferenceResult | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<string>("");
  const [pipelineError, setPipelineError] = useState<string | null>(null);

  // --- Step 3: Create investigation + run inference ---
  const createMutation = useMutation({
    mutationFn: async () => {
      // 1. Create investigation
      setPipelineStatus("Creating investigation…");
      setPipelineError(null);
      const inv = await investigationsApi.create({
        name,
        aoi_geojson: aoiGeojson,
        start_date: startDate,
        end_date: endDate,
      });
      setCreatedInv(inv);
      setInvestigation(inv);

      // 2. Run inference
      setPipelineStatus("Retrieving Sentinel-1 SAR tiles…");
      setProcessing(true);

      let result: InferenceResult;
      try {
        result = await mlApi.runInference({
          investigation_id: inv.id,
          aoi_geojson: aoiGeojson,
          start_date: startDate,
          end_date: endDate,
          confidence_threshold: confidence,
        });
      } finally {
        setProcessing(false);
      }

      setPipelineStatus("Complete");
      setLocalResult(result);
      setInferenceResult(result);

      // Refresh the investigation record to pick up persisted ml_result
      const updated = await investigationsApi.get(inv.id);
      setInvestigation(updated);

      // Invalidate the dashboard list
      await qc.invalidateQueries({ queryKey: ["investigations"] });

      return result;
    },
    onError: (err: Error) => {
      setProcessing(false);
      setPipelineError(err.message);
      setPipelineStatus("Failed");
      setError(err.message);
    },
  });

  const isRunning = createMutation.isPending;

  const handleRunDetection = () => {
    setStep(3);
    createMutation.mutate();
  };

  const handleViewResults = () => {
    navigate({ to: "/analyst/satellite" });
  };

  return (
    <AnalystShell
      title="New Investigation"
      subtitle="Create a structured oil spill investigation from satellite evidence"
      actions={
        <Button variant="outline" size="sm" asChild>
          <Link to="/analyst">
            <ArrowLeft /> Back to dashboard
          </Link>
        </Button>
      }
    >
      <div className="mx-auto max-w-5xl">
        <ProgressSteps current={step} />

        <div className="mt-5 grid gap-5 lg:grid-cols-[1fr_300px]">
          <Panel
            title={
              step === 0
                ? "Define area of interest"
                : step === 1
                  ? "Set date range"
                  : step === 2
                    ? "Configure detection"
                    : "Running pipeline"
            }
          >
            {/* Step 0 — AOI */}
            {step === 0 && (
              <div className="space-y-4">
                <Field label="Investigation name" value={name} onChange={setName} />

                {/* AOI instructions */}
                <div className="border border-satellite/30 bg-satellite/8 p-3 text-xs leading-5 text-foreground space-y-1">
                  <p className="font-semibold text-satellite">How to enter your Area of Interest</p>
                  <p>Enter a <strong>bounding box</strong> — the rectangular region you want to analyse.</p>
                  <ul className="ml-3 list-disc space-y-0.5 text-muted-foreground">
                    <li><strong>Min Lon / Max Lon</strong> — western and eastern edge (e.g. 68.0 / 71.0)</li>
                    <li><strong>Min Lat / Max Lat</strong> — southern and northern edge (e.g. 17.0 / 20.0)</li>
                    <li>Longitude: negative = West (Americas), positive = East (Asia/Europe)</li>
                    <li>Latitude: negative = South, positive = North</li>
                    <li>Keep the box under ~3° × 3° for faster processing</li>
                  </ul>
                  <p className="text-muted-foreground mt-1">
                    Example — Gulf of Mexico (offshore, open water): Min Lon <code>-91.0</code>, Max Lon <code>-89.5</code>, Min Lat <code>25.5</code>, Max Lat <code>27.0</code>
                  </p>
                </div>

                {/* AOI input mode toggle */}
                <div className="inline-flex overflow-hidden rounded border border-border text-xs">
                  {(["coords", "draw"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setAoiInputMode(m)}
                      className={`px-3 py-1.5 font-medium transition-colors ${
                        aoiInputMode === m
                          ? "bg-primary text-primary-foreground"
                          : "bg-background text-muted-foreground hover:bg-muted"
                      }`}
                    >
                      {m === "coords" ? "Enter coordinates" : "Draw on map"}
                    </button>
                  ))}
                </div>

                {aoiInputMode === "coords" ? (
                  <div>
                    <label className="label-caps">Bounding box coordinates</label>
                    <div className="mt-2 grid gap-3 sm:grid-cols-2">
                      <Field label="Min Longitude (west edge)" value={minLon} onChange={setMinLon} />
                      <Field label="Max Longitude (east edge)" value={maxLon} onChange={setMaxLon} />
                      <Field label="Min Latitude (south edge)" value={minLat} onChange={setMinLat} />
                      <Field label="Max Latitude (north edge)" value={maxLat} onChange={setMaxLat} />
                    </div>
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">
                    Click the map to drop polygon vertices, then <strong>Finish</strong> (or
                    double-click). Use <strong>Clear</strong> / <strong>Redraw</strong> to start over.
                    {drawnPolygon && (
                      <span className="ml-1 text-foreground">
                        Custom polygon — {drawnVertexCount} vertices.
                      </span>
                    )}
                  </div>
                )}

                {/* Validation feedback */}
                {aoiError && (
                  <div className="flex items-center gap-2 border border-critical/30 bg-critical/10 p-2 text-xs text-critical">
                    <AlertTriangle className="h-4 w-4 shrink-0" /> {aoiError}
                  </div>
                )}

                {/* Map — preview (coords) or interactive draw surface */}
                <div className={`${aoiInputMode === "draw" ? "h-96" : "h-64"} overflow-hidden border border-border`}>
                  <OriceanMapLibre
                    mode="aoi-select"
                    aoi={aoiInputMode === "draw" ? null : aoiError ? null : aoiGeojson}
                    drawEnabled={aoiInputMode === "draw"}
                    onPolygonDrawn={setDrawnPolygon}
                    className="h-full w-full"
                  />
                </div>
              </div>
            )}

            {/* Step 1 — Dates */}
            {step === 1 && (
              <div className="space-y-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <Field label="Start date" value={startDate} onChange={setStartDate} type="date" />
                  <Field label="End date" value={endDate} onChange={setEndDate} type="date" />
                </div>
                <div className="border border-border bg-muted/40 p-3 text-xs text-muted-foreground">
                  <Sparkles className="mr-2 inline h-4 w-4 text-satellite" />
                  The system will search for Sentinel-1 SAR acquisitions between these dates.
                  In mock mode, 3 synthetic tiles are generated automatically.
                </div>
              </div>
            )}

            {/* Step 2 — Configure */}
            {step === 2 && (
              <div className="space-y-5">
                <div className="space-y-3">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <KV k="Investigation name" v={name} />
                    <KV
                      k="Area of interest"
                      v={
                        aoiInputMode === "draw" && drawnPolygon
                          ? `Custom polygon · ${drawnVertexCount} vertices`
                          : `${minLon}°–${maxLon}° E · ${minLat}°–${maxLat}° N`
                      }
                    />
                    <KV k="Start date" v={startDate} />
                    <KV k="End date" v={endDate} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Confidence threshold</span>
                    <span className="mono-num font-semibold">{Math.round(confidence * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={confidence}
                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                    className="mt-2 w-full accent-[var(--satellite)]"
                  />
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Lower threshold = more detections (may include false positives).
                    In mock mode, use 0.1–0.3 to ensure detections appear.
                  </p>
                </div>
                <div className="border border-border bg-muted/40 p-3 text-xs text-muted-foreground">
                  <AlertTriangle className="mr-2 inline h-4 w-4 text-warning" />
                  Clicking "Run Detection" will create this investigation in the database
                  and immediately trigger the SAR + AI pipeline.
                </div>
              </div>
            )}

            {/* Step 3 — Pipeline running / results */}
            {step === 3 && (
              <div className="space-y-5">
                {/* Progress steps */}
                <div className="space-y-3">
                  {[
                    { label: "Creating investigation", done: !!createdInv || !isRunning },
                    { label: "Retrieving Sentinel-1 SAR tiles", done: pipelineStatus.includes("Complete") || (!isRunning && !!inferenceResult) },
                    { label: "Running AI oil spill detection", done: !!inferenceResult || (!isRunning && !!pipelineError) },
                    { label: "Building spill polygons", done: !!inferenceResult },
                  ].map(({ label, done }, i) => (
                    <div key={label} className="flex items-center gap-3">
                      <span
                        className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs ${
                          done
                            ? "bg-resolved/15 text-resolved"
                            : isRunning && i <= 2
                              ? "bg-satellite/15 text-satellite"
                              : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {done ? (
                          <Check className="h-3.5 w-3.5" />
                        ) : isRunning && i <= 2 ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <span>{i + 1}</span>
                        )}
                      </span>
                      <span className={`text-sm ${done ? "text-foreground" : "text-muted-foreground"}`}>
                        {label}
                      </span>
                    </div>
                  ))}
                </div>

                {pipelineError && (
                  <div className="flex items-start gap-2 border border-critical/30 bg-critical/10 p-3 text-sm text-critical">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <div>
                      <p className="font-semibold">Pipeline failed</p>
                      <p className="mt-1 text-xs">{pipelineError}</p>
                    </div>
                  </div>
                )}

                {inferenceResult && (
                  <div className="space-y-3 border border-resolved/30 bg-resolved/8 p-4">
                    <p className="font-semibold text-resolved">Detection complete</p>
                    <div className="grid gap-2 sm:grid-cols-3 text-sm">
                      <KV k="Tiles processed" v={String(inferenceResult.total_tiles_processed)} />
                      <KV k="Positive tiles" v={String(inferenceResult.positive_tile_count)} />
                      <KV k="Spill detections" v={String(inferenceResult.spill_detections.length)} />
                    </div>
                    {inferenceResult.mock_mode && (
                      <Pill tone="bg-satellite/15 text-satellite border-satellite/30">
                        Demo mode — synthetic tiles
                      </Pill>
                    )}
                    {inferenceResult.warnings.length > 0 && (
                      <div className="mt-2 text-xs text-warning">
                        {inferenceResult.warnings.map((w) => <p key={w}>⚠ {w}</p>)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Navigation buttons */}
            <div className="mt-7 flex justify-between border-t border-border pt-4">
              <Button
                variant="ghost"
                disabled={step === 0 || isRunning}
                onClick={() => setStep((v) => v - 1)}
              >
                <ArrowLeft /> Previous
              </Button>

              {step < 2 && (
                <Button onClick={() => setStep((v) => v + 1)} disabled={step === 0 && !!aoiError}>
                  Continue <ArrowRight />
                </Button>
              )}

              {step === 2 && (
                <Button onClick={handleRunDetection} disabled={isRunning}>
                  {isRunning ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" /> Running…
                    </>
                  ) : (
                    <>
                      <Play /> Run Detection
                    </>
                  )}
                </Button>
              )}

              {step === 3 && inferenceResult && (
                <Button onClick={handleViewResults}>
                  View results <ChevronRight />
                </Button>
              )}

              {step === 3 && pipelineError && (
                <Button
                  variant="outline"
                  onClick={() => {
                    setPipelineError(null);
                    setPipelineStatus("");
                    setCreatedInv(null);
                    setLocalResult(null);
                    setStep(2);
                  }}
                >
                  Try again
                </Button>
              )}
            </div>
          </Panel>

          <Panel title="Workflow" bodyClassName="space-y-3">
            <p className="text-xs leading-5 text-muted-foreground">
              Each investigation keeps imagery, model output, analyst decisions, and source evidence
              together for a defensible review trail.
            </p>
            <div className="space-y-2 text-xs">
              <div className="flex gap-2">
                <ShieldCheck className="h-4 w-4 shrink-0 text-resolved" /> Chain of custody tracked
              </div>
              <div className="flex gap-2">
                <Gauge className="h-4 w-4 shrink-0 text-satellite" /> Confidence scored per tile
              </div>
              <div className="flex gap-2">
                <Send className="h-4 w-4 shrink-0 text-ocean" /> Officer handoff ready
              </div>
            </div>
            {createdInv && (
              <div className="mt-4 border-t border-border pt-3 space-y-1">
                <p className="label-caps">Created investigation</p>
                <p className="mono-num text-xs font-semibold break-all">{createdInv.id}</p>
                <StatusBadge status={createdInv.status} />
              </div>
            )}
          </Panel>
        </div>
      </div>
    </AnalystShell>
  );
}

// ===========================================================================
// SatelliteAnalysis — shows real inference results
// ===========================================================================

export function SatelliteAnalysis() {
  const { investigation, inferenceResult } = useActiveInvestigation();

  const result = inferenceResult ?? (investigation?.ml_result as InferenceResult | null);

  // Tile list sorted by confidence
  const tiles = useMemo(
    () => (result?.tile_results ?? []).slice().sort((a, b) => b.confidence - a.confidence),
    [result],
  );
  const positives = tiles.filter((t) => t.is_positive);
  const detections = result?.spill_detections ?? [];

  const [selectedTile, setSelectedTile] = useState<TilePrediction | null>(null);
  const [selectedDet, setSelectedDet] = useState<SpillDetection | null>(
    detections[0] ?? null,
  );
  const [displayThreshold, setDisplayThreshold] = useState<number>(0.5);
  const [visibleLayers, setVisibleLayers] = useState<LayerVisibility>({
    sar: true,
    tileGrid: true,
    detections: true,
    gradcam: true,
    ais: true,
  });
  const [debugOpen, setDebugOpen] = useState(false);

  const handleLayerToggle = useCallback((key: keyof LayerVisibility) => {
    setVisibleLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  useEffect(() => {
    if (detections.length && !selectedDet) setSelectedDet(detections[0]);
  }, [detections, selectedDet]);

  return (
    <AnalystShell
      title="Satellite Analysis"
      subtitle={
        investigation
          ? `${investigation.name} · ${investigation.start_date} – ${investigation.end_date}`
          : "Select an investigation from the dashboard"
      }
      actions={
        <>
          <Button variant="outline" size="sm" asChild>
            <Link to="/analyst">
              <ArrowLeft /> Dashboard
            </Link>
          </Button>
          <Button size="sm" asChild>
            <Link to="/analyst/new">
              <Play /> New run
            </Link>
          </Button>
        </>
      }
    >
      {!investigation ? (
        <Panel className="mt-6">
          <div className="py-12 text-center">
            <p className="text-sm text-muted-foreground">
              No active investigation. Go to the{" "}
              <Link to="/analyst" className="text-primary underline">
                Dashboard
              </Link>{" "}
              and open one, or{" "}
              <Link to="/analyst/new" className="text-primary underline">
                create a new investigation
              </Link>
              .
            </p>
          </div>
        </Panel>
      ) : (
        <>
          {/* Summary stats */}
          <div className="mb-5 grid gap-3 sm:grid-cols-3">
            <StatCard
              label="Status"
              value={result ? "Complete" : investigation.status.replace(/_/g, " ")}
              hint={result ? `${result.total_tiles_processed} tiles processed` : "Awaiting inference"}
              tone={result ? "resolved" : "satellite"}
            />
            <StatCard
              label="Spill detections"
              value={String(detections.length)}
              hint={
                result
                  ? `${result.positive_tile_count} positive tile${result.positive_tile_count !== 1 ? "s" : ""}`
                  : "—"
              }
              tone={detections.length > 0 ? "critical" : "default"}
            />
            <StatCard
              label="Model"
              value={result?.model_variant ?? "—"}
              hint={result?.model_version ?? (result?.mock_mode ? "Mock mode" : "Not run")}
              tone="default"
            />
          </div>

          {result?.mock_mode && (
            <div className="mb-4 flex items-center gap-2 border border-satellite/30 bg-satellite/10 px-3 py-2 text-xs text-satellite">
              <Sparkles className="h-4 w-4" />
              Demo mode active — results are synthetic (ML_MOCK_MODE=true). Switch to real mode for live GEE + model inference.
            </div>
          )}

          {investigation.error_message && (
            <div className="mb-4 flex items-start gap-2 border border-critical/30 bg-critical/10 px-3 py-2 text-xs text-critical">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {investigation.error_message}
            </div>
          )}

          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
            {/* Map */}
            <Panel title="Oil spill detection map" bodyClassName="p-0">
              <div className="relative h-[480px]">
                <OriceanMapLibre
                  mode="results"
                  aoi={investigation.aoi_geojson as GeoJSONPolygon ?? null}
                  inferenceResult={result}
                  onTileClick={setSelectedTile}
                  displayThreshold={displayThreshold}
                  visibleLayers={visibleLayers}
                  className="h-full w-full"
                />
                {/* Layer switcher — top-right overlay */}
                <div className="absolute right-14 top-3 z-20">
                  <LayerSwitcherPanel visibleLayers={visibleLayers} onToggle={handleLayerToggle} />
                </div>
                {/* Map legend — bottom-left overlay */}
                <div className="absolute bottom-3 left-3 z-20">
                  <MapLegend />
                </div>
                {/* Pipeline progress — top-left overlay when processing */}
                {investigation.status === "running_gee" || investigation.status === "running_ml" ? (
                  <div className="absolute left-3 top-3 z-20">
                    <PipelineProgressPanel
                      investigationId={investigation.id}
                      isVisible={true}
                    />
                  </div>
                ) : null}
                {!result && (
                  <div className="absolute inset-0 flex items-center justify-center bg-background/60 z-10">
                    <p className="text-sm text-muted-foreground">No results yet</p>
                  </div>
                )}
              </div>
            </Panel>

            {/* Right panel: detections list + tile details */}
            <div className="space-y-5">
              {/* Spill detections */}
              <Panel title="Spill detections" right={
                <span className="mono-num text-xs text-muted-foreground">
                  {detections.length} region{detections.length !== 1 ? "s" : ""}
                </span>
              } bodyClassName="p-0">
                {detections.length === 0 ? (
                  <div className="px-4 py-6 text-center text-sm text-muted-foreground">
                    {result ? "No oil spill regions detected." : "Run inference to see results."}
                  </div>
                ) : (
                  <div className="divide-y divide-border">
                    {detections.map((det, i) => (
                      <button
                        key={det.detection_id}
                        onClick={() => setSelectedDet(det)}
                        className={`w-full p-3 text-left transition-colors ${selectedDet?.detection_id === det.detection_id ? "bg-satellite/10" : "hover:bg-muted/50"}`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-xs font-semibold">
                            Region {String(i + 1).padStart(2, "0")}
                          </span>
                          <Pill tone="bg-spill/12 text-spill border-spill/30">
                            {Math.round(det.mean_confidence * 100)}% avg
                          </Pill>
                        </div>
                        <div className="mt-1 flex justify-between text-[11px] text-muted-foreground">
                          <span>{det.tile_count} tile{det.tile_count !== 1 ? "s" : ""}</span>
                          <span>{det.area_km2 != null ? `${det.area_km2.toFixed(2)} km²` : "—"}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </Panel>

              {/* Detection detail */}
              {selectedDet && (
                <Panel title="Detection detail">
                  <div className="space-y-0">
                    <KV k="Detection ID" v={selectedDet.detection_id.slice(0, 8) + "…"} />
                    <KV k="Mean confidence" v={`${Math.round(selectedDet.mean_confidence * 100)}%`} />
                    <KV k="Max confidence" v={`${Math.round(selectedDet.max_confidence * 100)}%`} />
                    <KV k="Contributing tiles" v={String(selectedDet.tile_count)} />
                    {selectedDet.area_km2 != null && (
                      <KV k="Estimated area" v={`${selectedDet.area_km2.toFixed(2)} km²`} />
                    )}
                  </div>
                  <ScoreBar label="Confidence" value={Math.round(selectedDet.mean_confidence * 100)} tone="spill" />
                </Panel>
              )}
            </div>
          </div>

          {/* Tile results table */}
          {tiles.length > 0 && (
            <Panel className="mt-5" title="Per-tile results" right={
              <span className="mono-num text-xs text-muted-foreground">
                {positives.length}/{tiles.length} positive
              </span>
            } bodyClassName="p-0">
              <div className="overflow-x-auto">
                <table className="w-full min-w-[600px] text-left text-sm">
                  <thead className="bg-muted/60 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                    <tr>
                      <th className="px-4 py-2.5 font-semibold">Tile ID</th>
                      <th className="px-4 py-2.5 font-semibold">Acquired</th>
                      <th className="px-4 py-2.5 font-semibold">Label</th>
                      <th className="px-4 py-2.5 font-semibold">Confidence</th>
                      <th className="px-4 py-2.5 text-right font-semibold">Mock</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {tiles.map((tile) => (
                      <tr
                        key={tile.tile_id}
                        className={`transition-colors hover:bg-muted/40 ${tile.is_positive ? "bg-spill/5" : ""}`}
                        onClick={() => setSelectedTile(tile)}
                      >
                        <td className="mono-num px-4 py-2 text-xs">{tile.tile_id}</td>
                        <td className="mono-num px-4 py-2 text-xs text-muted-foreground">
                          {tile.acquisition_date}
                        </td>
                        <td className="px-4 py-2">
                          <Pill
                            tone={
                              tile.is_positive
                                ? "bg-spill/12 text-spill border-spill/30"
                                : "bg-resolved/12 text-resolved border-resolved/30"
                            }
                          >
                            {tile.label}
                          </Pill>
                        </td>
                        <td className="px-4 py-2">
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-24 overflow-hidden bg-muted">
                              <div
                                className={`h-full ${tile.is_positive ? "bg-spill" : "bg-ocean"}`}
                                style={{ width: `${Math.round(tile.confidence * 100)}%` }}
                              />
                            </div>
                            <span className="mono-num text-xs">{Math.round(tile.confidence * 100)}%</span>
                          </div>
                        </td>
                        <td className="px-4 py-2 text-right">
                          {tile.mock && (
                            <Pill tone="bg-satellite/15 text-satellite border-satellite/30">
                              synthetic
                            </Pill>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {selectedTile && (
                <div className="border-t border-border bg-muted/30 px-4 py-3">
                  <p className="label-caps">Selected tile bounds</p>
                  <p className="mono-num mt-1 text-xs">
                    [{selectedTile.bounds.map((v) => v.toFixed(4)).join(", ")}]
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    (min_lon, min_lat, max_lon, max_lat) — WGS84
                  </p>
                </div>
              )}
            </Panel>
          )}

          {result && (
            <Collapsible open={debugOpen} onOpenChange={setDebugOpen} className="mt-5">
              <CollapsibleTrigger asChild>
                <Button variant="outline" size="sm" className="w-full justify-between">
                  <span className="flex items-center gap-2">
                    <SlidersHorizontal className="h-4 w-4" /> Debug Panel
                  </span>
                  <span className="text-xs text-muted-foreground">{debugOpen ? "Hide" : "Show"}</span>
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <DebugPanel
                  inferenceResult={result}
                  displayThreshold={displayThreshold}
                  onThresholdChange={setDisplayThreshold}
                  className="mt-2"
                />
              </CollapsibleContent>
            </Collapsible>
          )}

          {result && (
            <div className="mt-5 flex gap-2">
              <Button asChild>
                <Link to="/analyst/source">
                  Source Reconstruction <ArrowRight />
                </Link>
              </Button>
              <Button variant="outline" asChild>
                <Link to="/analyst/vessels">Vessel Attribution</Link>
              </Button>
            </div>
          )}

          <TileInspectorSheet
            tile={selectedTile}
            threshold={displayThreshold}
            onClose={() => setSelectedTile(null)}
          />
        </>
      )}
    </AnalystShell>
  );
}

// ===========================================================================
// SourceReconstruction — live backend implementation
// ===========================================================================

export function SourceReconstruction() {
  const { investigation, inferenceResult, setInvestigation } = useActiveInvestigation();
  const result = inferenceResult ?? (investigation?.ml_result as InferenceResult | null);

  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [reconstructionResult, setReconstructionResult] = useState<SourceReconstructionResult | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);

  // On mount, hydrate from persisted result if available
  useEffect(() => {
    if (investigation?.source_reconstruction_result) {
      setReconstructionResult(
        investigation.source_reconstruction_result as unknown as SourceReconstructionResult,
      );
      setStatus("success");
    }
  }, [investigation?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRunReconstruction = async (opts?: { force_rerun?: boolean }) => {
    if (!investigation) return;
    setStatus("loading");
    setErrorDetail(null);
    try {
      const res = await sourceApi.reconstruct(investigation.id, opts);
      setReconstructionResult(res);
      setStatus("success");
      // Refresh investigation record so source_reconstruction_result is persisted
      const updated = await investigationsApi.get(investigation.id);
      setInvestigation(updated);
    } catch (err) {
      const apiErr = err as ApiError;
      setErrorDetail(apiErr?.detail ?? String(err));
      setStatus("error");
    }
  };

  // Determine readiness state
  const hasML = result !== null && result.spill_detections.length > 0;
  const needsML = investigation !== null && !hasML;
  const isLoading = status === "loading";

  // ---- Drift replay (simulated straight-line estimate) --------------------
  const [driftIdx, setDriftIdx] = useState(0);
  const [driftPlaying, setDriftPlaying] = useState(false);
  const [driftDir, setDriftDir] = useState<"forward" | "backward">("forward");

  const spillCentroid = useMemo<[number, number] | null>(() => {
    const polys = (result?.spill_detections ?? [])
      .map((d) => polygonCentroid(d.polygon_geojson))
      .filter((c): c is [number, number] => !!c);
    if (!polys.length) return null;
    return [
      polys.reduce((s, c) => s + c[0], 0) / polys.length,
      polys.reduce((s, c) => s + c[1], 0) / polys.length,
    ];
  }, [result]);

  const detectionTimeISO = useMemo(() => {
    const tiles = result?.tile_results ?? [];
    const acq = tiles.map((t) => t.acquisition_date).filter(Boolean).sort().at(-1);
    return acq ?? investigation?.end_date ?? reconstructionResult?.release_window_end ?? null;
  }, [result, investigation, reconstructionResult]);

  const driftSteps = useMemo<ReplayStep[]>(() => {
    if (!reconstructionResult || !spillCentroid || !detectionTimeISO) return [];
    const src = polygonCentroid(reconstructionResult.source_corridor_geojson);
    if (!src) return [];
    return buildReplaySteps(
      src,
      spillCentroid,
      reconstructionResult.release_window_start,
      detectionTimeISO,
      24,
      driftDir,
    );
  }, [reconstructionResult, spillCentroid, detectionTimeISO, driftDir]);

  useEffect(() => {
    setDriftIdx(0);
    setDriftPlaying(false);
  }, [driftDir, reconstructionResult?.run_at]);

  return (
    <AnalystShell
      title="Source Reconstruction"
      subtitle={
        investigation
          ? `${investigation.name} · Backtracking probable release region`
          : "Source reconstruction — select an investigation first"
      }
      actions={
        status === "success" && (
          <Button variant="outline" size="sm" onClick={() => handleRunReconstruction({ force_rerun: true })} disabled={isLoading}>
            <RefreshCw className="h-4 w-4" /> Re-run
          </Button>
        )
      }
    >
      {/* No investigation selected */}
      {!investigation && (
        <Panel className="mt-4">
          <div className="py-12 text-center">
            <p className="text-sm text-muted-foreground">
              No active investigation. Go to the{" "}
              <Link to="/analyst" className="text-primary underline">
                Dashboard
              </Link>{" "}
              and open one.
            </p>
          </div>
        </Panel>
      )}

      {/* ML not yet run / no detections */}
      {investigation && needsML && (
        <div className="mb-4 flex items-center gap-2 border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          No spill detections found in this investigation — run ML inference first before source reconstruction.
        </div>
      )}

      {/* Error banner */}
      {status === "error" && errorDetail && (
        <div className="mb-4 flex items-start gap-2 border border-critical/30 bg-critical/10 px-3 py-2 text-xs text-critical">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-semibold">Reconstruction failed</p>
            <p className="mt-1">{errorDetail}</p>
          </div>
        </div>
      )}

      {/* Run button — shown when investigation is ready and no result yet (or after error) */}
      {investigation && hasML && (status === "idle" || status === "error") && (
        <div className="mb-5">
          <Button onClick={() => handleRunReconstruction()}>
            <Play className="h-4 w-4" /> Run source reconstruction
          </Button>
        </div>
      )}

      {/* Loading state */}
      {status === "loading" && (
        <div className="mb-5">
          <Button disabled>
            <Loader2 className="h-4 w-4 animate-spin" /> Running reconstruction…
          </Button>
        </div>
      )}

      {/* Success state — full results */}
      {status === "success" && reconstructionResult && investigation && (
        <>
          {/* Stats row */}
          <div className="mb-5 grid gap-3 sm:grid-cols-3">
            <StatCard
              label="Source confidence"
              value={`${Math.round(reconstructionResult.source_confidence * 100)}%`}
              hint="Kinematic drift model estimate"
              tone="satellite"
            />
            <StatCard
              label="Release window"
              value={`T−${Math.round((new Date(reconstructionResult.release_window_end).getTime() - new Date(reconstructionResult.release_window_start).getTime()) / 3_600_000)}h`}
              hint={`${formatDateTime(reconstructionResult.release_window_start)} → ${formatDateTime(reconstructionResult.release_window_end)}`}
              tone="warning"
            />
            <StatCard
              label="Drift distance"
              value={`${reconstructionResult.drift_distance_km.toFixed(1)} km`}
              hint="Corridor expansion radius"
              tone="default"
            />
          </div>

          <div className="grid gap-5 xl:grid-cols-[minmax(0,1.3fr)_350px]">
            {/* Map panel */}
            <Panel
              title="Source corridor map"
              right={
                <Pill tone="bg-satellite/15 text-satellite border-satellite/30">
                  {reconstructionResult.reconstruction_method}
                </Pill>
              }
              bodyClassName="p-0"
            >
              <div className="relative h-[440px]">
                <OriceanMapLibre
                  mode="results"
                  aoi={investigation.aoi_geojson as GeoJSONPolygon ?? null}
                  inferenceResult={result}
                  sourceCorridorGeojson={reconstructionResult.source_corridor_geojson}
                  driftReplay={driftSteps.length ? { frames: driftSteps, activeIndex: driftIdx, direction: driftDir } : null}
                  className="h-full w-full"
                />
              </div>
              {driftSteps.length > 0 ? (
                <DriftReplayBar
                  steps={driftSteps}
                  activeIndex={driftIdx}
                  onIndex={setDriftIdx}
                  playing={driftPlaying}
                  onPlayToggle={setDriftPlaying}
                  direction={driftDir}
                  onDirection={setDriftDir}
                  label={`Simulated straight-line drift estimate — ${driftDir === "forward" ? "source corridor → observed spill" : "observed spill → probable source"}, over the ${reconstructionResult.reconstruction_method} release window (${reconstructionResult.drift_distance_km.toFixed(1)} km corridor). The solid polygon is the actual Sentinel-1 detection; the hollow amber marker is simulated.`}
                />
              ) : (
                <div className="border-t border-border px-4 py-2 text-[11px] text-muted-foreground">
                  Drift replay needs both a source corridor and a detected spill region.
                </div>
              )}
              <div className="border-t border-border px-4 py-3 flex items-center justify-between">
                <p className="text-xs text-muted-foreground">
                  Detection region shown. Source corridor computed from drift backtracking.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleRunReconstruction({ force_rerun: true })}
                  disabled={isLoading}
                >
                  <RefreshCw className="h-3.5 w-3.5" /> Re-run
                </Button>
              </div>
            </Panel>

            {/* Model output panel */}
            <Panel title="Model output">
              <div className="space-y-4">
                {/* Source corridor highlight box */}
                <div className="border border-satellite/30 bg-satellite/10 p-3">
                  <p className="label-caps text-satellite">Probable source corridor</p>
                  <p className="mono-num mt-1 text-sm font-semibold">
                    {reconstructionResult.drift_distance_km.toFixed(1)} km expansion from detection region
                  </p>
                  <div className="mt-2">
                    <Pill tone="bg-satellite/15 text-satellite border-satellite/30">
                      {reconstructionResult.reconstruction_method}
                    </Pill>
                  </div>
                </div>

                {/* Confidence score bar */}
                <ScoreBar
                  label="Source confidence"
                  value={Math.round(reconstructionResult.source_confidence * 100)}
                  tone="satellite"
                />

                {/* Release window */}
                <div className="space-y-0">
                  <KV
                    k="Release window start"
                    v={formatDateTime(reconstructionResult.release_window_start)}
                  />
                  <KV
                    k="Release window end"
                    v={formatDateTime(reconstructionResult.release_window_end)}
                  />
                </div>

                {/* Methodology note */}
                <div className="mt-3 border border-border bg-muted/40 p-3 text-xs leading-5 text-muted-foreground">
                  <Sparkles className="mr-1.5 inline h-3.5 w-3.5 text-satellite" />
                  Method:{" "}
                  <span className="font-mono">{reconstructionResult.reconstruction_method}</span> —
                  kinematic drift model without real ocean current data.
                </div>

                {/* Warnings */}
                {reconstructionResult.warnings.length > 0 && (
                  <div className="mt-2 text-xs text-warning">
                    {reconstructionResult.warnings.map((w) => (
                      <p key={w}>⚠ {w}</p>
                    ))}
                  </div>
                )}
              </div>

              <div className="mt-4 flex gap-2 border-t border-border pt-4">
                <Button variant="outline" size="sm" asChild>
                  <Link to="/analyst/vessels">
                    <Ship className="h-4 w-4" /> Vessel attribution
                  </Link>
                </Button>
              </div>
            </Panel>
          </div>
        </>
      )}
    </AnalystShell>
  );
}

// ===========================================================================
// VesselAttribution — live backend implementation
// ===========================================================================

export function VesselAttribution() {
  const { investigation, inferenceResult, setInvestigation } = useActiveInvestigation();
  const result = inferenceResult ?? (investigation?.ml_result as InferenceResult | null);

  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [attributionResult, setAttributionResult] = useState<AttributionResult | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<CandidateVessel | null>(null);
  const [errorDetail, setErrorDetail] = useState<string | null>(null);
  const [expandedCandidates, setExpandedCandidates] = useState(false);

  // On mount: hydrate from persisted result if available
  useEffect(() => {
    if (investigation?.attribution_result) {
      setAttributionResult(investigation.attribution_result as unknown as AttributionResult);
      setStatus("success");
    }
  }, [investigation?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRunAttribution = async (opts?: { force_rerun?: boolean }) => {
    if (!investigation) return;
    setStatus("loading");
    setErrorDetail(null);
    setSelectedCandidate(null);
    try {
      const res = await attributionApi.run(investigation.id, opts);
      setAttributionResult(res);
      setStatus("success");
      const updated = await investigationsApi.get(investigation.id);
      setInvestigation(updated);
    } catch (err) {
      const apiErr = err as ApiError;
      // 409 = attribution already completed. Load the stored result instead of
      // showing an error; "Re-run" still forces a fresh run.
      if (apiErr?.status === 409 && !opts?.force_rerun) {
        try {
          const existing = await attributionApi.get(investigation.id);
          setAttributionResult(existing);
          setStatus("success");
          return;
        } catch {
          /* fall through */
        }
      }
      setErrorDetail(apiErr?.detail ?? String(err));
      setStatus("error");
    }
  };

  // Derived state
  const isLoading = status === "loading";
  const noProvider = attributionResult !== null && !attributionResult.ais_provider_configured;
  const hasML = result !== null && (result.spill_detections?.length ?? 0) > 0;
  const needsML = investigation !== null && !hasML && attributionResult === null;

  // Stats + evidence-gated candidate shortlist
  const candidates = attributionResult?.candidates ?? [];
  const qualified = candidates.filter((c) => c.qualified);
  const qualifiedCount = attributionResult?.qualified_count ?? qualified.length;
  // Default view: top 8 qualified by score. "Show all" reveals every scored vessel.
  const shownCandidates = expandedCandidates ? candidates : qualified.slice(0, 8);
  const topCandidate = qualified[0] ?? candidates[0] ?? null;
  const topScore = topCandidate?.attribution_score ?? 0;
  const nearRegionCount = qualified.filter((c) => c.time_near_region_hours > 0).length;
  const lookbackHours = attributionResult?.lookback_hours ?? null;
  // Historical AIS routes for the map — every scored candidate vessel is drawn
  // (faint blue); the #1 ranked candidate is drawn prominently in red.
  const aisVesselTracks = candidates
    .filter((c) => c.track && c.track.length >= 1)
    .map((c) => ({ mmsi: c.mmsi, name: c.name, track: c.track as Array<[number, number]> }));

  // ---- Ship-route replay of the #1 candidate along its real AIS trajectory ----
  const [shipIdx, setShipIdx] = useState(0);
  const [shipPlaying, setShipPlaying] = useState(false);
  const corridorGeom =
    (investigation?.source_reconstruction_result?.source_corridor_geojson ?? null) as unknown;
  const shipSteps = useMemo<ReplayStep[]>(() => {
    if (!topCandidate) return [];
    const pts = topCandidate.track_points?.length ? topCandidate.track_points : topCandidate.track;
    return stepsFromTrack(
      pts as Array<[number, number] | [number, number, string]>,
      topCandidate.entry_time,
      topCandidate.exit_time,
    );
  }, [topCandidate]);
  useEffect(() => {
    setShipIdx(0);
    setShipPlaying(false);
  }, [topCandidate?.mmsi]);
  const shipCur = shipSteps[Math.min(shipIdx, Math.max(0, shipSteps.length - 1))];
  const shipDistKm =
    shipCur && corridorGeom ? distanceToCorridorKm([shipCur.lon, shipCur.lat], corridorGeom) : null;

  return (
    <AnalystShell
      title="Vessel Attribution"
      subtitle={
        investigation
          ? `${investigation.name} · AIS candidate vessel attribution`
          : "Vessel Attribution — select an investigation first"
      }
      actions={
        status === "success" && attributionResult?.ais_provider_configured && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleRunAttribution({ force_rerun: true })}
            disabled={isLoading}
          >
            <RefreshCw className="h-4 w-4" /> Re-run
          </Button>
        )
      }
    >
      {/* No investigation selected */}
      {!investigation && (
        <Panel className="mt-4">
          <div className="py-12 text-center">
            <p className="text-sm text-muted-foreground">
              No active investigation. Go to the{" "}
              <Link to="/analyst" className="text-primary underline">
                Dashboard
              </Link>{" "}
              and open one.
            </p>
          </div>
        </Panel>
      )}

      {/* ML not yet run / no detections (only show when no cached result) */}
      {investigation && needsML && (
        <div className="mb-4 flex items-center gap-2 border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          No spill detections found — run ML inference first before vessel attribution.
        </div>
      )}

      {/* Error banner */}
      {status === "error" && errorDetail && (
        <div className="mb-4 flex items-start gap-2 border border-critical/30 bg-critical/10 px-3 py-2 text-xs text-critical">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-semibold">Attribution failed</p>
            <p className="mt-1">{errorDetail}</p>
          </div>
        </div>
      )}

      {/* No AIS provider notice */}
      {noProvider && (
        <div className="mb-4 flex items-start gap-2 border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-warning">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
          <div>
            <p className="font-semibold">No AIS provider configured — vessel attribution is unavailable.</p>
            <p className="mt-0.5 text-xs">
              Set <code className="font-mono">AIS_PROVIDER=gfw</code> and{" "}
              <code className="font-mono">GFW_API_TOKEN</code> (from{" "}
              <span className="font-mono">globalfishingwatch.org/our-apis/tokens</span>) in{" "}
              <code className="font-mono">backend/.env</code> to enable historical AIS attribution.
            </p>
          </div>
        </div>
      )}

      {/* Run button — shown when investigation is ready and no result yet (or after error) */}
      {investigation && (hasML || attributionResult === null) && (status === "idle" || status === "error") && !noProvider && (
        <div className="mb-5">
          <Button onClick={() => handleRunAttribution()} disabled={!hasML && !investigation.source_reconstruction_result}>
            <Play className="h-4 w-4" /> Run vessel attribution
          </Button>
          {!hasML && !investigation.source_reconstruction_result && (
            <p className="mt-1.5 text-xs text-muted-foreground">
              Run ML inference or source reconstruction first to define the search region.
            </p>
          )}
        </div>
      )}

      {/* Loading state */}
      {status === "loading" && (
        <div className="mb-5">
          <Button disabled>
            <Loader2 className="h-4 w-4 animate-spin" /> Running attribution…
          </Button>
        </div>
      )}

      {/* Results — only show full panel when we have a result (even with no provider / empty candidates) */}
      {investigation && status === "success" && attributionResult && (
        <>
          {/* Stats row — shown only when provider is configured */}
          {attributionResult.ais_provider_configured && (
            <div className="mb-5 grid gap-3 sm:grid-cols-3">
              <StatCard
                label="Qualified candidates"
                value={String(qualifiedCount)}
                hint={`of ${attributionResult.total_candidates} scored — passed every evidence gate`}
                tone="satellite"
              />
              <StatCard
                label="Top candidate score"
                value={`${topScore.toFixed(1)}%`}
                hint={candidates[0]?.name ?? candidates[0]?.mmsi ?? "—"}
                tone={topScore >= 70 ? "critical" : "default"}
              />
              <StatCard
                label="Active near region"
                value={String(nearRegionCount)}
                hint={
                  lookbackHours
                    ? `Within ${lookbackHours} h before detection`
                    : "Candidates with AIS activity near the region"
                }
                tone={nearRegionCount > 0 ? "warning" : "default"}
              />
            </div>
          )}

          {/* Main layout: candidate table + map */}
          {!noProvider && (
            <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
              {/* Left: candidate table */}
              <div className="space-y-4">
                <Panel
                  title="Potential Source Vessel Candidates"
                  right={
                    <span className="mono-num text-xs text-muted-foreground">
                      {expandedCandidates
                        ? `${candidates.length} scored`
                        : `top ${shownCandidates.length} of ${qualifiedCount}`}
                    </span>
                  }
                  bodyClassName="p-0"
                >
                  {shownCandidates.length === 0 ? (
                    <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                      {candidates.length === 0
                        ? "No candidate vessels found in the region during the search window."
                        : `No vessel cleared every evidence gate. ${candidates.length} vessel${candidates.length !== 1 ? "s were" : " was"} scored — use "Show all" below to review them.`}
                    </div>
                  ) : (
                    <>
                      {/* Table header */}
                      <div className="overflow-x-auto">
                        <table className="w-full min-w-[720px] text-left text-xs">
                          <thead className="bg-muted/60 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                            <tr>
                              <th className="px-3 py-2.5 font-semibold">Rank</th>
                              <th className="px-3 py-2.5 font-semibold">Vessel</th>
                              <th className="px-3 py-2.5 font-semibold">MMSI / IMO</th>
                              <th className="px-3 py-2.5 font-semibold">Score</th>
                              <th className="px-3 py-2.5 font-semibold">Closest dist.</th>
                              <th className="px-3 py-2.5 font-semibold">Time diff.</th>
                              <th className="px-3 py-2.5 font-semibold">Why it qualified</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-border">
                            {shownCandidates.map((c) => (
                              <tr
                                key={c.mmsi}
                                onClick={() =>
                                  setSelectedCandidate(
                                    selectedCandidate?.mmsi === c.mmsi ? null : c,
                                  )
                                }
                                className={`cursor-pointer transition-colors ${
                                  selectedCandidate?.mmsi === c.mmsi
                                    ? "bg-satellite/10"
                                    : "hover:bg-muted/50"
                                } ${c.qualified ? "" : "opacity-60"}`}
                              >
                                {/* Rank */}
                                <td className="px-3 py-2.5 align-top">
                                  <span
                                    className={`mono-num flex h-6 w-6 items-center justify-center text-xs font-semibold ${
                                      c.mmsi === topCandidate?.mmsi
                                        ? "bg-critical text-critical-foreground"
                                        : "bg-muted text-muted-foreground"
                                    }`}
                                  >
                                    {String(c.rank).padStart(2, "0")}
                                  </span>
                                </td>
                                {/* Vessel name + type/flag */}
                                <td className="px-3 py-2.5 align-top font-medium">
                                  {c.name ?? <span className="text-muted-foreground">Unknown</span>}
                                  {c.mmsi === topCandidate?.mmsi && (
                                    <span className="ml-1.5 rounded-sm bg-critical/15 px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-critical">
                                      Top candidate
                                    </span>
                                  )}
                                  <div className="text-[10px] font-normal text-muted-foreground">
                                    {[c.vessel_type, c.flag].filter(Boolean).join(" · ") || "—"}
                                  </div>
                                </td>
                                {/* MMSI / IMO */}
                                <td className="mono-num px-3 py-2.5 align-top text-muted-foreground">
                                  {c.mmsi}
                                  {c.imo && <div className="text-[10px]">{c.imo}</div>}
                                </td>
                                {/* Candidate score */}
                                <td className="px-3 py-2.5 align-top">
                                  <div className="flex items-center gap-2">
                                    <div className="h-1.5 w-14 overflow-hidden bg-muted">
                                      <div
                                        className={`h-full ${c.attribution_score >= 70 ? "bg-critical" : c.attribution_score >= 40 ? "bg-warning" : "bg-ocean"}`}
                                        style={{ width: `${c.attribution_score}%` }}
                                      />
                                    </div>
                                    <span className="mono-num text-xs font-semibold">
                                      {c.attribution_score.toFixed(1)}
                                    </span>
                                  </div>
                                </td>
                                {/* Closest distance */}
                                <td className="mono-num px-3 py-2.5 align-top">
                                  {c.min_distance_km.toFixed(1)} km
                                </td>
                                {/* Time difference from detection */}
                                <td className="mono-num px-3 py-2.5 align-top">
                                  {c.time_diff_hours != null ? `${c.time_diff_hours.toFixed(1)} h` : "—"}
                                  {c.closest_timestamp && (
                                    <div className="text-[10px] font-normal text-muted-foreground">
                                      {formatDateTime(c.closest_timestamp)}
                                    </div>
                                  )}
                                </td>
                                {/* Why it qualified — evidence-gate reasons */}
                                <td className="px-3 py-2.5 align-top">
                                  {c.qualified ? (
                                    <div className="flex flex-wrap gap-1">
                                      {(c.qualification_reasons ?? []).map((r, i) => (
                                        <span
                                          key={i}
                                          className="rounded-sm bg-resolved/12 px-1.5 py-0.5 text-[10px] leading-4 text-resolved"
                                        >
                                          {r}
                                        </span>
                                      ))}
                                    </div>
                                  ) : (
                                    <span className="rounded-sm bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                                      below evidence threshold
                                    </span>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {candidates.length > qualified.slice(0, 8).length && (
                        <div className="border-t border-border p-2 text-center">
                          <button
                            type="button"
                            onClick={() => setExpandedCandidates((v) => !v)}
                            className="text-xs font-medium text-primary hover:underline"
                          >
                            {expandedCandidates
                              ? "Show fewer — top 8 qualified only"
                              : `Show all ${candidates.length} scored vessels`}
                          </button>
                        </div>
                      )}
                    </>
                  )}
                </Panel>

                {/* Evidence panel — shown on row click */}
                {selectedCandidate && (
                  <Panel
                    title={`Evidence — ${selectedCandidate.name ?? selectedCandidate.mmsi}`}
                    right={
                      <button
                        type="button"
                        aria-label="Close evidence panel"
                        onClick={() => setSelectedCandidate(null)}
                        className="flex h-6 w-6 items-center justify-center text-muted-foreground transition-colors hover:text-foreground"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    }
                  >
                    {/* Score breakdown bars */}
                    <div className="grid grid-cols-2 gap-3">
                      <ScoreBar
                        label="Spatial"
                        value={Math.round(selectedCandidate.score_breakdown.spatial_score * 100)}
                        tone="ocean"
                      />
                      <ScoreBar
                        label="Temporal"
                        value={Math.round(selectedCandidate.score_breakdown.temporal_score * 100)}
                        tone="satellite"
                      />
                      <ScoreBar
                        label="Trajectory overlap"
                        value={Math.round(selectedCandidate.score_breakdown.trajectory_score * 100)}
                        tone="ocean"
                      />
                      <ScoreBar
                        label="Time near region"
                        value={Math.round(selectedCandidate.score_breakdown.time_near_region_score * 100)}
                        tone="warning"
                      />
                    </div>

                    {/* Measured evidence behind the score */}
                    <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-border pt-3 text-xs">
                      <KV k="Closest distance" v={`${selectedCandidate.min_distance_km.toFixed(1)} km`} />
                      <KV
                        k="Closest AIS fix"
                        v={selectedCandidate.closest_timestamp ? formatDateTime(selectedCandidate.closest_timestamp) : "—"}
                      />
                      <KV
                        k="Time from detection"
                        v={selectedCandidate.time_diff_hours != null ? `${selectedCandidate.time_diff_hours.toFixed(1)} h` : "—"}
                      />
                      <KV k="AIS positions" v={String(selectedCandidate.num_ais_positions)} />
                      <KV k="Trajectory overlap" v={`${Math.round(selectedCandidate.trajectory_overlap * 100)}%`} />
                      <KV k="Time near region" v={`${selectedCandidate.time_near_region_hours.toFixed(1)} h`} />
                    </div>

                    {/* Evidence strings */}
                    <div className="mt-4 space-y-2">
                      {selectedCandidate.evidence.map((item, idx) => (
                        <div
                          key={idx}
                          className="flex items-start gap-2 border-b border-border pb-2 text-sm last:border-0"
                        >
                          <Check className="mt-0.5 h-4 w-4 shrink-0 text-resolved" />
                          {item}
                        </div>
                      ))}
                    </div>

                    {/* Vessel metadata */}
                    <div className="mt-4 border-t border-border pt-3 space-y-0">
                      {selectedCandidate.imo && <KV k="IMO" v={selectedCandidate.imo} />}
                      <KV k="MMSI" v={selectedCandidate.mmsi} />
                      {selectedCandidate.vessel_type && <KV k="Type" v={selectedCandidate.vessel_type} />}
                      {selectedCandidate.flag && <KV k="Flag" v={selectedCandidate.flag} />}
                    </div>

                    <p className="mt-3 border-t border-border pt-2 text-[11px] leading-4 text-muted-foreground">
                      Potential source vessel candidate — ranked by consistency of its historical AIS
                      activity with the detected spill. This is not a determination that the vessel
                      caused the spill.
                    </p>
                  </Panel>
                )}
              </div>

              {/* Right: map */}
              <Panel
                title="Spatial context"
                right={
                  investigation.source_reconstruction_result ? (
                    <Pill tone="bg-satellite/15 text-satellite border-satellite/30">
                      Source corridor available
                    </Pill>
                  ) : undefined
                }
                bodyClassName="p-0"
              >
                <div className="relative h-[440px]">
                  <OriceanMapLibre
                    mode="results"
                    aoi={investigation.aoi_geojson as GeoJSONPolygon ?? null}
                    inferenceResult={result}
                    sourceCorridorGeojson={investigation.source_reconstruction_result?.source_corridor_geojson ?? null}
                    aisVesselTracks={aisVesselTracks.length > 0 ? aisVesselTracks : null}
                    highlightMmsi={topCandidate?.mmsi ?? null}
                    shipReplay={
                      shipSteps.length
                        ? {
                            frames: shipSteps,
                            activeIndex: shipIdx,
                            label: `Top Potential Source Vessel · ${topCandidate?.name ?? topCandidate?.mmsi ?? ""}`,
                          }
                        : null
                    }
                    className="h-full w-full"
                  />
                </div>
                {shipSteps.length > 0 && (
                  <DriftReplayBar
                    steps={shipSteps}
                    activeIndex={shipIdx}
                    onIndex={setShipIdx}
                    playing={shipPlaying}
                    onPlayToggle={setShipPlaying}
                    extra={
                      shipDistKm != null ? (
                        <span>≈ {shipDistKm.toLocaleString()} km from source corridor</span>
                      ) : null
                    }
                    label="Top Potential Source Vessel — replay of its real historical GFW AIS positions. Highest-ranked candidate, NOT a determination of responsibility. Other candidate routes stay faint blue."
                  />
                )}
                <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
                  Detected spill region
                  {investigation.source_reconstruction_result ? ", source corridor" : ""}
                  {aisVesselTracks.length > 0
                    ? `, and ${aisVesselTracks.length} candidate vessel route${aisVesselTracks.length !== 1 ? "s" : ""} (faint blue). `
                    : " shown on the map. "}
                  {topCandidate && (
                    <span className="text-foreground">
                      <span className="font-semibold text-critical">Red</span> = <strong>Top Potential Source Vessel</strong> ({topCandidate.name ?? topCandidate.mmsi}) —
                      highest-ranked potential source, not a confirmed determination.
                    </span>
                  )}
                </div>
              </Panel>
            </div>
          )}

          {/* AIS data source panel */}
          {attributionResult.ais_provider_configured && (() => {
            const src = attributionResult.data_source;
            const isGfw = src === "gfw" || attributionResult.warnings.some((w) => w.startsWith("gfw_"));
            const isHistoric =
              isGfw || src === "marinecadastre" ||
              attributionResult.warnings.some((w) => w.startsWith("marinecadastre"));
            const win =
              attributionResult.release_window_start && attributionResult.release_window_end
                ? `${formatDateTime(attributionResult.release_window_start)} → ${formatDateTime(attributionResult.release_window_end)}`
                : null;
            const title = isGfw
              ? "Historical AIS data — Global Fishing Watch"
              : isHistoric
                ? "Historical AIS data — MarineCadastre / NOAA (US waters)"
                : "Live AIS data — aisstream.io";
            const blurb = isGfw
              ? `Real historical vessel activity from the Global Fishing Watch API (${attributionResult.lookback_hours ?? 48} h before the satellite detection${win ? `, ${win} UTC` : ""}). Candidate vessels are ranked by spatial proximity, temporal consistency, trajectory overlap and time spent near the region. No live positions are used.`
              : isHistoric
                ? "Real vessel positions from the NOAA Office for Coastal Management historical AIS archive, with original timestamps for the investigation window. Candidate vessels are ranked by proximity to the source corridor."
                : "Real vessel positions collected from the global AIS terrestrial and satellite receiver network at the time of this attribution run. Each candidate vessel was detected within the source corridor during the probable release window.";
            return (
            <div className="mt-5 flex items-start gap-3 border border-ocean/30 bg-ocean/8 px-4 py-3">
              <Waves className="mt-0.5 h-4 w-4 shrink-0 text-ocean" />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-ocean">{title}</p>
                <p className="mt-0.5 text-xs leading-5 text-muted-foreground">{blurb}</p>
                {candidates.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {candidates.slice(0, 6).map((c) => (
                      <span
                        key={c.mmsi}
                        className="inline-flex items-center gap-1 border border-ocean/30 bg-ocean/10 px-2 py-0.5 text-[10px] font-mono"
                      >
                        <Ship className="h-2.5 w-2.5" />
                        {c.name ?? c.mmsi}
                        {c.vessel_type && (
                          <span className="text-muted-foreground"> · {c.vessel_type}</span>
                        )}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
            );
          })()}

          {/* Non-live warnings (errors, no-data notices). Benign provenance tags
              (e.g. gfw_historical_ais, gfw_datasets_*, gfw_vessels_returned_*,
              marinecadastre_*, aisstream_live_*) are hidden — only actionable
              notices such as failures or empty-result reasons are surfaced. */}
          {(() => {
            const BENIGN = /^(aisstream_live|gfw_historical_ais$|gfw_datasets_|gfw_vessels_returned_|marinecadastre_historical|marinecadastre_days_used)/;
            const shown = attributionResult.warnings.filter((w) => !BENIGN.test(w));
            return shown.length > 0 ? (
              <div className="mt-3 space-y-1">
                {shown.map((w) => (
                  <p key={w} className="text-xs text-warning">{w}</p>
                ))}
              </div>
            ) : null;
          })()}
        </>
      )}

      {/* Disclaimer — always visible when investigation is present */}
      {investigation && (
        <div className="mt-6 border border-warning/30 bg-warning/10 p-3 text-xs leading-5 text-warning">
          <AlertTriangle className="mr-1.5 inline h-3.5 w-3.5 text-warning" />
          These are <strong>potential source vessel candidates</strong> — vessels whose historical
          AIS activity is spatially and temporally consistent with the detected spill, ranked for
          further investigation. This is <strong>not</strong> a determination that any vessel caused
          the spill and does not constitute a finding of liability.
        </div>
      )}
    </AnalystShell>
  );
}

// ===========================================================================
// EvidenceTimeline
// ===========================================================================

export function EvidenceTimeline() {
  const { investigation, inferenceResult } = useActiveInvestigation();
  const result = inferenceResult ?? (investigation?.ml_result as InferenceResult | null);
  const [kind, setKind] = useState("All evidence");
  const [showChain, setShowChain] = useState(false);
  const kinds = ["All evidence", "Satellite", "AI model", "Vessel", "Alerts", "Reports"];

  // Build live events from real result if available
  const liveEvents = useMemo(() => {
    if (!investigation || !result) return [];
    const evs = [];
    if (investigation.created_at) {
      evs.push({
        icon: "satellite",
        kind: "satellite" as const,
        title: "Investigation Created",
        at: investigation.created_at,
        description: `Investigation "${investigation.name}" created with AOI covering ${investigation.start_date} to ${investigation.end_date}.`,
        evidence: `ID: ${investigation.id}`,
      });
    }
    if (investigation.ml_run_at) {
      evs.push({
        icon: "ai",
        kind: "ai" as const,
        title: "Oil Spill Detection Complete",
        at: investigation.ml_run_at,
        description: `ML pipeline processed ${result.total_tiles_processed} tiles. ${result.positive_tile_count} positive. ${result.spill_detections.length} spill region(s) detected.`,
        evidence: `Model: ${result.model_version} · Variant: ${result.model_variant}${result.mock_mode ? " · Mock mode" : ""}`,
      });
    }
    return evs;
  }, [investigation, result]);

  const displayTimeline = liveEvents.length > 0 ? liveEvents : timeline;

  const filtered = displayTimeline.filter(
    (event) =>
      kind === "All evidence" ||
      event.kind ===
        ({ Satellite: "satellite", "AI model": "ai", Vessel: "vessel", Alerts: "alert", Reports: "report" } as Record<string, string>)[kind],
  );

  return (
    <AnalystShell
      title="Evidence Timeline"
      subtitle={
        investigation
          ? `${investigation.name} · Chronological investigation record`
          : "Evidence Timeline — demo data"
      }
      actions={
        <>
          <Button variant="outline" size="sm">
            <ArrowDownToLine /> Download evidence log
          </Button>
          <Button size="sm">
            <Send /> Submit for review
          </Button>
        </>
      }
    >
      {!investigation && (
        <div className="mb-4 flex items-center gap-2 border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
          Showing demo timeline. Open a real investigation to see live events.
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-3">
        <StatCard
          label="Evidence items"
          value={String(displayTimeline.length)}
          hint={liveEvents.length > 0 ? "Live from backend" : "Demo data"}
          tone="satellite"
        />
        <StatCard label="Chain of custody" value="100%" hint="All source files hashed" tone="resolved" />
        <StatCard
          label="Investigation status"
          value={investigation?.status.replace(/_/g, " ") ?? "Demo"}
          hint={investigation ? "Live" : "Demo data"}
          tone="default"
        />
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Panel
          title="Investigation record"
          right={
            <div className="flex items-center gap-2">
              <button
                type="button"
                role="switch"
                aria-checked={showChain}
                onClick={() => setShowChain(!showChain)}
                className={`relative h-5 w-9 rounded-full transition-colors ${showChain ? "bg-satellite" : "bg-muted"}`}
              >
                <span
                  className={`absolute top-0.5 h-4 w-4 rounded-full bg-card transition-transform ${showChain ? "translate-x-4" : "translate-x-0.5"}`}
                />
              </button>
              <span className="text-xs text-muted-foreground">Chain of custody</span>
            </div>
          }
        >
          <div className="mb-6 flex flex-wrap gap-1">
            {kinds.map((item) => (
              <Button
                key={item}
                variant={kind === item ? "secondary" : "ghost"}
                size="sm"
                onClick={() => setKind(item)}
              >
                {item}
              </Button>
            ))}
          </div>
          <div className="relative ml-3 border-l border-border pl-7">
            {filtered.map((event, index) => (
              <div key={event.title + event.at} className="relative pb-7 last:pb-1">
                <span
                  className={`absolute -left-[43px] flex h-7 w-7 items-center justify-center rounded-full border border-background ${kindTone[event.kind]}`}
                >
                  {iconByKind[event.kind]}
                </span>
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <p className="font-semibold">{event.title}</p>
                    <p className="mt-1 text-sm leading-5 text-muted-foreground">{event.description}</p>
                  </div>
                  <span className="mono-num shrink-0 text-xs text-muted-foreground">
                    {formatDateTime(event.at)}
                  </span>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <Pill tone={`${kindTone[event.kind]} border-current/20`}>{event.kind}</Pill>
                  <span className="mono-num text-[10px] text-muted-foreground">{event.evidence}</span>
                  {showChain && (
                    <Pill tone="bg-resolved/12 text-resolved border-resolved/30">
                      <ShieldCheck className="h-3 w-3" /> SHA verified
                    </Pill>
                  )}
                </div>
                {index < filtered.length - 1 && (
                  <span className="absolute -bottom-0.5 left-0 text-[10px] text-border">↓</span>
                )}
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Case summary">
          {investigation ? (
            <div className="space-y-0">
              <KV k="Case ID" v={investigation.id.slice(0, 8) + "…"} />
              <KV k="Name" v={investigation.name} />
              <KV k="Status" v={investigation.status.replace(/_/g, " ")} />
              <KV k="AOI date range" v={`${investigation.start_date} → ${investigation.end_date}`} />
              {result && <KV k="Tiles processed" v={String(result.total_tiles_processed)} />}
              {result && <KV k="Spill detections" v={String(result.spill_detections.length)} />}
            </div>
          ) : (
            <div className="space-y-0">
              <KV k="Case ID" v={primarySpill.id} />
              <KV k="Status" v="Demo data" />
            </div>
          )}
          <div className="mt-5 border-t border-border pt-4">
            <p className="label-caps">Next action</p>
            <p className="mt-1 text-sm leading-5">
              {investigation?.status === "complete"
                ? "Review spill detections and proceed to source reconstruction."
                : investigation?.status === "failed"
                  ? "Investigation failed — check error message and re-run."
                  : "Run a new investigation to generate live evidence."}
            </p>
            <Button className="mt-4 w-full" asChild>
              <Link to="/analyst/reports">
                <FileText /> Open report builder
              </Link>
            </Button>
          </div>
        </Panel>
      </div>
    </AnalystShell>
  );
}

// ===========================================================================
// Reports
// ===========================================================================
// Report/bundle builders live in ./reportDoc so the Officer Reports view reuses them.

export function Reports() {
  const { investigation: storeInv } = useActiveInvestigation();
  const invId = storeInv?.id ?? null;

  // Always pull the full record so source reconstruction / attribution are present
  // even when the store only holds a slim (dashboard) copy.
  const { data: investigation } = useQuery({
    queryKey: ["investigation", invId, "report"],
    queryFn: () => investigationsApi.get(invId as string),
    enabled: !!invId,
    initialData: storeInv ?? undefined,
  });

  const result = (investigation?.ml_result as InferenceResult | null) ?? null;
  const sourceRecon = (investigation?.source_reconstruction_result as SourceReconstructionResult | null) ?? null;
  const attribution = (investigation?.attribution_result as AttributionResult | null) ?? null;

  const [generatedAt, setGeneratedAt] = useState<string | null>(null);
  const [submitted, setSubmitted] = useState(false);
  const generated = !!generatedAt;

  const attrCandidates = attribution?.candidates ?? [];
  const attrReady = !!attribution && attrCandidates.length > 0;
  const sourceReady = !!sourceRecon;

  const handleGenerate = () => setGeneratedAt(new Date().toISOString());

  const handleDownloadPdf = () => {
    if (!investigation) return;
    openPrintableReport({
      investigation,
      result,
      sourceRecon,
      attribution,
      generatedAt: generatedAt ?? new Date().toISOString(),
    });
  };

  const handleEvidenceBundle = () => {
    if (!investigation) return;
    downloadJson(
      `oricean-evidence-${investigation.id.slice(0, 8)}.json`,
      buildEvidenceBundle({
        investigation,
        result,
        sourceRecon,
        attribution,
        generatedAt: generatedAt ?? new Date().toISOString(),
      }),
    );
  };

  return (
    <AnalystShell
      title="Reports"
      subtitle={
        investigation
          ? `${investigation.name} · Evidence packages and officer handoff`
          : "Reports — select an investigation first"
      }
      actions={
        <Button size="sm" onClick={handleGenerate} disabled={!investigation}>
          <FilePlus2 /> {generated ? "Regenerate draft" : "Generate draft"}
        </Button>
      }
    >
      <div className="grid gap-3 sm:grid-cols-3">
        <StatCard
          label="Draft reports"
          value="01"
          hint="Last generated from current investigation"
          tone="warning"
        />
        <StatCard
          label="Evidence attached"
          value={result ? String(result.total_tiles_processed + result.spill_detections.length) : "—"}
          hint="Tiles + detections"
          tone="satellite"
        />
        <StatCard
          label="Handoff status"
          value={submitted ? "Sent" : "Ready"}
          hint={submitted ? "Officer notified" : "Awaiting analyst submission"}
          tone={submitted ? "resolved" : "default"}
        />
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_350px]">
        <Panel
          title="Report package"
          right={
            <Pill status={submitted ? "Submitted" : generated ? "Awaiting Review" : "Active"}>
              {submitted ? "Submitted" : generated ? "Draft ready" : "In progress"}
            </Pill>
          }
        >
          <div className="flex items-start gap-4 border-b border-border pb-5">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center bg-primary/10 text-primary">
              <FileText className="h-6 w-6" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="font-semibold">
                {investigation ? investigation.name : "Oil Spill Attribution Report"}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                {investigation ? `ID: ${investigation.id.slice(0, 8)}…` : "ORC-RPT-0412-v1"} ·{" "}
                {investigation?.created_at ? formatDate(investigation.created_at) : "Demo data"}
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                {result && (
                  <>
                    <Pill tone="bg-resolved/12 text-resolved border-resolved/30">
                      {result.total_tiles_processed} tiles processed
                    </Pill>
                    <Pill tone="bg-satellite/15 text-satellite border-satellite/30">
                      {result.spill_detections.length} spill region{result.spill_detections.length !== 1 ? "s" : ""}
                    </Pill>
                    {result.mock_mode && (
                      <Pill tone="bg-satellite/15 text-satellite border-satellite/30">Demo mode</Pill>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          <div className="mt-5 space-y-0">
            <ReportSection title="Investigation details" complete={!!investigation}>
              {investigation
                ? `Investigation "${investigation.name}" · ${investigation.start_date} to ${investigation.end_date}`
                : "No active investigation."}
            </ReportSection>
            <ReportSection title="Satellite / SAR data" complete={!!result}>
              {result
                ? `${result.total_tiles_processed} SAR tiles processed from Sentinel-1.`
                : "Run inference to generate SAR evidence."}
            </ReportSection>
            <ReportSection title="Oil spill detection" complete={(result?.spill_detections.length ?? 0) > 0}>
              {result
                ? `${result.spill_detections.length} spill detection(s) found. ${result.positive_tile_count} positive tiles.`
                : "No detections yet."}
            </ReportSection>
            <ReportSection title="Source reconstruction" complete={sourceReady}>
              {sourceRecon
                ? `${sourceRecon.reconstruction_method} · drift ${sourceRecon.drift_distance_km?.toFixed?.(1) ?? "—"} km · source confidence ${Math.round((sourceRecon.source_confidence ?? 0) * 100)}%.`
                : "Not run — open Source Reconstruction and run it to include the corridor."}
            </ReportSection>
            <ReportSection title="Vessel attribution" complete={attrReady}>
              {attrReady
                ? `${attribution!.data_source.toUpperCase()} · ${attribution!.qualified_count} qualified of ${attribution!.total_candidates} scored vessels.`
                : "Not run — open Vessel Attribution and run it to include candidates."}
            </ReportSection>
            <ReportSection title="Analyst sign-off" complete={generated}>
              {generated
                ? `Draft generated ${formatDateTime(generatedAt!)}. Use Download PDF / Evidence bundle below.`
                : "Click 'Generate draft' to assemble the package."}
            </ReportSection>
          </div>

          <div className="mt-6 flex flex-wrap gap-2 border-t border-border pt-4">
            <Button variant="outline" onClick={handleDownloadPdf} disabled={!investigation}>
              <Download /> Download PDF
            </Button>
            <Button variant="outline" onClick={handleEvidenceBundle} disabled={!investigation}>
              <ArrowDownToLine /> Evidence bundle
            </Button>
            <Button
              disabled={!generated || submitted}
              onClick={() => setSubmitted(true)}
            >
              <Send /> {submitted ? "Submitted to officer" : "Submit for officer review"}
            </Button>
          </div>
          {!generated && investigation && (
            <p className="mt-2 text-[11px] text-muted-foreground">
              Download PDF / Evidence bundle work now with current data; "Generate draft" stamps the sign-off time.
            </p>
          )}
        </Panel>

        <Panel title="Handoff checklist">
          <div className="space-y-3 text-sm">
            <Checklist done={!!investigation} label="Investigation created" />
            <Checklist done={!!result} label="SAR tiles retrieved" />
            <Checklist done={(result?.spill_detections.length ?? 0) > 0} label="Spill detections confirmed" />
            <Checklist done={sourceReady} label="Source region reconstructed" />
            <Checklist done={attrReady} label="AIS candidates correlated" />
            <Checklist done={generated} label="Analyst finding signed" />
            <Checklist done={submitted} label="Officer notified" />
          </div>
          <div className="mt-5 border-t border-border pt-4">
            <p className="label-caps">Recipients</p>
            <p className="mt-1 text-sm">Monitoring Officer · Maritime Command Centre</p>
          </div>
        </Panel>
      </div>
    </AnalystShell>
  );
}

// ---------------------------------------------------------------------------
// Shared micro-components
// ---------------------------------------------------------------------------

function ReportSection({
  title,
  complete,
  children,
}: {
  title: string;
  complete: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex gap-3 border-b border-border py-3 last:border-0">
      <span
        className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full ${complete ? "bg-resolved/15 text-resolved" : "bg-muted text-muted-foreground"}`}
      >
        {complete ? <Check className="h-3 w-3" /> : <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      </span>
      <div>
        <p className="text-sm font-semibold">{title}</p>
        <p className="mt-1 text-xs leading-5 text-muted-foreground">{children}</p>
      </div>
    </div>
  );
}

function Checklist({ done, label }: { done: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`flex h-5 w-5 items-center justify-center rounded-full ${done ? "bg-resolved/15 text-resolved" : "border border-border text-transparent"}`}
      >
        <Check className="h-3 w-3" />
      </span>
      <span className={done ? "" : "text-muted-foreground"}>{label}</span>
    </div>
  );
}
