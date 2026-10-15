import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  Anchor,
  ArrowRight,
  Check,
  CheckCircle2,
  Clock3,
  Download,
  FileCheck2,
  FileText,
  Filter,
  Flag,
  Layers,
  Map as MapIcon,
  MessageSquareText,
  Radio,
  RefreshCw,
  Send,
  ShieldAlert,
  Ship,
  Target,
  Users,
  Waves,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AppShell } from "@/components/oricean/AppShell";
import { MapView } from "@/components/oricean/MapView";
import { OriceanMapLibre } from "@/components/oricean/OriceanMapLibre";
import { KV, Legend, Panel, Pill, StatCard } from "@/components/oricean/bits";
import { formatDate, formatDateTime } from "@/lib/oricean-data";
import {
  DriftReplayBar,
  buildReplaySteps,
  polygonCentroid,
  type ReplayStep,
} from "@/components/oricean/DriftReplayBar";
import { openPrintableReport } from "@/components/oricean/reportDoc";
import {
  alertsApi,
  attributionApi,
  investigationsApi,
  officerApi,
  type AlertRead,
  type AttributionResult,
  type ChecklistItem,
  type GeoJSONPolygon,
  type InferenceResult,
  type InvestigationRead,
  type NotificationSettings,
  type OfficerCaseSummary,
  type SourceReconstructionResult,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Shared helpers for the live officer views
// ---------------------------------------------------------------------------

const OFFICER_SEV_PILL: Record<string, "critical" | "high" | "medium" | "low"> = {
  critical: "critical",
  high: "high",
  moderate: "medium",
  low: "low",
  test: "low",
};

function officerDetectionSummary(inv: InvestigationRead) {
  const ml = inv.ml_result as InferenceResult | null;
  const dets = ml?.spill_detections ?? [];
  const maxConf = dets.reduce((m, d) => Math.max(m, d.max_confidence ?? 0), 0);
  const area = dets.reduce((s, d) => s + (d.area_km2 ?? 0), 0);
  return {
    hasDetection: dets.length > 0 && (ml?.positive_tile_count ?? 0) > 0,
    detectionCount: dets.length,
    positiveTiles: ml?.positive_tile_count ?? 0,
    maxConfidence: maxConf,
    areaKm2: area,
  };
}

function useOfficerData() {
  const investigationsQ = useQuery({
    queryKey: ["investigations", "officer-slim"],
    queryFn: () => investigationsApi.list(1, 60, true),
    refetchInterval: 120_000,
  });
  const alertsQ = useQuery({
    queryKey: ["alerts"],
    queryFn: () => alertsApi.list(50),
    refetchInterval: 60_000,
  });
  return { investigationsQ, alertsQ };
}

function AlertPillRow({
  a,
  active,
  onSelect,
}: {
  a: AlertRead;
  active?: boolean;
  onSelect?: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full border-b border-border p-4 text-left transition-colors last:border-0 ${
        active ? "bg-satellite/10" : "hover:bg-muted/50"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold">{a.headline}</p>
          <p className="mt-1 text-xs text-muted-foreground">
            {a.region_label ?? "—"} · {a.detection_time ?? "—"}
          </p>
        </div>
        <Pill severity={OFFICER_SEV_PILL[a.severity] ?? "low"}>{a.severity}</Pill>
      </div>
      <div className="mt-3 flex items-center justify-between gap-2">
        <span
          className={`rounded-sm px-1.5 py-0.5 text-[10px] font-semibold uppercase ${
            a.delivery_status === "sent"
              ? "bg-resolved/12 text-resolved"
              : a.delivery_status === "partial"
                ? "bg-warning/12 text-warning"
                : "bg-muted text-muted-foreground"
          }`}
        >
          {a.delivery_status.replace("_", " ")}
        </span>
        <span className="mono-num text-[10px] text-muted-foreground">
          {formatDateTime(a.created_at)}
        </span>
      </div>
    </button>
  );
}

function OfficerShell({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <AppShell role="officer" title={title} subtitle={subtitle} actions={actions}>
      <div className="mx-auto w-full max-w-[1500px]">{children}</div>
    </AppShell>
  );
}


// ---------------------------------------------------------------------------
// Real-data shared helpers for the officer views
// ---------------------------------------------------------------------------

const REVIEW_LABEL: Record<string, string> = {
  pending: "Pending review", approved: "Approved for handoff",
  returned: "Returned to analyst", escalated: "Escalated",
};
const RESPONSE_LABEL: Record<string, string> = {
  not_started: "Not started", responding: "Responding",
  contained: "Contained", closed: "Closed",
};
const REPORT_LABEL: Record<string, string> = {
  draft: "Draft", awaiting_review: "Awaiting review",
  approved: "Approved", distributed: "Distributed",
};

const SEV_FILL: Record<string, string> = {
  critical: "var(--critical)", high: "var(--warning)",
  moderate: "var(--caution)", low: "var(--resolved)",
};

/** [lon,lat] rings for a Polygon / MultiPolygon geometry. */
function ringsOf(geom: unknown): [number, number][][] {
  const g = geom as { type?: string; coordinates?: unknown };
  if (!g || !g.coordinates) return [];
  if (g.type === "Polygon") return g.coordinates as [number, number][][];
  if (g.type === "MultiPolygon")
    return (g.coordinates as number[][][][]).flatMap((poly) => poly as [number, number][][]);
  return [];
}

interface MapIncident {
  id: string;
  name: string;
  severity: string;
  status: string;
  polygons: unknown[];
  centroid: [number, number] | null;
}

function IncidentPolygonsMap({
  incidents,
  selected,
  onSelect,
  showDetections = true,
  showAlerts = true,
  alertPins = [],
}: {
  incidents: MapIncident[];
  selected?: string;
  onSelect?: (id: string) => void;
  showDetections?: boolean;
  showAlerts?: boolean;
  alertPins?: Array<{ id: string; lon: number; lat: number; severity: string }>;
}) {
  return (
    <MapView center={[40, 15]} span={320} className="min-h-[460px] w-full">
      {({ p, poly, k }) => (
        <>
          {showDetections &&
            incidents.map((inc) => {
              const color = SEV_FILL[inc.severity] ?? "var(--resolved)";
              const active = inc.id === selected;
              return (
                <g key={inc.id} onClick={() => onSelect?.(inc.id)} style={{ cursor: onSelect ? "pointer" : "default" }}>
                  {inc.polygons.flatMap((g, gi) =>
                    ringsOf(g).map((ring, ri) => (
                      <path key={gi + "-" + ri} d={poly(ring)} fill={color}
                        fillOpacity={active ? 0.7 : 0.38} stroke={color} strokeWidth={1.1 / k} />
                    )),
                  )}
                  {inc.centroid && (
                    <circle cx={p(inc.centroid)[0]} cy={p(inc.centroid)[1]}
                      r={(active ? 4.4 : 3) / k} fill={color} stroke="var(--card)" strokeWidth={1 / k} />
                  )}
                </g>
              );
            })}
          {showAlerts &&
            alertPins.map((a) => {
              const [x, y] = p([a.lon, a.lat]);
              const color = SEV_FILL[a.severity] ?? "var(--warning)";
              return <circle key={a.id} cx={x} cy={y} r={5 / k} fill="none" stroke={color} strokeWidth={1.6 / k} />;
            })}
        </>
      )}
    </MapView>
  );
}


export function CommandDashboard() {
  const { investigationsQ, alertsQ } = useOfficerData();
  const items = investigationsQ.data?.items ?? [];
  const officerAlerts = (alertsQ.data ?? []).filter((a) => a.severity !== "test");

  const withDetection = items.filter((i) => officerDetectionSummary(i).hasDetection);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected =
    withDetection.find((i) => i.id === selectedId) ?? withDetection[0] ?? null;
  const selectedResult = (selected?.ml_result as InferenceResult | null) ?? null;

  const sentAlerts = officerAlerts.filter((a) => a.delivery_status === "sent").length;
  const criticalAlerts = officerAlerts.filter((a) => a.severity === "critical").length;

  return (
    <OfficerShell
      title="Command Dashboard"
      subtitle="Live oil-spill detections, incidents and officer alerts"
      actions={
        <Button size="sm" variant="outline" onClick={() => { investigationsQ.refetch(); alertsQ.refetch(); }}>
          <RefreshCw /> Refresh
        </Button>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Investigations" value={String(items.length)} hint="Total on record" tone="satellite" icon={<FileCheck2 className="h-4 w-4" />} />
        <StatCard label="With detections" value={String(withDetection.length)} hint="Confirmed spill regions" tone="critical" icon={<ShieldAlert className="h-4 w-4" />} />
        <StatCard label="Alerts issued" value={String(officerAlerts.length)} hint={`${sentAlerts} delivered`} tone="warning" icon={<Radio className="h-4 w-4" />} />
        <StatCard label="Critical alerts" value={String(criticalAlerts)} hint="Highest severity" tone="critical" icon={<AlertTriangle className="h-4 w-4" />} />
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_360px]">
        <Panel
          title="Operational picture"
          right={selected ? <Pill tone="bg-satellite/15 text-ocean border-satellite/30">{selected.name}</Pill> : undefined}
          bodyClassName="p-0"
        >
          {selected ? (
            <div className="h-[440px]">
              <OriceanMapLibre
                mode="results"
                aoi={(selected.aoi_geojson as GeoJSONPolygon) ?? null}
                inferenceResult={selectedResult}
                sourceCorridorGeojson={selected.source_reconstruction_result?.source_corridor_geojson ?? null}
                className="h-full w-full"
              />
            </div>
          ) : (
            <div className="p-10 text-center text-sm text-muted-foreground">
              {investigationsQ.isLoading ? "Loading investigations…" : "No detections to display yet."}
            </div>
          )}
        </Panel>
        <Panel
          title="Priority queue"
          right={<Pill tone="bg-critical/12 text-critical border-critical/30">Alerts</Pill>}
          bodyClassName="p-0"
        >
          {officerAlerts.length === 0 ? (
            <p className="p-6 text-center text-xs text-muted-foreground">
              {alertsQ.isLoading ? "Loading alerts…" : "No alerts have been issued yet."}
            </p>
          ) : (
            officerAlerts.slice(0, 5).map((a) =>
              a.investigation_id ? (
                <Link key={a.id} to="/officer/incidents/$id" params={{ id: a.investigation_id }} className="block">
                  <AlertPillRow a={a} active={a.investigation_id === selected?.id} />
                </Link>
              ) : (
                <AlertPillRow key={a.id} a={a} />
              ),
            )
          )}
          <div className="border-t border-border p-3">
            <Button variant="ghost" size="sm" className="w-full" asChild>
              <Link to="/officer/alerts">Manage alerts & notifications <ArrowRight /></Link>
            </Button>
          </div>
        </Panel>
      </div>

      <Panel className="mt-5" title="Detected incidents" bodyClassName="p-0">
        {withDetection.length === 0 ? (
          <p className="p-6 text-center text-xs text-muted-foreground">No confirmed spill detections.</p>
        ) : (
          <div className="grid divide-y divide-border sm:grid-cols-2 sm:divide-x lg:grid-cols-3">
            {withDetection.slice(0, 9).map((inv) => {
              const s = officerDetectionSummary(inv);
              return (
                <Link
                  key={inv.id}
                  to="/officer/incidents/$id"
                  params={{ id: inv.id }}
                  className="block p-4 transition-colors hover:bg-muted/50"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-semibold">{inv.name}</span>
                    <Pill status={inv.status}>{inv.status}</Pill>
                  </div>
                  <div className="mono-num mt-2 flex items-center justify-between text-[11px] text-muted-foreground">
                    <span>{s.detectionCount} region(s) · {s.positiveTiles} tiles</span>
                    <span>{Math.round(s.maxConfidence * 100)}%</span>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </Panel>
    </OfficerShell>
  );
}

const SEV_ORDER = ["critical", "high", "moderate", "low", "test"];

export function ActiveAlerts() {
  const qc = useQueryClient();
  const alertsQ = useQuery({ queryKey: ["alerts"], queryFn: () => alertsApi.list(100), refetchInterval: 60_000 });
  const settingsQ = useQuery({ queryKey: ["alert-settings"], queryFn: () => alertsApi.getSettings() });
  const statusQ = useQuery({ queryKey: ["alert-status"], queryFn: () => alertsApi.status() });

  const [severityFilter, setSeverityFilter] = useState("All severities");
  const [form, setForm] = useState<NotificationSettings | null>(null);
  const effForm = form ?? settingsQ.data ?? null;

  const saveMut = useMutation({
    mutationFn: (patch: Partial<Omit<NotificationSettings, "updated_at">>) => alertsApi.updateSettings(patch),
    onSuccess: (data) => {
      setForm(data);
      qc.invalidateQueries({ queryKey: ["alert-settings"] });
      qc.invalidateQueries({ queryKey: ["alert-status"] });
    },
  });
  const testMut = useMutation({
    mutationFn: () => alertsApi.sendTest(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }),
  });

  const rows = (alertsQ.data ?? [])
    .slice()
    .sort((a, b) => SEV_ORDER.indexOf(a.severity) - SEV_ORDER.indexOf(b.severity))
    .filter((a) => severityFilter === "All severities" || a.severity === severityFilter.toLowerCase());

  const status = statusQ.data;
  const twilioReady = !!status?.account_configured && !!status?.from_number_set;

  return (
    <OfficerShell
      title="Active Alerts"
      subtitle="Detection alerts and officer notification settings"
      actions={
        <Button size="sm" onClick={() => testMut.mutate()} disabled={testMut.isPending}>
          <Send /> {testMut.isPending ? "Sending…" : "Send test alert"}
        </Button>
      }
    >
      {status && !twilioReady && (
        <div className="mb-4 flex items-start gap-2 border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <div>
            <p className="font-semibold">Twilio not fully configured — alerts are recorded but not delivered.</p>
            <p className="mt-0.5">
              Set <code className="font-mono">TWILIO_ACCOUNT_SID</code>,{" "}
              <code className="font-mono">TWILIO_AUTH_TOKEN</code> and{" "}
              <code className="font-mono">TWILIO_FROM_NUMBER</code> in{" "}
              <code className="font-mono">backend/.env</code>.
              {" "}Package installed: {status.package_installed ? "yes" : "no"}.
            </p>
          </div>
        </div>
      )}
      {testMut.data && (
        <div className={`mb-4 border px-3 py-2 text-xs ${
          testMut.data.delivery_status === "sent"
            ? "border-resolved/30 bg-resolved/10 text-resolved"
            : "border-warning/30 bg-warning/10 text-warning"
        }`}>
          Test alert {testMut.data.delivery_status} — {testMut.data.delivery_detail ?? "no detail"}
        </div>
      )}

      <div className="grid gap-3 sm:grid-cols-3">
        <StatCard label="Alerts recorded" value={String((alertsQ.data ?? []).length)} hint="All time" tone="satellite" />
        <StatCard label="Delivered" value={String((alertsQ.data ?? []).filter((a) => a.delivery_status === "sent").length)} hint="SMS sent to officers" tone="resolved" />
        <StatCard label="Recipients" value={String(status?.recipients_count ?? 0)} hint={status?.alerting_enabled ? "Alerting enabled" : "Alerting disabled"} tone={status?.alerting_enabled ? "warning" : "default"} />
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_360px]">
        <Panel
          title="Alert register"
          right={
            <select
              aria-label="Filter by severity"
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="h-8 border border-input bg-background px-2 text-xs"
            >
              {["All severities", "Critical", "High", "Moderate", "Low", "Test"].map((s) => <option key={s}>{s}</option>)}
            </select>
          }
          bodyClassName="p-0"
        >
          {rows.length === 0 ? (
            <p className="p-8 text-center text-sm text-muted-foreground">
              {alertsQ.isLoading ? "Loading alerts…" : "No alerts match this filter."}
            </p>
          ) : (
            <div className="divide-y divide-border">
              {rows.map((a) =>
                a.investigation_id ? (
                  <Link key={a.id} to="/officer/incidents/$id" params={{ id: a.investigation_id }} className="block">
                    <AlertPillRow a={a} />
                  </Link>
                ) : (
                  <AlertPillRow key={a.id} a={a} />
                ),
              )}
            </div>
          )}
        </Panel>

        <Panel title="Notification settings">
          {!effForm ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : (
            <div className="space-y-4">
              <label className="flex items-center justify-between text-sm">
                <span className="font-medium">Alerting enabled</span>
                <input
                  type="checkbox"
                  checked={effForm.enabled}
                  onChange={(e) => setForm({ ...effForm, enabled: e.target.checked })}
                  className="accent-[var(--satellite)]"
                />
              </label>
              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-muted-foreground">Min confidence to alert</span>
                  <span className="mono-num font-semibold">{Math.round(effForm.min_confidence * 100)}%</span>
                </div>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={effForm.min_confidence}
                  onChange={(e) => setForm({ ...effForm, min_confidence: parseFloat(e.target.value) })}
                  className="mt-1 w-full accent-[var(--satellite)]"
                />
              </div>
              <div>
                <label className="label-caps">Min positive tiles</label>
                <input
                  type="number" min={1}
                  value={effForm.min_positive_tiles}
                  onChange={(e) => setForm({ ...effForm, min_positive_tiles: parseInt(e.target.value || "1", 10) })}
                  className="mt-1 h-9 w-full border border-input bg-background px-2 text-sm"
                />
              </div>
              <div>
                <label className="label-caps">Officer phone numbers (one per line, E.164)</label>
                <textarea
                  rows={3}
                  value={effForm.recipients.join("\n")}
                  onChange={(e) => setForm({ ...effForm, recipients: e.target.value.split("\n").map((x) => x.trim()).filter(Boolean) })}
                  placeholder="+919876543210"
                  className="mt-1 w-full border border-input bg-background px-2 py-1.5 font-mono text-xs"
                />
              </div>
              <Button
                size="sm"
                className="w-full"
                disabled={saveMut.isPending}
                onClick={() => saveMut.mutate({
                  enabled: effForm.enabled,
                  min_confidence: effForm.min_confidence,
                  min_positive_tiles: effForm.min_positive_tiles,
                  recipients: effForm.recipients,
                })}
              >
                {saveMut.isPending ? "Saving…" : "Save settings"}
              </Button>
              {saveMut.isSuccess && <p className="text-xs text-resolved">Settings saved.</p>}
            </div>
          )}
        </Panel>
      </div>
    </OfficerShell>
  );
}

export function IncidentMap() {
  const [selected, setSelected] = useState<string | null>(null);
  const [showDetections, setShowDetections] = useState(true);
  const [showAlerts, setShowAlerts] = useState(true);

  const casesQ = useQuery({ queryKey: ["officer-cases"], queryFn: () => officerApi.listCases(), refetchInterval: 60_000 });
  const invQ = useQuery({ queryKey: ["investigations", "officer-map"], queryFn: () => investigationsApi.list(1, 80), refetchInterval: 120_000 });
  const alertsQ = useQuery({ queryKey: ["alerts"], queryFn: () => alertsApi.list(80), refetchInterval: 60_000 });

  const invById = useMemo(() => {
    const m = new Map<string, InvestigationRead>();
    for (const it of invQ.data?.items ?? []) m.set(it.id, it);
    return m;
  }, [invQ.data]);

  const incidents: MapIncident[] = useMemo(() => {
    return (casesQ.data ?? []).map((c) => {
      const inv = invById.get(c.investigation_id);
      const dets = (inv?.ml_result as InferenceResult | null)?.spill_detections ?? [];
      const polys = dets.map((d) => d.polygon_geojson).filter(Boolean) as unknown[];
      const centroids = polys.map((g) => polygonCentroid(g)).filter((x): x is [number, number] => !!x);
      const centroid = centroids.length
        ? ([
            centroids.reduce((s, x) => s + x[0], 0) / centroids.length,
            centroids.reduce((s, x) => s + x[1], 0) / centroids.length,
          ] as [number, number])
        : null;
      return { id: c.investigation_id, name: c.name, severity: c.severity, status: c.status, polygons: polys, centroid };
    });
  }, [casesQ.data, invById]);

  const alertPins = useMemo(() => {
    const byInv = new Map(incidents.map((i) => [i.id, i]));
    return (alertsQ.data ?? [])
      .filter((a) => a.severity !== "test" && a.investigation_id && byInv.get(a.investigation_id)?.centroid)
      .map((a) => {
        const c = byInv.get(a.investigation_id as string)!.centroid as [number, number];
        return { id: a.id, lon: c[0], lat: c[1], severity: a.severity };
      });
  }, [alertsQ.data, incidents]);

  const sel = incidents.find((i) => i.id === selected) ?? incidents[0] ?? null;

  return (
    <OfficerShell
      title="Incident Map"
      subtitle="Every detected spill and issued alert on one map"
      actions={
        <Button size="sm" variant="outline" onClick={() => { casesQ.refetch(); invQ.refetch(); alertsQ.refetch(); }}>
          <RefreshCw /> Refresh
        </Button>
      }
    >
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_330px]">
        <Panel
          title="Operational map"
          right={<Legend items={[{ color: "var(--critical)", label: "Critical" }, { color: "var(--warning)", label: "High" }]} />}
          bodyClassName="p-0"
        >
          <IncidentPolygonsMap
            incidents={incidents}
            selected={sel?.id}
            onSelect={setSelected}
            showDetections={showDetections}
            showAlerts={showAlerts}
            alertPins={alertPins}
          />
        </Panel>
        <div className="space-y-5">
          <Panel title="Map layers" bodyClassName="space-y-3">
            <label className="flex items-center justify-between gap-3 text-sm">
              <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-critical" />Spill detections</span>
              <input type="checkbox" checked={showDetections} onChange={(e) => setShowDetections(e.target.checked)} className="accent-[var(--satellite)]" />
            </label>
            <label className="flex items-center justify-between gap-3 text-sm">
              <span className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-warning" />Issued alerts</span>
              <input type="checkbox" checked={showAlerts} onChange={(e) => setShowAlerts(e.target.checked)} className="accent-[var(--satellite)]" />
            </label>
            <p className="mono-num text-[11px] text-muted-foreground">{incidents.length} incidents · {alertPins.length} alert markers</p>
          </Panel>
          {sel && (
            <Panel
              title="Selected incident"
              right={<Pill severity={sel.severity === "moderate" ? "medium" : (sel.severity as "critical" | "high" | "low")}>{sel.severity}</Pill>}
            >
              <p className="text-sm font-semibold">{sel.name}</p>
              <div className="mt-3 space-y-0"><KV k="Status" v={sel.status} /><KV k="Regions" v={String(sel.polygons.length)} /></div>
              <Button className="mt-4 w-full" size="sm" asChild>
                <Link to="/officer/incidents/$id" params={{ id: sel.id }}><ArrowRight /> Open incident detail</Link>
              </Button>
            </Panel>
          )}
        </div>
      </div>
      <Panel className="mt-5" title="Incident register" bodyClassName="p-0">
        {incidents.length === 0 ? (
          <p className="p-6 text-center text-xs text-muted-foreground">{casesQ.isLoading ? "Loading…" : "No detected incidents."}</p>
        ) : (
          <div className="grid divide-y divide-border sm:grid-cols-2 lg:grid-cols-3 lg:divide-x">
            {incidents.map((inc) => (
              <Link
                key={inc.id}
                to="/officer/incidents/$id"
                params={{ id: inc.id }}
                className={`block p-4 transition-colors hover:bg-muted/50 ${inc.id === sel?.id ? "bg-satellite/10" : ""}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="text-sm font-semibold">{inc.name}</span>
                  <Pill severity={inc.severity === "moderate" ? "medium" : (inc.severity as "critical" | "high" | "low")}>{inc.severity}</Pill>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{inc.status} · {inc.polygons.length} spill region(s)</p>
              </Link>
            ))}
          </div>
        )}
      </Panel>
    </OfficerShell>
  );
}

function EvidenceStep({ label, at, done }: { label: string; at?: string | null; done: boolean }) {
  return (
    <div className="flex items-start gap-3 p-3">
      <span className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full ${done ? "bg-resolved/15 text-resolved" : "bg-muted text-muted-foreground"}`}>
        {done ? <Check className="h-3 w-3" /> : <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      </span>
      <div>
        <p className="text-xs font-semibold">{label}</p>
        <p className="mono-num mt-0.5 text-[10px] text-muted-foreground">{at ? formatDateTime(at) : done ? "done" : "pending"}</p>
      </div>
    </div>
  );
}

export function CaseReview() {
  const qc = useQueryClient();
  const casesQ = useQuery({ queryKey: ["officer-cases"], queryFn: () => officerApi.listCases(), refetchInterval: 60_000 });
  const cases = useMemo(
    () =>
      (casesQ.data ?? [])
        .slice()
        .sort((a, b) => Number(a.review_decision !== "pending") - Number(b.review_decision !== "pending")),
    [casesQ.data],
  );
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const sel = cases.find((c) => c.investigation_id === selectedId) ?? cases[0] ?? null;
  const [notes, setNotes] = useState("");

  const invQ = useQuery({
    queryKey: ["investigation", sel?.investigation_id, "case"],
    queryFn: () => investigationsApi.get(sel!.investigation_id),
    enabled: !!sel,
  });
  const attrQ = useQuery({
    queryKey: ["attribution", sel?.investigation_id],
    queryFn: () => attributionApi.get(sel!.investigation_id).catch(() => null),
    enabled: !!sel,
  });

  const mut = useMutation({
    mutationFn: (patch: Parameters<typeof officerApi.updateCase>[1]) => officerApi.updateCase(sel!.investigation_id, patch),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["officer-cases"] });
      setNotes("");
    },
  });

  const inv = invQ.data ?? null;
  const ml = (inv?.ml_result as InferenceResult | null) ?? null;
  const recon = (inv?.source_reconstruction_result as SourceReconstructionResult | null) ?? null;
  const attr = (attrQ.data as AttributionResult | null) ?? null;
  const top = (attr?.candidates ?? []).filter((c) => c.qualified)[0] ?? attr?.candidates?.[0] ?? null;

  return (
    <OfficerShell title="Case Review" subtitle="Validate analyst evidence before inter-agency handoff">
      <div className="grid gap-5 xl:grid-cols-[300px_minmax(0,1fr)]">
        <Panel title="Cases" right={<span className="mono-num text-xs text-muted-foreground">{cases.length}</span>} bodyClassName="p-0">
          <div className="divide-y divide-border">
            {cases.map((c) => (
              <button
                key={c.investigation_id}
                type="button"
                onClick={() => setSelectedId(c.investigation_id)}
                className={`w-full p-3 text-left transition-colors ${c.investigation_id === sel?.investigation_id ? "bg-satellite/10" : "hover:bg-muted/50"}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="truncate text-xs font-semibold">{c.name}</span>
                  <Pill severity={c.severity === "moderate" ? "medium" : (c.severity as "critical" | "high" | "low")}>{c.severity}</Pill>
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-[10px] text-muted-foreground">{REVIEW_LABEL[c.review_decision] ?? c.review_decision}</span>
                  <span className="mono-num text-[10px] text-muted-foreground">{Math.round(c.max_confidence * 100)}%</span>
                </div>
              </button>
            ))}
            {cases.length === 0 && (
              <p className="p-6 text-center text-xs text-muted-foreground">{casesQ.isLoading ? "Loading…" : "No cases."}</p>
            )}
          </div>
        </Panel>

        {sel && (
          <div className="space-y-5">
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
              <Panel title="Evidence snapshot" right={<Pill>{REVIEW_LABEL[sel.review_decision] ?? sel.review_decision}</Pill>} bodyClassName="p-0">
                <div className="h-[320px]">
                  <OriceanMapLibre
                    mode="results"
                    aoi={(inv?.aoi_geojson as GeoJSONPolygon) ?? null}
                    inferenceResult={ml}
                    sourceCorridorGeojson={recon?.source_corridor_geojson ?? null}
                    className="h-full w-full"
                  />
                </div>
              </Panel>
              <Panel title="Case disposition">
                <p className="text-sm font-semibold">{sel.name}</p>
                <div className="mt-3 space-y-0">
                  <KV k="Detection" v={`${Math.round(sel.max_confidence * 100)}% · ${sel.positive_tiles} tiles`} />
                  <KV k="Source confidence" v={recon ? `${Math.round((recon.source_confidence ?? 0) * 100)}%` : "not run"} />
                  <KV k="Top candidate" v={top ? `${top.name ?? top.mmsi} (${top.attribution_score.toFixed(0)})` : "not run"} />
                  <KV k="Alerts issued" v={String(sel.alert_count)} />
                </div>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={2}
                  placeholder="Review notes…"
                  className="mt-3 w-full border border-input bg-background px-2 py-1.5 text-xs"
                />
                <div className="mt-3 space-y-2">
                  <Button className="w-full" size="sm" disabled={mut.isPending} onClick={() => mut.mutate({ review_decision: "approved", review_notes: notes || sel.review_notes || undefined })}>
                    <Check /> Approve for handoff
                  </Button>
                  <Button className="w-full" size="sm" variant="outline" disabled={mut.isPending} onClick={() => mut.mutate({ review_decision: "returned", review_notes: notes || undefined })}>
                    <MessageSquareText /> Return to analyst
                  </Button>
                  <Button className="w-full" size="sm" variant="outline" disabled={mut.isPending} onClick={() => mut.mutate({ review_decision: "escalated", review_notes: notes || undefined })}>
                    <ShieldAlert /> Escalate
                  </Button>
                </div>
                {sel.reviewed_at && (
                  <p className="mt-2 text-[11px] text-muted-foreground">
                    Last decision {formatDateTime(sel.reviewed_at)}
                    {sel.review_notes ? ` — "${sel.review_notes}"` : ""}
                  </p>
                )}
              </Panel>
            </div>
            <Panel title="Evidence chain" bodyClassName="p-0">
              <div className="divide-y divide-border">
                <EvidenceStep label="Investigation created" at={inv?.created_at} done={!!inv} />
                <EvidenceStep
                  label={`Sentinel-1 SAR detection — ${ml?.positive_tile_count ?? 0} positive tiles`}
                  at={inv?.ml_run_at}
                  done={!!ml && (ml.spill_detections?.length ?? 0) > 0}
                />
                <EvidenceStep label="Source reconstruction" at={recon?.run_at} done={!!recon} />
                <EvidenceStep
                  label={`Vessel attribution — ${attr?.qualified_count ?? 0} qualified of ${attr?.total_candidates ?? 0}`}
                  at={attr?.run_at}
                  done={!!attr}
                />
                <EvidenceStep label={`Officer alert${sel.alert_count === 1 ? "" : "s"} issued`} done={sel.alert_count > 0} />
              </div>
            </Panel>
          </div>
        )}
      </div>
    </OfficerShell>
  );
}

export function ResponseStatus() {
  const qc = useQueryClient();
  const casesQ = useQuery({ queryKey: ["officer-cases"], queryFn: () => officerApi.listCases(), refetchInterval: 60_000 });
  const active = (casesQ.data ?? []).filter(
    (c) => (c.severity === "critical" || c.severity === "high") && c.response_status !== "closed",
  );
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const sel = active.find((c) => c.investigation_id === selectedId) ?? active[0] ?? null;

  const mut = useMutation({
    mutationFn: (patch: Parameters<typeof officerApi.updateCase>[1]) => officerApi.updateCase(sel!.investigation_id, patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["officer-cases"] }),
  });

  const counts = (s: string) => (casesQ.data ?? []).filter((c) => c.response_status === s).length;
  const toggleItem = (i: number) => {
    if (!sel) return;
    const next = sel.response_checklist.map((it, idx) => (idx === i ? { ...it, done: !it.done } : it));
    mut.mutate({ response_checklist: next });
  };

  return (
    <OfficerShell
      title="Response Status"
      subtitle="Field response state for high-severity incidents"
      actions={<Button size="sm" variant="outline" onClick={() => casesQ.refetch()}><RefreshCw /> Refresh</Button>}
    >
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Needs response" value={String(active.length)} hint="High / critical, not closed" tone="critical" />
        <StatCard label="Responding" value={String(counts("responding"))} tone="warning" />
        <StatCard label="Contained" value={String(counts("contained"))} tone="satellite" />
        <StatCard label="Closed" value={String(counts("closed"))} tone="resolved" />
      </div>
      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Panel title="Active incidents" bodyClassName="p-0">
          <div className="divide-y divide-border">
            {active.map((c) => (
              <button
                key={c.investigation_id}
                type="button"
                onClick={() => setSelectedId(c.investigation_id)}
                className={`w-full p-4 text-left transition-colors ${c.investigation_id === sel?.investigation_id ? "bg-satellite/10" : "hover:bg-muted/50"}`}
              >
                <div className="flex items-start justify-between gap-3">
                  <span className="text-sm font-semibold">{c.name}</span>
                  <Pill severity={c.severity === "moderate" ? "medium" : (c.severity as "critical" | "high" | "low")}>{c.severity}</Pill>
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">{RESPONSE_LABEL[c.response_status] ?? c.response_status}</span>
                  <span className="mono-num text-[10px] text-muted-foreground">
                    {c.response_checklist.filter((x) => x.done).length}/{c.response_checklist.length} tasks
                  </span>
                </div>
              </button>
            ))}
            {active.length === 0 && (
              <p className="p-6 text-center text-xs text-muted-foreground">{casesQ.isLoading ? "Loading…" : "No incidents need response."}</p>
            )}
          </div>
        </Panel>
        {sel && (
          <div className="space-y-5">
            <Panel title="Response state">
              <p className="text-sm font-semibold">{sel.name}</p>
              <label className="mt-3 block text-xs text-muted-foreground">Status</label>
              <select
                value={sel.response_status}
                onChange={(e) => mut.mutate({ response_status: e.target.value })}
                className="mt-1 h-9 w-full border border-input bg-background px-2 text-sm"
              >
                {["not_started", "responding", "contained", "closed"].map((s) => (
                  <option key={s} value={s}>{RESPONSE_LABEL[s]}</option>
                ))}
              </select>
            </Panel>
            <Panel title="Containment checklist">
              <div className="space-y-3">
                {sel.response_checklist.map((it: ChecklistItem, i: number) => (
                  <label key={i} className="flex items-center gap-3 text-sm">
                    <input type="checkbox" checked={it.done} onChange={() => toggleItem(i)} className="accent-[var(--satellite)]" />
                    <span className={it.done ? "text-muted-foreground line-through" : ""}>{it.label}</span>
                  </label>
                ))}
              </div>
              {sel.updated_at && <p className="mt-3 text-[11px] text-muted-foreground">Updated {formatDateTime(sel.updated_at)}</p>}
            </Panel>
          </div>
        )}
      </div>
    </OfficerShell>
  );
}

export function OfficerReports() {
  const qc = useQueryClient();
  const casesQ = useQuery({ queryKey: ["officer-cases"], queryFn: () => officerApi.listCases(), refetchInterval: 60_000 });
  const rows = casesQ.data ?? [];
  const mut = useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: Parameters<typeof officerApi.updateCase>[1] }) =>
      officerApi.updateCase(id, patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["officer-cases"] }),
  });

  const downloadReport = async (id: string) => {
    try {
      const [inv, attr] = await Promise.all([investigationsApi.get(id), attributionApi.get(id).catch(() => null)]);
      openPrintableReport({
        investigation: inv,
        result: (inv.ml_result as InferenceResult | null) ?? null,
        sourceRecon: (inv.source_reconstruction_result as SourceReconstructionResult | null) ?? null,
        attribution: (attr as AttributionResult | null) ?? null,
        generatedAt: new Date().toISOString(),
      });
    } catch {
      alert("Could not load the investigation for this report.");
    }
  };

  const count = (s: string) => rows.filter((r) => r.report_status === s).length;

  return (
    <OfficerShell title="Reports" subtitle="Review, approve, and distribute investigation handoff packages">
      <div className="grid gap-3 sm:grid-cols-3">
        <StatCard label="Awaiting review" value={String(count("awaiting_review"))} tone="satellite" icon={<FileCheck2 className="h-4 w-4" />} />
        <StatCard label="Approved" value={String(count("approved"))} tone="warning" icon={<Check className="h-4 w-4" />} />
        <StatCard label="Distributed" value={String(count("distributed"))} tone="resolved" icon={<CheckCircle2 className="h-4 w-4" />} />
      </div>
      <Panel className="mt-5" title="Handoff register" right={<Pill tone="bg-satellite/15 text-ocean border-satellite/30">Officer access</Pill>} bodyClassName="p-0">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="bg-muted/60 text-[10px] tracking-[0.12em] text-muted-foreground uppercase">
              <tr>
                <th className="px-4 py-2.5 font-semibold">Incident</th>
                <th className="px-4 py-2.5 font-semibold">Severity</th>
                <th className="px-4 py-2.5 font-semibold">Review</th>
                <th className="px-4 py-2.5 font-semibold">Report status</th>
                <th className="px-4 py-2.5 text-right font-semibold">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {rows.map((r) => (
                <tr key={r.investigation_id} className="hover:bg-muted/40">
                  <td className="px-4 py-3">
                    <span className="text-xs font-semibold">{r.name}</span>
                    <span className="mono-num mt-0.5 block text-[10px] text-muted-foreground">{r.investigation_id.slice(0, 8)}</span>
                  </td>
                  <td className="px-4 py-3">
                    <Pill severity={r.severity === "moderate" ? "medium" : (r.severity as "critical" | "high" | "low")}>{r.severity}</Pill>
                  </td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">{REVIEW_LABEL[r.review_decision] ?? r.review_decision}</td>
                  <td className="px-4 py-3"><Pill>{REPORT_LABEL[r.report_status] ?? r.report_status}</Pill></td>
                  <td className="px-4 py-3">
                    <div className="flex justify-end gap-1.5">
                      <Button size="sm" variant="ghost" onClick={() => downloadReport(r.investigation_id)}><Download /> PDF</Button>
                      {r.report_status !== "approved" && r.report_status !== "distributed" && (
                        <Button size="sm" variant="outline" disabled={mut.isPending} onClick={() => mut.mutate({ id: r.investigation_id, patch: { report_status: "approved" } })}>
                          <Check /> Approve
                        </Button>
                      )}
                      {r.report_status === "approved" && (
                        <Button size="sm" disabled={mut.isPending} onClick={() => mut.mutate({ id: r.investigation_id, patch: { report_status: "distributed" } })}>
                          <Send /> Distribute
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-xs text-muted-foreground">
                    {casesQ.isLoading ? "Loading…" : "No incidents on record."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </OfficerShell>
  );
}

// ===========================================================================
// IncidentDetail — full officer view of a single investigation
// ===========================================================================

export function IncidentDetail({ investigationId }: { investigationId: string }) {
  const invQ = useQuery({
    queryKey: ["investigation", investigationId],
    queryFn: () => investigationsApi.get(investigationId),
    enabled: !!investigationId,
  });
  const attrQ = useQuery({
    queryKey: ["attribution", investigationId],
    queryFn: () => attributionApi.get(investigationId).catch(() => null),
    enabled: !!investigationId,
  });
  const alertsQ = useQuery({ queryKey: ["alerts"], queryFn: () => alertsApi.list(100) });

  const inv = invQ.data ?? null;
  const ml = (inv?.ml_result as InferenceResult | null) ?? null;
  const attr = (attrQ.data as AttributionResult | null) ?? null;
  const s = inv ? officerDetectionSummary(inv) : null;

  const qualified = (attr?.candidates ?? []).filter((c) => c.qualified);
  const shortlist = (qualified.length ? qualified : (attr?.candidates ?? [])).slice(0, 8);
  const topMmsi = shortlist[0]?.mmsi ?? null;
  const aisTracks = shortlist
    .filter((c) => c.track && c.track.length >= 1)
    .map((c) => ({ mmsi: c.mmsi, name: c.name, track: c.track as Array<[number, number]> }));

  const incidentAlerts = (alertsQ.data ?? []).filter((a) => a.investigation_id === investigationId);

  // ---- Drift replay (simulated straight-line estimate) --------------------
  const recon = (inv?.source_reconstruction_result as SourceReconstructionResult | null) ?? null;
  const [driftIdx, setDriftIdx] = useState(0);
  const [driftPlaying, setDriftPlaying] = useState(false);
  const [driftDir, setDriftDir] = useState<"forward" | "backward">("forward");
  const spillCentroid = useMemo<[number, number] | null>(() => {
    const cs = (ml?.spill_detections ?? [])
      .map((d) => polygonCentroid(d.polygon_geojson))
      .filter((c): c is [number, number] => !!c);
    if (!cs.length) return null;
    return [cs.reduce((s, c) => s + c[0], 0) / cs.length, cs.reduce((s, c) => s + c[1], 0) / cs.length];
  }, [ml]);
  const detTimeISO = useMemo(() => {
    const acq = (ml?.tile_results ?? []).map((t) => t.acquisition_date).filter(Boolean).sort().at(-1);
    return acq ?? inv?.end_date ?? recon?.release_window_end ?? null;
  }, [ml, inv, recon]);
  const driftSteps = useMemo<ReplayStep[]>(() => {
    if (!recon || !spillCentroid || !detTimeISO) return [];
    const src = polygonCentroid(recon.source_corridor_geojson);
    if (!src) return [];
    return buildReplaySteps(src, spillCentroid, recon.release_window_start, detTimeISO, 24, driftDir);
  }, [recon, spillCentroid, detTimeISO, driftDir]);
  useEffect(() => { setDriftIdx(0); setDriftPlaying(false); }, [driftDir, recon?.run_at]);

  if (invQ.isLoading) {
    return (
      <OfficerShell title="Incident" subtitle="Loading…">
        <p className="p-8 text-sm text-muted-foreground">Loading incident…</p>
      </OfficerShell>
    );
  }
  if (!inv) {
    return (
      <OfficerShell title="Incident not found" subtitle="No investigation with that id">
        <Button size="sm" variant="outline" asChild><Link to="/officer"><ArrowRight /> Back to dashboard</Link></Button>
      </OfficerShell>
    );
  }

  return (
    <OfficerShell
      title={inv.name}
      subtitle={`Investigation ${inv.id.slice(0, 8)} · ${inv.start_date} → ${inv.end_date}`}
      actions={<Button size="sm" variant="outline" asChild><Link to="/officer"><ArrowRight /> Dashboard</Link></Button>}
    >
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Status" value={inv.status} tone={inv.status === "complete" ? "resolved" : "warning"} />
        <StatCard label="Spill regions" value={String(s?.detectionCount ?? 0)} hint={`${s?.positiveTiles ?? 0} positive tiles`} tone="critical" />
        <StatCard label="Max confidence" value={`${Math.round((s?.maxConfidence ?? 0) * 100)}%`} tone="warning" />
        <StatCard label="Qualified vessels" value={String(attr?.qualified_count ?? 0)} hint={attr ? `${attr.total_candidates} scored` : "not run"} tone="satellite" />
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1.5fr)_minmax(320px,0.7fr)]">
        <Panel title="Satellite analysis & vessel context" bodyClassName="p-0">
          <div className="h-[460px]">
            <OriceanMapLibre
              mode="results"
              aoi={(inv.aoi_geojson as GeoJSONPolygon) ?? null}
              inferenceResult={ml}
              sourceCorridorGeojson={inv.source_reconstruction_result?.source_corridor_geojson ?? null}
              aisVesselTracks={aisTracks.length ? aisTracks : null}
              highlightMmsi={topMmsi}
              driftReplay={driftSteps.length ? { frames: driftSteps, activeIndex: driftIdx, direction: driftDir } : null}
              className="h-full w-full"
            />
          </div>
          {driftSteps.length > 0 && (
            <DriftReplayBar
              steps={driftSteps}
              activeIndex={driftIdx}
              onIndex={setDriftIdx}
              playing={driftPlaying}
              onPlayToggle={setDriftPlaying}
              direction={driftDir}
              onDirection={setDriftDir}
              label={`Simulated straight-line drift — ${driftDir === "forward" ? "source corridor → observed spill" : "observed spill → probable source"}, over the reconstruction release window. Hollow amber marker is simulated; the solid polygon is the actual Sentinel-1 detection.`}
            />
          )}
          <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
            Detected spill region{inv.source_reconstruction_result ? ", source corridor" : ""}
            {aisTracks.length ? `, and ${aisTracks.length} candidate vessel route(s). Red = top-ranked candidate vessel.` : "."}
          </div>
        </Panel>

        <div className="space-y-5">
          <Panel title="Source reconstruction">
            {inv.source_reconstruction_result ? (
              <div className="space-y-0">
                <KV k="Method" v={inv.source_reconstruction_result.reconstruction_method} />
                <KV k="Drift distance" v={`${inv.source_reconstruction_result.drift_distance_km?.toFixed?.(1) ?? "—"} km`} />
                <KV k="Confidence" v={`${Math.round((inv.source_reconstruction_result.source_confidence ?? 0) * 100)}%`} />
                <KV k="Window" v={`${formatDateTime(inv.source_reconstruction_result.release_window_start)} → ${formatDateTime(inv.source_reconstruction_result.release_window_end)}`} />
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">Not run for this investigation.</p>
            )}
          </Panel>

          <Panel title="Incident alerts" bodyClassName="p-0">
            {incidentAlerts.length === 0 ? (
              <p className="p-4 text-xs text-muted-foreground">No alerts issued for this incident.</p>
            ) : (
              <div className="divide-y divide-border">
                {incidentAlerts.map((a) => <AlertPillRow key={a.id} a={a} />)}
              </div>
            )}
          </Panel>
        </div>
      </div>

      <Panel className="mt-5" title="Candidate vessel ranking" right={<Pill tone="bg-vessel/12 text-vessel border-vessel/30">Restricted</Pill>} bodyClassName="p-0">
        {shortlist.length === 0 ? (
          <p className="p-6 text-center text-xs text-muted-foreground">
            {attr ? "No candidate vessels for this incident." : "Vessel attribution has not been run."}
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-xs">
              <thead className="bg-muted/60 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                <tr>
                  <th className="px-3 py-2.5 font-semibold">Rank</th>
                  <th className="px-3 py-2.5 font-semibold">Vessel</th>
                  <th className="px-3 py-2.5 font-semibold">Score</th>
                  <th className="px-3 py-2.5 font-semibold">Closest dist.</th>
                  <th className="px-3 py-2.5 font-semibold">Time diff.</th>
                  <th className="px-3 py-2.5 font-semibold">Why it qualified</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {shortlist.map((c) => (
                  <tr key={c.mmsi} className={c.qualified ? "" : "opacity-60"}>
                    <td className="px-3 py-2.5 align-top">
                      <span className={`mono-num flex h-6 w-6 items-center justify-center text-xs font-semibold ${c.mmsi === topMmsi ? "bg-critical text-critical-foreground" : "bg-muted text-muted-foreground"}`}>
                        {String(c.rank).padStart(2, "0")}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 align-top font-medium">
                      {c.name ?? <span className="text-muted-foreground">Unknown</span>}
                      <div className="text-[10px] font-normal text-muted-foreground">
                        {[c.vessel_type, c.flag, c.mmsi].filter(Boolean).join(" · ")}
                      </div>
                    </td>
                    <td className="mono-num px-3 py-2.5 align-top font-semibold">{c.attribution_score.toFixed(1)}</td>
                    <td className="mono-num px-3 py-2.5 align-top">{c.min_distance_km.toFixed(1)} km</td>
                    <td className="mono-num px-3 py-2.5 align-top">{c.time_diff_hours != null ? `${c.time_diff_hours.toFixed(1)} h` : "—"}</td>
                    <td className="px-3 py-2.5 align-top">
                      {c.qualified ? (
                        <div className="flex flex-wrap gap-1">
                          {(c.qualification_reasons ?? []).map((r, i) => (
                            <span key={i} className="rounded-sm bg-resolved/12 px-1.5 py-0.5 text-[10px] leading-4 text-resolved">{r}</span>
                          ))}
                        </div>
                      ) : (
                        <span className="rounded-sm bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">below evidence threshold</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {(ml?.warnings?.length ?? 0) > 0 && (
        <Panel className="mt-5" title="Processing notes" bodyClassName="p-4">
          <ul className="space-y-1 text-xs text-muted-foreground">
            {ml!.warnings.map((w, i) => <li key={i}>• {w}</li>)}
          </ul>
        </Panel>
      )}
    </OfficerShell>
  );
}