import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Droplets,
  Globe2,
  Loader2,
  Radio,
  Search,
  ShieldCheck,
  Waves,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { AppShell } from "@/components/oricean/AppShell";
import { MapView } from "@/components/oricean/MapView";
import { KV, Legend, Panel, Pill, StatCard } from "@/components/oricean/bits";
import { formatDate, formatDateTime, insights } from "@/lib/oricean-data";
import { publicApi, type PublicIncident } from "@/lib/api";

function PublicShell({
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
    <AppShell role="public" title={title} subtitle={subtitle} actions={actions}>
      <div className="mx-auto w-full max-w-[1500px]">{children}</div>
    </AppShell>
  );
}

const sevFill: Record<string, string> = {
  critical: "var(--critical)",
  high: "var(--warning)",
  moderate: "var(--caution)",
  low: "var(--resolved)",
};
const sevLabel: Record<string, string> = {
  critical: "Critical",
  high: "High",
  moderate: "Moderate",
  low: "Low",
};

/** [lon,lat] rings for a Polygon / MultiPolygon geometry. */
function ringsOf(geom: PublicIncident["spill_region_geojson"]): [number, number][][] {
  if (!geom) return [];
  if (geom.type === "Polygon") return geom.coordinates as [number, number][][];
  if (geom.type === "MultiPolygon")
    return (geom.coordinates as number[][][][]).flatMap((poly) => poly as [number, number][][]);
  return [];
}

function usePublicIncidents() {
  return useQuery({
    queryKey: ["public-incidents"],
    queryFn: () => publicApi.incidents(),
    refetchInterval: 60_000,
  });
}

function IncidentsMap({
  data,
  selected,
  onSelect,
  center = [30, 18],
  span = 340,
}: {
  data: PublicIncident[];
  selected?: string;
  onSelect?: (id: string) => void;
  center?: [number, number];
  span?: number;
}) {
  return (
    <MapView
      center={center}
      span={span}
      className="min-h-[460px] w-full"
      overlay={
        <div className="absolute bottom-3 left-3 flex flex-wrap gap-3 border border-white/10 bg-ocean-deep/90 px-3 py-2 text-[10px] text-white/75 backdrop-blur-sm">
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-critical" /> Critical</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-warning" /> High</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-caution" /> Moderate</span>
          <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-resolved" /> Low</span>
        </div>
      }
    >
      {({ p, poly, k }) => (
        <>
          {data.map((inc) => {
            const color = sevFill[inc.severity] ?? "var(--resolved)";
            const active = inc.id === selected;
            const c = inc.centroid;
            return (
              <g key={inc.id} onClick={() => onSelect?.(inc.id)} style={{ cursor: onSelect ? "pointer" : "default" }}>
                {ringsOf(inc.spill_region_geojson).map((ring, i) => (
                  <path
                    key={i}
                    d={poly(ring)}
                    fill={color}
                    fillOpacity={active ? 0.8 : 0.45}
                    stroke={color}
                    strokeWidth={1.1 / k}
                  />
                ))}
                {c && (
                  <circle
                    cx={p(c)[0]}
                    cy={p(c)[1]}
                    r={(active ? 4.2 : 3) / k}
                    fill={color}
                    stroke="var(--card)"
                    strokeWidth={1 / k}
                  />
                )}
              </g>
            );
          })}
        </>
      )}
    </MapView>
  );
}

function IncidentCard({
  inc,
  onSelect,
  active,
}: {
  inc: PublicIncident;
  onSelect?: () => void;
  active?: boolean;
}) {
  return (
    <button
      onClick={onSelect}
      className={`w-full p-4 text-left transition-colors ${active ? "bg-satellite/10" : "hover:bg-muted/50"}`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="text-sm font-semibold">{inc.region_label}</span>
        <Pill severity={inc.severity === "moderate" ? "medium" : (inc.severity as "critical" | "high" | "low")}>
          {sevLabel[inc.severity] ?? inc.severity}
        </Pill>
      </div>
      <div className="mono-num mt-2 flex items-center justify-between text-[11px] text-muted-foreground">
        <span>{inc.detection_date ? formatDate(inc.detection_date) : "—"}</span>
        <span>{inc.area_km2 != null ? `${Math.round(inc.area_km2).toLocaleString()} km²` : "—"}</span>
      </div>
      <div className="mt-1 text-[11px] text-muted-foreground">Status: {inc.status}</div>
    </button>
  );
}

function Disclaimer() {
  return (
    <div className="mt-5 flex items-start gap-2 rounded-md border border-border bg-muted/40 p-3 text-xs leading-5 text-muted-foreground">
      <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-resolved" />
      Public data is aggregated from satellite detections and published with a delay. Vessel identities,
      attribution evidence and internal model scores are withheld from public access and remain restricted
      to authorised agencies.
    </div>
  );
}

function LoadState({ query }: { query: ReturnType<typeof usePublicIncidents> }) {
  if (query.isLoading)
    return (
      <div className="flex items-center gap-2 p-8 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading published incidents…
      </div>
    );
  if (query.isError)
    return (
      <div className="flex items-center gap-2 border border-critical/30 bg-critical/10 p-3 text-sm text-critical">
        <AlertTriangle className="h-4 w-4 shrink-0" /> Could not reach the incident service. Try again shortly.
      </div>
    );
  if ((query.data ?? []).length === 0)
    return <div className="p-8 text-center text-sm text-muted-foreground">No spill incidents have been published yet.</div>;
  return null;
}

/* ------------------------------- Global map ------------------------------ */

function GlobalMap() {
  const query = usePublicIncidents();
  const incidents = query.data ?? [];

  const [severity, setSeverity] = useState("All severities");
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const filtered = incidents
    .filter((i) => severity === "All severities" || (sevLabel[i.severity] ?? i.severity) === severity)
    .filter((i) => i.region_label.toLowerCase().includes(search.toLowerCase()));
  const active = filtered.find((i) => i.id === selected) ?? filtered[0];

  const activeCount = incidents.filter((i) => i.status !== "complete" && i.status !== "closed").length;
  const totalArea = incidents.reduce((s, i) => s + (i.area_km2 ?? 0), 0);

  return (
    <PublicShell
      title="Global Spill Map"
      subtitle="Publicly released marine oil spill detections"
      actions={
        <Button size="sm" variant="outline" asChild>
          <Link to="/explore/insights"><BarChart3 /> View insights</Link>
        </Button>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Detections published" value={String(incidents.length)} hint="From satellite analysis" tone="satellite" icon={<Globe2 className="h-4 w-4" />} />
        <StatCard label="Currently active" value={String(activeCount)} hint="Not yet closed" tone="critical" icon={<AlertTriangle className="h-4 w-4" />} />
        <StatCard label="Total area affected" value={`${Math.round(totalArea).toLocaleString()} km²`} hint="Cumulative detected surface" tone="warning" icon={<Droplets className="h-4 w-4" />} />
        <StatCard label="Continuous SAR" value="Sentinel-1" hint="Primary sensor" tone="default" icon={<Waves className="h-4 w-4" />} />
      </div>

      <div className="mt-5">
        <LoadState query={query} />
      </div>

      {incidents.length > 0 && (
        <div className="mt-3 grid gap-5 xl:grid-cols-[minmax(0,1.5fr)_minmax(320px,0.7fr)]">
          <Panel
            title="Published detections"
            right={<Legend items={[{ color: "var(--critical)", label: "Critical" }, { color: "var(--warning)", label: "High" }]} />}
            bodyClassName="p-0"
          >
            <IncidentsMap data={filtered} selected={active?.id} onSelect={setSelected} />
          </Panel>

          <div className="space-y-5">
            <Panel title="Filters" bodyClassName="space-y-3">
              <div className="relative">
                <Search className="absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search region"
                  className="h-9 w-full border border-input bg-background pr-3 pl-8 text-xs outline-none focus:ring-1 focus:ring-ring"
                />
              </div>
              <select aria-label="Filter by severity" value={severity} onChange={(e) => setSeverity(e.target.value)} className="h-9 w-full border border-input bg-background px-2 text-xs">
                {["All severities", "Critical", "High", "Moderate", "Low"].map((s) => <option key={s}>{s}</option>)}
              </select>
              <p className="mono-num text-[11px] text-muted-foreground">{filtered.length} incidents shown</p>
            </Panel>

            {active && (
              <Panel title="Incident detail" right={<Pill>{active.status}</Pill>}>
                <p className="text-sm font-semibold">{active.region_label}</p>
                <div className="mt-3 space-y-0">
                  <KV k="Detection date" v={active.detection_date ? formatDate(active.detection_date) : "—"} />
                  <KV k="Status" v={active.status} />
                  <KV k="Severity" v={sevLabel[active.severity] ?? active.severity} />
                  <KV k="Confidence" v={active.confidence_band} />
                  <KV k="Area" v={active.area_km2 != null ? `${Math.round(active.area_km2).toLocaleString()} km²` : "—"} />
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}

      {incidents.length > 0 && (
        <Panel className="mt-5" title="Most recent detections" bodyClassName="p-0">
          <div className="grid divide-y divide-border sm:grid-cols-2 sm:divide-x lg:grid-cols-4">
            {filtered.slice(0, 4).map((i) => (
              <IncidentCard key={i.id} inc={i} active={i.id === active?.id} onSelect={() => setSelected(i.id)} />
            ))}
          </div>
        </Panel>
      )}

      <Disclaimer />
    </PublicShell>
  );
}

/* ------------------------------ Live incidents --------------------------- */

function LiveIncidents() {
  const query = usePublicIncidents();
  const live = (query.data ?? []).filter((i) => i.status !== "complete" && i.status !== "closed");
  const [selected, setSelected] = useState<string | null>(null);
  const active = live.find((i) => i.id === selected) ?? live[0];

  return (
    <PublicShell title="Live Incidents" subtitle="Ongoing marine pollution events currently being tracked">
      <div className="grid gap-3 sm:grid-cols-3">
        <StatCard label="Active incidents" value={String(live.length).padStart(2, "0")} hint="Not yet closed" tone="critical" icon={<Radio className="h-4 w-4" />} />
        <StatCard label="Critical / high" value={String(live.filter((i) => i.severity === "critical" || i.severity === "high").length).padStart(2, "0")} hint="By detection severity" tone="warning" icon={<AlertTriangle className="h-4 w-4" />} />
        <StatCard label="Refreshed" value="60 s" hint="Auto-updating feed" tone="satellite" icon={<Activity className="h-4 w-4" />} />
      </div>

      <div className="mt-5">
        <LoadState query={query} />
      </div>

      {live.length > 0 && (
        <div className="mt-3 grid gap-5 xl:grid-cols-[minmax(320px,0.8fr)_minmax(0,1.4fr)]">
          <Panel title="Incident feed" right={<Pill tone="bg-critical/12 text-critical border-critical/30">Live</Pill>} bodyClassName="p-0">
            <div className="divide-y divide-border">
              {live.map((i) => (
                <IncidentCard key={i.id} inc={i} active={i.id === active?.id} onSelect={() => setSelected(i.id)} />
              ))}
            </div>
          </Panel>

          <div className="space-y-5">
            <Panel title="Affected region" bodyClassName="p-0">
              {active ? (
                <IncidentsMap
                  data={[active]}
                  selected={active.id}
                  center={(active.centroid as [number, number]) ?? [30, 18]}
                  span={active.centroid ? 6 : 340}
                />
              ) : (
                <div className="p-10 text-center text-sm text-muted-foreground">Select an incident.</div>
              )}
            </Panel>
            {active && (
              <Panel title="Incident information">
                <div className="space-y-0">
                  <KV k="Region" v={active.region_label} />
                  <KV k="Detection date" v={active.detection_date ? formatDateTime(active.detection_date) : "—"} />
                  <KV k="Status" v={active.status} />
                  <KV k="Severity" v={sevLabel[active.severity] ?? active.severity} />
                  <KV k="Confidence" v={active.confidence_band} />
                  <KV k="Area" v={active.area_km2 != null ? `${Math.round(active.area_km2).toLocaleString()} km²` : "—"} />
                </div>
              </Panel>
            )}
          </div>
        </div>
      )}

      <Disclaimer />
    </PublicShell>
  );
}

/* ---------------------------- Historical spills -------------------------- */

function HistoricalSpills() {
  const query = usePublicIncidents();
  const incidents = query.data ?? [];
  const [year, setYear] = useState("All years");

  const years = useMemo(
    () => [
      "All years",
      ...Array.from(
        new Set(incidents.map((i) => (i.detection_date ?? "").slice(0, 4)).filter(Boolean)),
      ).sort().reverse(),
    ],
    [incidents],
  );
  const rows = incidents
    .filter((i) => year === "All years" || (i.detection_date ?? "").startsWith(year))
    .slice()
    .sort((a, b) => (b.detection_date ?? "").localeCompare(a.detection_date ?? ""));

  const byYear = useMemo(() => {
    const m = new Map<string, number>();
    for (const i of incidents) {
      const y = (i.detection_date ?? "").slice(0, 4);
      if (y) m.set(y, (m.get(y) ?? 0) + 1);
    }
    return Array.from(m.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [incidents]);
  const maxYear = Math.max(1, ...byYear.map(([, n]) => n));

  return (
    <PublicShell title="Historical Spills" subtitle="Archive of released detections and closed incidents">
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
        <Panel
          title="Detection archive"
          right={
            <select aria-label="Filter by year" value={year} onChange={(e) => setYear(e.target.value)} className="h-7 border border-input bg-background px-2 text-xs">
              {years.map((y) => <option key={y}>{y}</option>)}
            </select>
          }
          bodyClassName="p-0"
        >
          <LoadState query={query} />
          {incidents.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-left text-sm">
                <thead className="bg-muted/60 text-[10px] tracking-[0.12em] text-muted-foreground uppercase">
                  <tr>
                    <th className="px-4 py-2.5 font-semibold">Region</th>
                    <th className="px-4 py-2.5 font-semibold">Detected</th>
                    <th className="px-4 py-2.5 font-semibold">Area</th>
                    <th className="px-4 py-2.5 font-semibold">Status</th>
                    <th className="px-4 py-2.5 text-right font-semibold">Severity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {rows.map((i) => (
                    <tr key={i.id} className="transition-colors hover:bg-muted/40">
                      <td className="px-4 py-3 font-medium">{i.region_label}</td>
                      <td className="mono-num px-4 py-3 text-xs text-muted-foreground">
                        {i.detection_date ? formatDate(i.detection_date) : "—"}
                      </td>
                      <td className="mono-num px-4 py-3 text-xs">
                        {i.area_km2 != null ? `${Math.round(i.area_km2).toLocaleString()} km²` : "—"}
                      </td>
                      <td className="px-4 py-3"><Pill>{i.status}</Pill></td>
                      <td className="px-4 py-3 text-right">
                        <Pill severity={i.severity === "moderate" ? "medium" : (i.severity as "critical" | "high" | "low")}>
                          {sevLabel[i.severity] ?? i.severity}
                        </Pill>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.length === 0 && (
                <div className="px-4 py-10 text-center text-sm text-muted-foreground">No records match this filter.</div>
              )}
            </div>
          )}
        </Panel>

        <div className="space-y-5">
          <Panel title="Detections per year" bodyClassName="space-y-2.5">
            {byYear.length === 0 ? (
              <p className="text-xs text-muted-foreground">No dated records yet.</p>
            ) : (
              byYear.map(([y, n]) => (
                <div key={y} className="flex items-center gap-3">
                  <span className="mono-num w-10 text-xs text-muted-foreground">{y}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                    <div className="h-full rounded-full bg-ocean" style={{ width: `${(n / maxYear) * 100}%` }} />
                  </div>
                  <span className="mono-num w-6 text-right text-xs font-semibold">{n}</span>
                </div>
              ))
            )}
          </Panel>
          <Panel title="Archive summary">
            <div className="space-y-0">
              <KV k="Records published" v={String(incidents.length)} />
              <KV k="Sources" v="Sentinel-1 SAR" />
              <KV k="Update cadence" v="On detection" />
            </div>
          </Panel>
        </div>
      </div>

      <Disclaimer />
    </PublicShell>
  );
}

/* ------------------------------- Insights -------------------------------- */

function GlobalInsights() {
  const maxOcean = Math.max(...insights.byOcean.map((o) => o.value));
  const totalSeverity = insights.bySeverity.reduce((sum, s) => sum + s.value, 0);
  const maxYear = Math.max(...insights.byYear.map((y) => y.spills));

  return (
    <PublicShell title="Global Insights" subtitle="Trends and statistics across published oil spill detections">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Detections 2026" value="48" hint="+9% vs 2025" tone="satellite" />
        <StatCard label="Median slick area" value="8.4 km²" hint="Across published events" tone="default" />
        <StatCard label="Source identified" value="61%" hint="Of investigated incidents" tone="resolved" />
        <StatCard label="Critical events" value="09" hint="High coastal impact risk" tone="critical" />
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Panel title="Detections by ocean" bodyClassName="space-y-3">
          {insights.byOcean.map((o) => (
            <div key={o.name}>
              <div className="flex items-baseline justify-between text-xs">
                <span className="text-muted-foreground">{o.name}</span>
                <span className="mono-num font-semibold">{o.value}</span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full rounded-full bg-satellite" style={{ width: `${(o.value / maxOcean) * 100}%` }} />
              </div>
            </div>
          ))}
        </Panel>

        <Panel title="Severity distribution" bodyClassName="space-y-3">
          <div className="flex h-3 overflow-hidden rounded-full">
            {insights.bySeverity.map((s) => (
              <div
                key={s.name}
                className={
                  s.name === "Critical" ? "bg-critical" : s.name === "High" ? "bg-warning" : s.name === "Medium" ? "bg-caution" : "bg-resolved"
                }
                style={{ width: `${(s.value / totalSeverity) * 100}%` }}
              />
            ))}
          </div>
          <div className="grid grid-cols-2 gap-2">
            {insights.bySeverity.map((s) => (
              <div key={s.name} className="flex items-center justify-between border border-border p-2.5">
                <span className="text-xs text-muted-foreground">{s.name}</span>
                <span className="mono-num text-sm font-semibold">{Math.round((s.value / totalSeverity) * 100)}%</span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <Panel className="mt-5" title="Yearly trend" bodyClassName="p-6">
        <div className="flex h-48 items-end gap-4">
          {insights.byYear.map((y) => (
            <div key={y.year} className="flex flex-1 flex-col items-center gap-2">
              <span className="mono-num text-xs font-semibold">{y.spills}</span>
              <div className="w-full rounded-t-sm bg-ocean/80" style={{ height: `${(y.spills / maxYear) * 100}%` }} />
              <span className="mono-num text-[11px] text-muted-foreground">{y.year}</span>
            </div>
          ))}
        </div>
      </Panel>

      <Disclaimer />
    </PublicShell>
  );
}

export { GlobalMap, LiveIncidents, HistoricalSpills, GlobalInsights };
