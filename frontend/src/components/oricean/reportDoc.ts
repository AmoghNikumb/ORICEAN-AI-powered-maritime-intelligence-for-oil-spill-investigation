/**
 * Shared oil-spill report document builders.
 *
 * Extracted from AnalystPages so the Officer Reports view can produce the same
 * artefact. Pure functions — no React, no side effects except the download /
 * print helpers at the bottom.
 */

import type {
  AttributionResult,
  InferenceResult,
  InvestigationRead,
  SourceReconstructionResult,
} from "@/lib/api";
import { formatDate, formatDateTime } from "@/lib/oricean-data";

export const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));

export interface ReportParts {
  investigation: InvestigationRead;
  result: InferenceResult | null;
  sourceRecon: SourceReconstructionResult | null;
  attribution: AttributionResult | null;
  generatedAt: string;
}

/** Full standalone HTML report — opened in a new tab and sent to the browser's PDF printer. */
export function buildReportHtml({
  investigation,
  result,
  sourceRecon,
  attribution,
  generatedAt,
}: ReportParts): string {
  const dets = result?.spill_detections ?? [];
  const cands = attribution?.candidates ?? [];
  const qualified = cands.filter((c) => c.qualified);
  const shortlist = (qualified.length ? qualified : cands).slice(0, 10);

  const detRows = dets
    .map(
      (d, i) => `<tr>
        <td>${i + 1}</td>
        <td>${((d.mean_confidence ?? 0) * 100).toFixed(0)}%</td>
        <td>${((d.max_confidence ?? 0) * 100).toFixed(0)}%</td>
        <td>${d.tile_count ?? "—"}</td>
        <td>${d.area_km2 != null ? Math.round(d.area_km2).toLocaleString() + " km²" : "—"}</td>
      </tr>`,
    )
    .join("");

  const candRows = shortlist
    .map(
      (c) => `<tr>
        <td>${c.rank}</td>
        <td>${esc(c.name ?? "Unknown")}<br><span class="mut">${esc([c.vessel_type, c.flag].filter(Boolean).join(" · "))}</span></td>
        <td>${esc(c.mmsi)}</td>
        <td>${c.attribution_score.toFixed(1)}</td>
        <td>${c.min_distance_km.toFixed(1)} km</td>
        <td>${c.time_diff_hours != null ? c.time_diff_hours.toFixed(1) + " h" : "—"}</td>
        <td>${c.qualified ? esc((c.qualification_reasons ?? []).join("; ")) : '<span class="mut">below evidence threshold</span>'}</td>
      </tr>`,
    )
    .join("");

  const win =
    attribution?.release_window_start && attribution?.release_window_end
      ? `${formatDateTime(attribution.release_window_start)} → ${formatDateTime(attribution.release_window_end)} UTC`
      : "—";

  return `<!doctype html><html><head><meta charset="utf-8"><title>ORICEAN Report — ${esc(investigation.name)}</title>
<style>
  *{box-sizing:border-box} body{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a;margin:0;padding:32px 40px;max-width:900px}
  h1{font-size:20px;margin:0 0 2px} h2{font-size:14px;margin:22px 0 6px;border-bottom:2px solid #0ea5e9;padding-bottom:3px;color:#0369a1}
  .sub{color:#64748b;font-size:12px;margin-bottom:16px}
  table{border-collapse:collapse;width:100%;margin:6px 0;font-size:12px}
  th,td{border:1px solid #e2e8f0;padding:5px 8px;text-align:left;vertical-align:top}
  th{background:#f1f5f9;font-weight:600}
  .mut{color:#94a3b8} .kv{display:grid;grid-template-columns:180px 1fr;gap:2px 10px;font-size:12px}
  .kv div:nth-child(odd){color:#64748b}
  .pill{display:inline-block;background:#e0f2fe;color:#0369a1;border-radius:3px;padding:1px 6px;font-size:11px;margin-right:4px}
  .disclaimer{margin-top:24px;padding:10px 12px;background:#fef3c7;border:1px solid #fde68a;border-radius:4px;font-size:11px;color:#78350f}
  .sign{margin-top:20px;border-top:1px dashed #cbd5e1;padding-top:12px;font-size:12px}
  @media print{body{padding:0}}
</style></head><body>
<h1>ORICEAN — Oil Spill Attribution Report</h1>
<div class="sub">${esc(investigation.name)} · ID ${esc(investigation.id)} · generated ${formatDateTime(generatedAt)}</div>

<h2>1 · Investigation details</h2>
<div class="kv">
  <div>Name</div><div>${esc(investigation.name)}</div>
  <div>Investigation ID</div><div>${esc(investigation.id)}</div>
  <div>Date range</div><div>${esc(investigation.start_date)} → ${esc(investigation.end_date)}</div>
  <div>Status</div><div>${esc(investigation.status)}</div>
  <div>Created</div><div>${investigation.created_at ? formatDateTime(investigation.created_at) : "—"}</div>
</div>

<h2>2 · Satellite / SAR data</h2>
${
  result
    ? `<div class="kv">
        <div>Sensor</div><div>Sentinel-1 SAR (Google Earth Engine)</div>
        <div>Model</div><div>${esc(result.model_version || "—")}${result.mock_mode ? " (demo mode)" : ""}</div>
        <div>Tiles processed</div><div>${result.total_tiles_processed}</div>
        <div>Positive tiles</div><div>${result.positive_tile_count}</div>
        <div>Inference run</div><div>${result.inference_timestamp ? formatDateTime(result.inference_timestamp) : "—"}</div>
      </div>
      ${(result.warnings ?? []).length ? `<div style="margin-top:6px">${result.warnings.map((w) => `<span class="pill">${esc(w)}</span>`).join("")}</div>` : ""}`
    : "<p class='mut'>No SAR inference has been run for this investigation.</p>"
}

<h2>3 · Oil spill detection</h2>
${
  dets.length
    ? `<table><thead><tr><th>#</th><th>Mean conf.</th><th>Max conf.</th><th>Tiles</th><th>Area</th></tr></thead><tbody>${detRows}</tbody></table>`
    : "<p class='mut'>No spill regions detected.</p>"
}

<h2>4 · Source reconstruction</h2>
${
  sourceRecon
    ? `<div class="kv">
        <div>Method</div><div>${esc(sourceRecon.reconstruction_method)}</div>
        <div>Drift distance</div><div>${sourceRecon.drift_distance_km?.toFixed?.(1) ?? "—"} km</div>
        <div>Source confidence</div><div>${Math.round((sourceRecon.source_confidence ?? 0) * 100)}%</div>
        <div>Release window</div><div>${formatDateTime(sourceRecon.release_window_start)} → ${formatDateTime(sourceRecon.release_window_end)} UTC</div>
      </div>
      ${(sourceRecon.warnings ?? []).length ? `<div style="margin-top:6px">${sourceRecon.warnings.map((w) => `<span class="pill">${esc(w)}</span>`).join("")}</div>` : ""}`
    : "<p class='mut'>Source reconstruction has not been run for this investigation.</p>"
}

<h2>5 · Vessel attribution</h2>
${
  attribution && cands.length
    ? `<div class="kv">
        <div>AIS data source</div><div>${esc(attribution.data_source)}${attribution.data_source === "gfw" ? " (Global Fishing Watch, historical)" : ""}</div>
        <div>Search window</div><div>${win}</div>
        <div>Vessels scored</div><div>${attribution.total_candidates}</div>
        <div>Qualified candidates</div><div>${attribution.qualified_count}</div>
      </div>
      <table style="margin-top:8px"><thead><tr><th>Rank</th><th>Vessel</th><th>MMSI</th><th>Score</th><th>Closest dist.</th><th>Δt detection</th><th>Why it qualified</th></tr></thead><tbody>${candRows}</tbody></table>`
    : "<p class='mut'>Vessel attribution has not been run for this investigation.</p>"
}

<div class="disclaimer">
  This report ranks vessels by the <em>consistency</em> of their historical AIS activity with the detected spill and probable
  source corridor. A ranking is <strong>not</strong> a determination that any vessel caused the spill or a finding of liability.
  Confidence figures are model outputs and require analyst review.
</div>

<div class="sign">
  <strong>Sign-off:</strong> _____________________________ &nbsp;&nbsp; Date: ${formatDate(generatedAt)}<br>
  Prepared via ORICEAN · for Monitoring Officer handoff.
</div>
</body></html>`;
}

/** Machine-readable evidence bundle — base64 tile imagery stripped, everything else retained. */
export function buildEvidenceBundle({
  investigation,
  result,
  sourceRecon,
  attribution,
  generatedAt,
}: ReportParts) {
  const leanTiles = (result?.tile_results ?? []).map((t) => ({
    tile_id: t.tile_id,
    label: t.label,
    confidence: t.confidence,
    is_positive: t.is_positive,
    bounds: t.bounds,
    acquisition_date: t.acquisition_date,
    pixel_stats: t.pixel_stats ?? null,
  }));
  return {
    schema: "oricean.evidence-bundle/v1",
    generated_at: generatedAt,
    investigation: {
      id: investigation.id,
      name: investigation.name,
      aoi_geojson: investigation.aoi_geojson,
      start_date: investigation.start_date,
      end_date: investigation.end_date,
      status: investigation.status,
      created_at: investigation.created_at,
    },
    sar_detection: result
      ? {
          model_version: result.model_version,
          model_variant: result.model_variant,
          inference_timestamp: result.inference_timestamp,
          total_tiles_processed: result.total_tiles_processed,
          positive_tile_count: result.positive_tile_count,
          mock_mode: result.mock_mode,
          warnings: result.warnings,
          spill_detections: result.spill_detections,
          tiles: leanTiles,
        }
      : null,
    source_reconstruction: sourceRecon ?? null,
    vessel_attribution: attribution
      ? { ...attribution, note: "candidate.track / track_points are real historical AIS positions" }
      : null,
  };
}

export function downloadJson(name: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Open the printable report in a new tab and trigger the print dialog. */
export function openPrintableReport(parts: ReportParts) {
  const html = buildReportHtml(parts);
  const w = window.open("", "_blank", "width=900,height=1000");
  if (!w) {
    alert("Allow pop-ups for this site to open the printable report.");
    return;
  }
  w.document.open();
  w.document.write(html);
  w.document.close();
  w.focus();
  setTimeout(() => {
    try {
      w.print();
    } catch {
      /* user can still print manually */
    }
  }, 500);
}
