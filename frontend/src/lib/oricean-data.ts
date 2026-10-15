export type Severity = "critical" | "high" | "medium" | "low";
export type SpillStatus = "confirmed" | "investigating" | "potential" | "resolved";

export type LonLat = [number, number];

export interface Spill {
  id: string;
  name: string;
  region: string;
  ocean: string;
  country: string;
  lon: number;
  lat: number;
  detectedAt: string;
  areaKm2: number;
  lengthKm: number;
  shape: string;
  detectionConfidence: number;
  sourceConfidence: number;
  severity: Severity;
  status: SpillStatus;
  coastalRisk: "HIGH" | "MEDIUM" | "LOW";
  satellite: string;
  polygon: LonLat[];
  summary: string;
  updates: { at: string; text: string }[];
}

export interface Vessel {
  id: string;
  name: string;
  mmsi: string;
  imo: string;
  type: string;
  flag: string;
  score: number;
  breakdown: { spatial: number; temporal: number; trajectory: number; aisAnomaly: number };
  evidence: string[];
  entryIntoRegion: string;
  speedPattern: string;
  courseChanges: string;
  aisGap: string;
  proximityKm: number;
  track: LonLat[];
}

function ring(lon: number, lat: number, rx: number, ry: number, rot = 0, n = 16): LonLat[] {
  const out: LonLat[] = [];
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    const x = Math.cos(t) * rx;
    const y = Math.sin(t) * ry;
    const w = 1 + 0.14 * Math.sin(t * 3 + 1.2);
    out.push([
      lon + (x * Math.cos(rot) - y * Math.sin(rot)) * w,
      lat + (x * Math.sin(rot) + y * Math.cos(rot)) * w,
    ]);
  }
  return out;
}

export const spills: Spill[] = [
  {
    id: "ORC-2026-0412",
    name: "Arabian Sea Oil Spill",
    region: "Arabian Sea — 210 NM W of Mumbai",
    ocean: "Indian Ocean",
    country: "India",
    lon: 69.4,
    lat: 18.6,
    detectedAt: "2026-08-28T06:42:00Z",
    areaKm2: 18.7,
    lengthKm: 21.4,
    shape: "Elongated linear slick",
    detectionConfidence: 92,
    sourceConfidence: 87,
    severity: "critical",
    status: "investigating",
    coastalRisk: "HIGH",
    satellite: "Sentinel-1A IW GRDH",
    polygon: ring(69.4, 18.6, 0.42, 0.13, 0.5),
    summary:
      "A linear slick consistent with operational discharge was detected in SAR imagery 210 NM west of Mumbai. Backtracking places the release inside a transit corridor used by product tankers.",
    updates: [
      { at: "2026-08-28T07:10:00Z", text: "Detection confirmed by duty analyst; case opened." },
      { at: "2026-08-28T11:30:00Z", text: "Source region reconstructed with 78% confidence." },
      { at: "2026-08-29T08:05:00Z", text: "Coast guard notified — coastal risk assessed HIGH." },
    ],
  },
  {
    id: "ORC-2026-0409",
    name: "Gulf of Kutch Slick",
    region: "Gulf of Kutch — Kandla approaches",
    ocean: "Indian Ocean",
    country: "India",
    lon: 69.1,
    lat: 22.5,
    detectedAt: "2026-08-26T04:15:00Z",
    areaKm2: 6.2,
    lengthKm: 9.1,
    shape: "Patchy fragmented",
    detectionConfidence: 81,
    sourceConfidence: 54,
    severity: "high",
    status: "confirmed",
    coastalRisk: "HIGH",
    satellite: "Sentinel-1B IW GRDH",
    polygon: ring(69.1, 22.5, 0.2, 0.11, -0.3),
    summary:
      "Fragmented slick close to the Kandla port approaches. Proximity to mangrove habitat elevates the environmental risk.",
    updates: [{ at: "2026-08-26T09:00:00Z", text: "Response vessel dispatched for verification." }],
  },
  {
    id: "ORC-2026-0398",
    name: "Strait of Malacca Discharge",
    region: "Strait of Malacca — traffic separation scheme",
    ocean: "Pacific Ocean",
    country: "Malaysia",
    lon: 100.9,
    lat: 3.2,
    detectedAt: "2026-08-24T22:05:00Z",
    areaKm2: 11.4,
    lengthKm: 16.8,
    shape: "Linear along traffic lane",
    detectionConfidence: 88,
    sourceConfidence: 71,
    severity: "high",
    status: "investigating",
    coastalRisk: "MEDIUM",
    satellite: "Sentinel-1A IW GRDH",
    polygon: ring(100.9, 3.2, 0.35, 0.1, 0.9),
    summary: "Classic night-time operational discharge signature along the northbound lane.",
    updates: [{ at: "2026-08-25T02:40:00Z", text: "Candidate vessels correlated from AIS archive." }],
  },
  {
    id: "ORC-2026-0387",
    name: "North Sea Platform Sheen",
    region: "North Sea — Block 21/9",
    ocean: "Atlantic Ocean",
    country: "United Kingdom",
    lon: 1.8,
    lat: 57.4,
    detectedAt: "2026-08-21T09:55:00Z",
    areaKm2: 3.1,
    lengthKm: 4.6,
    shape: "Radial sheen",
    detectionConfidence: 67,
    sourceConfidence: 42,
    severity: "medium",
    status: "potential",
    coastalRisk: "LOW",
    satellite: "Sentinel-1B IW GRDH",
    polygon: ring(1.8, 57.4, 0.16, 0.12, 0),
    summary: "Low-contrast dark formation near a producing platform; look-alike (low wind) not excluded.",
    updates: [],
  },
  {
    id: "ORC-2026-0361",
    name: "Gulf of Mexico Spill",
    region: "Gulf of Mexico — 90 NM SE of Galveston",
    ocean: "Atlantic Ocean",
    country: "United States",
    lon: -93.6,
    lat: 27.4,
    detectedAt: "2026-08-14T16:30:00Z",
    areaKm2: 24.3,
    lengthKm: 27.9,
    shape: "Feathered plume",
    detectionConfidence: 94,
    sourceConfidence: 66,
    severity: "critical",
    status: "confirmed",
    coastalRisk: "MEDIUM",
    satellite: "Sentinel-1A IW GRDH",
    polygon: ring(-93.6, 27.4, 0.5, 0.18, -0.4),
    summary: "Large plume with persistent signature across two consecutive passes.",
    updates: [{ at: "2026-08-15T12:00:00Z", text: "Containment booms deployed; response active." }],
  },
  {
    id: "ORC-2026-0344",
    name: "Gulf of Guinea Slick",
    region: "Gulf of Guinea — Bonny offshore",
    ocean: "Atlantic Ocean",
    country: "Nigeria",
    lon: 6.4,
    lat: 3.9,
    detectedAt: "2026-08-09T07:20:00Z",
    areaKm2: 14.9,
    lengthKm: 18.2,
    shape: "Elongated linear slick",
    detectionConfidence: 86,
    sourceConfidence: 59,
    severity: "high",
    status: "resolved",
    coastalRisk: "MEDIUM",
    satellite: "Sentinel-1B IW GRDH",
    polygon: ring(6.4, 3.9, 0.34, 0.12, 0.2),
    summary: "Slick dispersed naturally within 72 hours; case closed after verification pass.",
    updates: [{ at: "2026-08-13T10:00:00Z", text: "Case resolved — no residual signature detected." }],
  },
  {
    id: "ORC-2026-0321",
    name: "East China Sea Incident",
    region: "East China Sea — Zhoushan approaches",
    ocean: "Pacific Ocean",
    country: "China",
    lon: 124.2,
    lat: 29.8,
    detectedAt: "2026-07-30T01:10:00Z",
    areaKm2: 8.8,
    lengthKm: 12.5,
    shape: "Curved band",
    detectionConfidence: 79,
    sourceConfidence: 48,
    severity: "medium",
    status: "resolved",
    coastalRisk: "MEDIUM",
    satellite: "Sentinel-1A IW GRDH",
    polygon: ring(124.2, 29.8, 0.28, 0.12, 1.1),
    summary: "Curved band consistent with a vessel turning manoeuvre during discharge.",
    updates: [],
  },
  {
    id: "ORC-2026-0298",
    name: "Mediterranean Sea Slick",
    region: "Mediterranean — S of Crete",
    ocean: "Atlantic Ocean",
    country: "Greece",
    lon: 24.6,
    lat: 33.4,
    detectedAt: "2026-07-18T05:35:00Z",
    areaKm2: 5.4,
    lengthKm: 8.9,
    shape: "Linear",
    detectionConfidence: 74,
    sourceConfidence: 51,
    severity: "medium",
    status: "resolved",
    coastalRisk: "LOW",
    satellite: "Sentinel-1B IW GRDH",
    polygon: ring(24.6, 33.4, 0.24, 0.08, 0.4),
    summary: "Short-lived linear slick along an east–west transit route.",
    updates: [],
  },
  {
    id: "ORC-2026-0277",
    name: "Brazil Basin Sheen",
    region: "South Atlantic — Campos Basin",
    ocean: "Atlantic Ocean",
    country: "Brazil",
    lon: -39.4,
    lat: -22.6,
    detectedAt: "2026-07-05T13:45:00Z",
    areaKm2: 4.2,
    lengthKm: 6.1,
    shape: "Radial sheen",
    detectionConfidence: 71,
    sourceConfidence: 39,
    severity: "low",
    status: "resolved",
    coastalRisk: "LOW",
    satellite: "Sentinel-1A IW GRDH",
    polygon: ring(-39.4, -22.6, 0.2, 0.14, 0),
    summary: "Minor sheen near an FPSO; resolved after operator report.",
    updates: [],
  },
  {
    id: "ORC-2026-0255",
    name: "Sea of Japan Slick",
    region: "Sea of Japan — Niigata offshore",
    ocean: "Pacific Ocean",
    country: "Japan",
    lon: 137.4,
    lat: 38.6,
    detectedAt: "2026-06-22T20:15:00Z",
    areaKm2: 7.6,
    lengthKm: 10.8,
    shape: "Linear",
    detectionConfidence: 83,
    sourceConfidence: 44,
    severity: "medium",
    status: "resolved",
    coastalRisk: "LOW",
    satellite: "Sentinel-1B IW GRDH",
    polygon: ring(137.4, 38.6, 0.26, 0.1, -0.6),
    summary: "Linear slick detected during a night pass; no coastal impact recorded.",
    updates: [],
  },
];

export const primarySpill = spills[0];

/** Backtracked spill positions (index 0 = now, later = further back in time). */
export const backtrack: { hoursAgo: number; at: string; polygon: LonLat[]; drift: string }[] = [
  { hoursAgo: 0, at: "2026-08-28T06:42:00Z", polygon: primarySpill.polygon, drift: "Current detection" },
  { hoursAgo: 6, at: "2026-08-28T00:42:00Z", polygon: ring(69.28, 18.53, 0.34, 0.11, 0.5), drift: "Drift 0.31 m/s" },
  { hoursAgo: 12, at: "2026-08-27T18:42:00Z", polygon: ring(69.15, 18.47, 0.27, 0.1, 0.5), drift: "Drift 0.29 m/s" },
  { hoursAgo: 24, at: "2026-08-27T06:42:00Z", polygon: ring(68.94, 18.38, 0.2, 0.09, 0.45), drift: "Drift 0.26 m/s" },
  { hoursAgo: 48, at: "2026-08-26T06:42:00Z", polygon: ring(68.62, 18.24, 0.14, 0.08, 0.4), drift: "Drift 0.22 m/s" },
];

export const sourceRegion: LonLat[] = ring(68.55, 18.2, 0.36, 0.3, 0, 20);

export const currentField: { from: LonLat; to: LonLat }[] = Array.from({ length: 26 }, (_, i) => {
  const lon = 68.2 + (i % 6) * 0.32;
  const lat = 17.8 + Math.floor(i / 6) * 0.34;
  return { from: [lon, lat] as LonLat, to: [lon + 0.19, lat + 0.07] as LonLat };
});

export const vessels: Vessel[] = [
  {
    id: "v1",
    name: "MT ORION STAR",
    mmsi: "419001234",
    imo: "9456123",
    type: "Crude Oil Tanker",
    flag: "Panama",
    score: 87,
    breakdown: { spatial: 92, temporal: 84, trajectory: 89, aisAnomaly: 70 },
    evidence: [
      "Present inside probable source region at T-38h",
      "Strong temporal overlap with reconstructed release window",
      "Historical trajectory correlates with slick axis (r = 0.89)",
      "AIS transmission gap of 3h 12m during release window",
    ],
    entryIntoRegion: "2026-08-26T16:20Z",
    speedPattern: "Reduced from 12.4 kn to 6.1 kn for 2h 40m",
    courseChanges: "2 course alterations (>25°) inside region",
    aisGap: "3h 12m gap — 2026-08-26 18:40Z to 21:52Z",
    proximityKm: 4.2,
    track: [
      [67.6, 17.7],
      [68.1, 17.95],
      [68.45, 18.12],
      [68.62, 18.24],
      [68.95, 18.4],
      [69.4, 18.62],
      [69.95, 18.85],
    ],
  },
  {
    id: "v2",
    name: "MV BLUE HORIZON",
    mmsi: "563008876",
    imo: "9312884",
    type: "Product Tanker",
    flag: "Singapore",
    score: 64,
    breakdown: { spatial: 71, temporal: 68, trajectory: 58, aisAnomaly: 34 },
    evidence: [
      "Spatial proximity to source region (11.6 km)",
      "Partial temporal overlap with release window",
      "No AIS anomaly detected",
    ],
    entryIntoRegion: "2026-08-26T21:05Z",
    speedPattern: "Steady 13.8 kn transit",
    courseChanges: "1 minor alteration (12°)",
    aisGap: "None",
    proximityKm: 11.6,
    track: [
      [67.9, 18.7],
      [68.35, 18.6],
      [68.8, 18.5],
      [69.3, 18.42],
      [69.85, 18.34],
    ],
  },
  {
    id: "v3",
    name: "MT CORAL DAWN",
    mmsi: "477112390",
    imo: "9601772",
    type: "Chemical Tanker",
    flag: "Hong Kong",
    score: 38,
    breakdown: { spatial: 44, temporal: 41, trajectory: 30, aisAnomaly: 18 },
    evidence: ["Passed 24 km from source region", "Weak temporal overlap (edge of window)"],
    entryIntoRegion: "2026-08-27T04:40Z",
    speedPattern: "Steady 15.1 kn transit",
    courseChanges: "None",
    aisGap: "None",
    proximityKm: 24.1,
    track: [
      [68.0, 17.4],
      [68.6, 17.6],
      [69.2, 17.85],
      [69.9, 18.05],
    ],
  },
  {
    id: "v4",
    name: "MV SEA PIONEER",
    mmsi: "636019922",
    imo: "9288114",
    type: "Bulk Carrier",
    flag: "Liberia",
    score: 21,
    breakdown: { spatial: 26, temporal: 22, trajectory: 15, aisAnomaly: 12 },
    evidence: ["Distant transit (38 km)", "Minimal temporal overlap"],
    entryIntoRegion: "2026-08-27T12:10Z",
    speedPattern: "Steady 11.2 kn transit",
    courseChanges: "None",
    aisGap: "None",
    proximityKm: 38.4,
    track: [
      [68.3, 19.3],
      [68.9, 19.1],
      [69.5, 19.0],
      [70.1, 18.9],
    ],
  },
];

export interface TimelineEvent {
  icon: string;
  title: string;
  at: string;
  description: string;
  evidence: string;
  kind: "satellite" | "ai" | "ocean" | "vessel" | "alert" | "report";
}

export const timeline: TimelineEvent[] = [
  {
    icon: "satellite",
    kind: "satellite",
    title: "Satellite Image Acquired",
    at: "2026-08-28T06:42Z",
    description: "Sentinel-1A IW GRDH scene covering AOI-2026-0412 ingested and calibrated.",
    evidence: "Scene S1A_IW_GRDH_1SDV_20260828T064200",
  },
  {
    icon: "ai",
    kind: "ai",
    title: "Oil Slick Detected",
    at: "2026-08-28T06:58Z",
    description: "Segmentation model flagged a dark linear formation, 18.7 km², confidence 92%.",
    evidence: "Detection mask v3.2 · IoU 0.88 vs. analyst polygon",
  },
  {
    icon: "ocean",
    kind: "ocean",
    title: "Source Region Reconstructed",
    at: "2026-08-28T11:30Z",
    description: "48h backtracking using HYCOM currents and ERA5 winds; source confidence 78%.",
    evidence: "Drift model run #4417 · 500 particles",
  },
  {
    icon: "vessel",
    kind: "vessel",
    title: "Candidate Vessels Correlated",
    at: "2026-08-28T12:12Z",
    description: "4 vessels transited the probable source region in the release window.",
    evidence: "AIS archive query · 68.2–69.0E / 17.9–18.5N",
  },
  {
    icon: "alert",
    kind: "alert",
    title: "AIS Anomaly Detected",
    at: "2026-08-28T12:26Z",
    description: "MT ORION STAR shows a 3h 12m AIS transmission gap inside the release window.",
    evidence: "Gap 2026-08-26 18:40Z → 21:52Z",
  },
  {
    icon: "report",
    kind: "report",
    title: "Investigation Report Generated",
    at: "2026-08-29T09:05Z",
    description: "Draft attribution report prepared for Monitoring Officer review.",
    evidence: "Report ORC-RPT-0412-v1 · 14 pages",
  },
];

export interface Investigation {
  id: string;
  location: string;
  date: string;
  status: "Active" | "Awaiting Review" | "Submitted" | "Closed";
  confidence: number;
  priority: Severity;
}

export const investigations: Investigation[] = [
  { id: "ORC-2026-0412", location: "Arabian Sea — W of Mumbai", date: "2026-08-28", status: "Active", confidence: 87, priority: "critical" },
  { id: "ORC-2026-0409", location: "Gulf of Kutch", date: "2026-08-26", status: "Awaiting Review", confidence: 54, priority: "high" },
  { id: "ORC-2026-0398", location: "Strait of Malacca", date: "2026-08-24", status: "Active", confidence: 71, priority: "high" },
  { id: "ORC-2026-0387", location: "North Sea — Block 21/9", date: "2026-08-21", status: "Submitted", confidence: 42, priority: "medium" },
  { id: "ORC-2026-0361", location: "Gulf of Mexico", date: "2026-08-14", status: "Closed", confidence: 66, priority: "critical" },
];

export interface SatelliteScene {
  id: string;
  at: string;
  satellite: string;
  mode: string;
  coverage: number;
  hasDetection: boolean;
}

export const scenes: SatelliteScene[] = [
  { id: "S1A_20260828T0642", at: "2026-08-28 06:42Z", satellite: "Sentinel-1A", mode: "IW GRDH · VV+VH", coverage: 98, hasDetection: true },
  { id: "S1B_20260827T1810", at: "2026-08-27 18:10Z", satellite: "Sentinel-1B", mode: "IW GRDH · VV", coverage: 87, hasDetection: true },
  { id: "S1A_20260826T0630", at: "2026-08-26 06:30Z", satellite: "Sentinel-1A", mode: "IW GRDH · VV+VH", coverage: 74, hasDetection: false },
  { id: "S1B_20260825T1755", at: "2026-08-25 17:55Z", satellite: "Sentinel-1B", mode: "IW GRDH · VV", coverage: 91, hasDetection: false },
];

export interface Alert {
  id: string;
  spillId: string;
  title: string;
  region: string;
  severity: Severity;
  areaKm2: number;
  confidence: number;
  coastalRisk: "HIGH" | "MEDIUM" | "LOW";
  sourceIdentified: boolean;
  at: string;
  status: "New" | "Investigating" | "Response Active" | "Resolved";
}

export const alerts: Alert[] = [
  { id: "ALR-1188", spillId: "ORC-2026-0412", title: "Oil Spill — Arabian Sea", region: "Indian Ocean", severity: "critical", areaKm2: 24, confidence: 94, coastalRisk: "HIGH", sourceIdentified: true, at: "2026-08-28T06:58Z", status: "Investigating" },
  { id: "ALR-1187", spillId: "ORC-2026-0409", title: "Oil Spill — Gulf of Kutch", region: "Indian Ocean", severity: "high", areaKm2: 6.2, confidence: 81, coastalRisk: "HIGH", sourceIdentified: false, at: "2026-08-26T04:31Z", status: "Response Active" },
  { id: "ALR-1186", spillId: "ORC-2026-0398", title: "Discharge — Strait of Malacca", region: "Pacific Ocean", severity: "high", areaKm2: 11.4, confidence: 88, coastalRisk: "MEDIUM", sourceIdentified: true, at: "2026-08-24T22:20Z", status: "Investigating" },
  { id: "ALR-1185", spillId: "ORC-2026-0387", title: "Potential Sheen — North Sea", region: "Atlantic Ocean", severity: "medium", areaKm2: 3.1, confidence: 67, coastalRisk: "LOW", sourceIdentified: false, at: "2026-08-21T10:04Z", status: "New" },
  { id: "ALR-1184", spillId: "ORC-2026-0361", title: "Oil Spill — Gulf of Mexico", region: "Atlantic Ocean", severity: "critical", areaKm2: 24.3, confidence: 94, coastalRisk: "MEDIUM", sourceIdentified: true, at: "2026-08-14T16:44Z", status: "Response Active" },
  { id: "ALR-1183", spillId: "ORC-2026-0344", title: "Oil Spill — Gulf of Guinea", region: "Atlantic Ocean", severity: "high", areaKm2: 14.9, confidence: 86, coastalRisk: "MEDIUM", sourceIdentified: false, at: "2026-08-09T07:33Z", status: "Resolved" },
];

export const insights = {
  byOcean: [
    { name: "Indian Ocean", value: 12 },
    { name: "Atlantic Ocean", value: 18 },
    { name: "Pacific Ocean", value: 15 },
    { name: "Arctic Ocean", value: 2 },
    { name: "Southern Ocean", value: 1 },
  ],
  byYear: [
    { year: "2021", spills: 24 },
    { year: "2022", spills: 31 },
    { year: "2023", spills: 28 },
    { year: "2024", spills: 37 },
    { year: "2025", spills: 44 },
    { year: "2026", spills: 48 },
  ],
  bySeverity: [
    { name: "Critical", value: 9 },
    { name: "High", value: 17 },
    { name: "Medium", value: 14 },
    { name: "Low", value: 8 },
  ],
};

export const severityLabel: Record<Severity, string> = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
};

export function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

export function formatDateTime(iso: string) {
  const d = new Date(iso);
  return `${d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" })} · ${d
    .toISOString()
    .slice(11, 16)}Z`;
}
