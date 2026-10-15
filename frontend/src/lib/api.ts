/**
 * ORICEAN API client
 *
 * Central API layer for all backend communication.
 * Base URL is read from VITE_API_BASE_URL env var (defaults to localhost:8010).
 * All responses follow the { data: ... } / { error: ..., detail: ... } envelope.
 */

const BASE_URL = (import.meta.env["VITE_API_BASE_URL"] as string) ?? "http://localhost:8010";

// ---------------------------------------------------------------------------
// Types — mirroring backend Pydantic schemas exactly
// ---------------------------------------------------------------------------

export type InvestigationStatus =
  | "pending"
  | "running_gee"
  | "running_ml"
  | "complete"
  | "failed";

export interface InvestigationRead {
  id: string;
  name: string;
  aoi_geojson: object;
  start_date: string;
  end_date: string;
  status: InvestigationStatus;
  ml_result: InferenceResult | null;
  ml_run_at: string | null;
  error_message: string | null;
  source_reconstruction_result: SourceReconstructionResult | null;
  attribution_result: AttributionResult | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface InvestigationCreate {
  name: string;
  aoi_geojson: object;
  start_date: string;
  end_date: string;
}

export interface PaginatedInvestigations {
  items: InvestigationRead[];
  total: number;
  page: number;
  page_size: number;
}

export interface PixelStats {
  min_value: number;
  max_value: number;
  mean_value: number;
  std_value: number;
}

export interface DebugInfo {
  input_dimensions: [number, number];           // [height, width] of raw tile
  preprocessing_steps: string[];
  model_input_dims: [number, number, number];   // [C, H, W]
}

export interface TilePrediction {
  tile_id: string;
  label: "No Oil" | "Oil Spill";
  confidence: number;
  is_positive: boolean;
  bounds: [number, number, number, number]; // min_lon, min_lat, max_lon, max_lat
  acquisition_date: string;
  mock: boolean;
  // new optional fields (Requirements 12.1, 12.5, 12.6, 13.3)
  image_url?: string | null;
  image_data_b64?: string | null;
  pixel_stats?: PixelStats | null;
  debug_info?: DebugInfo | null;
  gradcam_b64?: string | null;
}

export interface SpillDetection {
  detection_id: string;
  polygon_geojson: GeoJSONPolygon | GeoJSONMultiPolygon;
  mean_confidence: number;
  max_confidence: number;
  tile_count: number;
  area_km2: number | null;
}

export interface GeoJSONPolygon {
  type: "Polygon";
  coordinates: number[][][];
}

export interface GeoJSONMultiPolygon {
  type: "MultiPolygon";
  coordinates: number[][][][];
}

export interface InferenceResult {
  investigation_id: string;
  model_version: string;
  model_variant: string;
  inference_timestamp: string;
  total_tiles_processed: number;
  positive_tile_count: number;
  spill_detections: SpillDetection[];
  tile_results: TilePrediction[];
  mock_mode: boolean;
  warnings: string[];
  // new optional field (Requirement 12.4)
  sar_scene_url?: string | null;
}

// ---------------------------------------------------------------------------
// Source Reconstruction & Attribution types
// ---------------------------------------------------------------------------

export interface SourceReconstructionResult {
  investigation_id: string;
  source_corridor_geojson: object;  // GeoJSON Polygon or MultiPolygon
  release_window_start: string;     // ISO-8601 UTC
  release_window_end: string;       // ISO-8601 UTC
  source_confidence: number;        // [0.0, 1.0]
  reconstruction_method: string;
  drift_distance_km: number;
  warnings: string[];
  run_at: string;
}

export interface ScoreBreakdown {
  spatial_score: number;
  temporal_score: number;
  trajectory_score: number;
  time_near_region_score: number;
  weights: {
    spatial: number;
    temporal: number;
    trajectory: number;
    time_near_region: number;
  };
}

export interface CandidateVessel {
  rank: number;
  mmsi: string;
  name: string | null;
  imo: string | null;
  vessel_type: string | null;
  flag: string | null;
  attribution_score: number;    // [0.0, 100.0]
  min_distance_km: number;
  entry_time: string | null;
  exit_time: string | null;
  closest_timestamp: string | null;
  time_diff_hours: number | null;
  num_ais_positions: number;
  trajectory_overlap: number;        // [0.0, 1.0] — fraction of positions inside the buffer
  time_near_region_hours: number;
  points_near_source: number;        // in-window AIS fixes within the corridor buffer
  qualified: boolean;                // cleared every evidence gate
  qualification_reasons: string[];   // one string per gate passed
  track: Array<[number, number]>;    // (lon, lat) positions for the map
  track_points: Array<[number, number, string]>; // (lon, lat, iso-utc) — for the ship-route replay
  score_breakdown: ScoreBreakdown;
  evidence: string[];
}

export interface AttributionResult {
  investigation_id: string;
  candidates: CandidateVessel[];
  total_candidates: number;          // total vessels scored
  qualified_count: number;           // vessels clearing every evidence gate
  ais_provider_configured: boolean;
  data_source: "gfw" | "marinecadastre" | "aisstream" | "none" | string;
  search_region_geojson: object | null;
  release_window_start: string | null;
  release_window_end: string | null;
  lookback_hours: number | null;
  warnings: string[];
  run_at: string;
}

export interface MLHealthResponse {
  status: "ready" | "not_loaded" | "mock_mode";
  model_loaded: boolean;
  model_version: string | null;
  device: string;
}

export interface MLInferenceRequest {
  investigation_id: string;
  aoi_geojson: object;
  start_date: string;
  end_date: string;
  model_variant?: "primary" | "student";
  confidence_threshold?: number;
  // new optional field (Requirement 8.5)
  stride?: number;
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly error: string,
    public readonly detail: string,
  ) {
    super(`${error}: ${detail}`);
    this.name = "ApiError";
  }
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (body != null) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers,
    body: body != null ? JSON.stringify(body) : undefined,
  });

  if (res.status === 204) return undefined as T;

  const json = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new ApiError(
      res.status,
      (json as { error?: string }).error ?? "Error",
      (json as { detail?: string }).detail ?? res.statusText,
    );
  }

  // Success responses are wrapped in { data: ... }
  return (json as { data: T }).data ?? (json as T);
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body: unknown) => request<T>("POST", path, body);
const patch = <T>(path: string, body: unknown) => request<T>("PATCH", path, body);
const del = (path: string) => request<void>("DELETE", path);

// ---------------------------------------------------------------------------
// Investigations API
// ---------------------------------------------------------------------------

export const investigationsApi = {
  create: (data: InvestigationCreate) =>
    post<InvestigationRead>("/investigations/", data),

  list: (page = 1, pageSize = 20, slim = false) =>
    get<PaginatedInvestigations>(
      `/investigations/?page=${page}&page_size=${pageSize}${slim ? "&slim=true" : ""}`,
    ),

  get: (id: string) =>
    get<InvestigationRead>(`/investigations/${id}`),

  update: (id: string, data: Partial<InvestigationCreate>) =>
    patch<InvestigationRead>(`/investigations/${id}`, data),

  delete: (id: string) => del(`/investigations/${id}`),
};

// ---------------------------------------------------------------------------
// ML API
// ---------------------------------------------------------------------------

export const mlApi = {
  runInference: (req: MLInferenceRequest) =>
    post<InferenceResult>("/api/ml/run-inference", req),

  health: () => get<MLHealthResponse>("/api/ml/health"),
};

// ---------------------------------------------------------------------------
// Source Reconstruction API
// ---------------------------------------------------------------------------

export const sourceApi = {
  reconstruct: (investigationId: string, opts?: { force_rerun?: boolean }) =>
    post<SourceReconstructionResult>("/api/source/reconstruct", {
      investigation_id: investigationId,
      ...opts,
    }),
};

// ---------------------------------------------------------------------------
// Attribution API
// ---------------------------------------------------------------------------

export const attributionApi = {
  run: (investigationId: string, opts?: { force_rerun?: boolean }) =>
    post<AttributionResult>("/api/attribution/run", {
      investigation_id: investigationId,
      ...opts,
    }),

  get: (investigationId: string) =>
    get<AttributionResult>(`/api/attribution/${investigationId}`),
};

// ---------------------------------------------------------------------------
// Officer alerts API
// ---------------------------------------------------------------------------

export interface AlertRead {
  id: string;
  investigation_id: string | null;
  created_at: string;
  severity: "critical" | "high" | "moderate" | "low" | "test" | string;
  headline: string;
  region_label: string | null;
  detection_time: string | null;
  confidence: number | null;
  positive_tiles: number | null;
  area_km2: number | null;
  investigation_url: string | null;
  delivery_status: "sent" | "partial" | "failed" | "skipped" | "not_configured" | string;
  delivery_detail: string | null;
  recipients_count: number;
}

export interface NotificationSettings {
  enabled: boolean;
  min_confidence: number;
  min_positive_tiles: number;
  recipients: string[];
  updated_at: string | null;
}

export interface AlertStatus {
  package_installed: boolean;
  account_configured: boolean;
  from_number_set: boolean;
  recipients_count: number;
  alerting_enabled: boolean;
}

export const alertsApi = {
  list: (limit = 50) => get<AlertRead[]>(`/api/alerts?limit=${limit}`),
  status: () => get<AlertStatus>("/api/alerts/status"),
  getSettings: () => get<NotificationSettings>("/api/alerts/settings"),
  updateSettings: (patch: Partial<Omit<NotificationSettings, "updated_at">>) =>
    request<NotificationSettings>("PUT", "/api/alerts/settings", patch),
  sendTest: () => post<AlertRead>("/api/alerts/test", {}),
};

// ---------------------------------------------------------------------------
// Public incident feed (sanitised — safe for unauthenticated viewers)
// ---------------------------------------------------------------------------

export interface PublicIncident {
  id: string;
  region_label: string;
  status: string;
  detection_date: string | null;
  centroid: [number, number] | null;
  spill_region_geojson: GeoJSONPolygon | GeoJSONMultiPolygon | null;
  severity: "critical" | "high" | "moderate" | "low" | string;
  confidence_band: "high" | "moderate" | "low" | string;
  area_km2: number | null;
  updated_at: string | null;
}

export const publicApi = {
  incidents: () => get<PublicIncident[]>("/api/public/incidents"),
  incident: (id: string) => get<PublicIncident>(`/api/public/incidents/${id}`),
};

// ---------------------------------------------------------------------------
// Officer case workflow
// ---------------------------------------------------------------------------

export interface ChecklistItem {
  label: string;
  done: boolean;
}

export interface OfficerCase {
  investigation_id: string;
  review_decision: "pending" | "approved" | "returned" | "escalated" | string;
  review_notes: string | null;
  reviewed_at: string | null;
  response_status: "not_started" | "responding" | "contained" | "closed" | string;
  response_checklist: ChecklistItem[];
  report_status: "draft" | "awaiting_review" | "approved" | "distributed" | string;
  created_at: string;
  updated_at: string | null;
}

export interface OfficerCaseSummary extends OfficerCase {
  name: string;
  status: string;
  severity: "critical" | "high" | "moderate" | "low" | string;
  max_confidence: number;
  positive_tiles: number;
  area_km2: number;
  detection_date: string | null;
  has_source_reconstruction: boolean;
  has_attribution: boolean;
  alert_count: number;
}

export type OfficerCaseUpdate = Partial<
  Pick<
    OfficerCase,
    "review_decision" | "review_notes" | "response_status" | "response_checklist" | "report_status"
  >
>;

export const officerApi = {
  listCases: () => get<OfficerCaseSummary[]>("/api/officer/cases"),
  getCase: (investigationId: string) =>
    get<OfficerCaseSummary>(`/api/officer/cases/${investigationId}`),
  updateCase: (investigationId: string, body: OfficerCaseUpdate) =>
    patch<OfficerCaseSummary>(`/api/officer/cases/${investigationId}`, body),
};

// ---------------------------------------------------------------------------
// App health
// ---------------------------------------------------------------------------

export const healthApi = {
  check: () =>
    get<{ status: string; model_loaded: boolean; db_connected: boolean }>("/health"),
};

export { ApiError };
