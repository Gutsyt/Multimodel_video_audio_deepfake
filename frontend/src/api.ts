// Typed client for the avdd backend. Every value shown in the UI comes from these calls.
export type ModalityStatus = "available" | "missing" | "corrupted" | "unreliable" | "not_analyzed";
export type JobStatus = "queued" | "processing" | "completed" | "failed";
export type FusionState = "insufficient_evidence" | "single_modality" | "multimodal";
export type Decision = "flagged" | "not_flagged";
export type AssessmentStatus = Decision | "insufficient_evidence";
export type Basis = "multimodal" | "audio_only" | "visual_only" | "none";
export type MediaKind = "video" | "audio" | "image";

export interface ApiErrorBody { code: string; message: string; details?: Record<string, unknown> | null }
export interface ModelCard {
  name: string; version: string; modality: string; license: string; source_url: string;
  citation: string; trained_on: string; limitations: string[]; is_test_fixture: boolean;
}
export interface MediaInfo {
  kind?: MediaKind | null; container: string | null; duration_s: number | null; has_video: boolean; has_audio: boolean;
  width: number | null; height: number | null; fps: number | null; sample_rate: number | null;
  video_codec?: string | null; audio_codec?: string | null; channels?: number | null;
}
export type ClassValues = Record<string, number>;
export interface AudioSegment { start_s: number; end_s: number; score: number; logits?: ClassValues | null; class_probs?: ClassValues | null }
export interface FrameEvidence {
  frame_index: number; t_s: number | null; n_faces: number; score: number | null;
  face_box?: { x: number; y: number; w: number; h: number } | null;
  face_area_ratio?: number | null; crop_px?: number[] | null; logits?: ClassValues | null; class_probs?: ClassValues | null;
}
export interface OutputSemantics {
  class_labels: string[]; score_class: string; score_class_index: number; activation?: string;
  reference?: string; higher_means?: string;
}
export interface BranchDiagnostics {
  source?: string;
  input?: Record<string, string | number | boolean | null>;
  output_semantics?: OutputSemantics | null;
  windows?: Record<string, number | string>;
  scores?: Record<string, number | string>;
}
export interface ModalityResult {
  status: ModalityStatus; score: number | null; reasons: string[]; aggregation?: string | null;
  decision?: Decision | null; source?: "video" | "image" | null; error_code?: string | null;
  runtime_s?: number | null; model?: ModelCard | null; analyzed_limit_s?: number | null;
  diagnostics?: BranchDiagnostics | null;
  segments?: AudioSegment[] | null; n_windows?: number | null; max_window_score?: number | null;
  metrics?: Record<string, number> | null; sample_rate?: number | null;
  frames?: FrameEvidence[] | null; n_frames_sampled?: number | null; n_face_frames?: number | null;
  face_detector?: string | null; median_frame_score?: number | null; max_frame_score?: number | null;
  timestamps?: string | null;
}
export interface Fusion {
  method: string; rule?: string | null; fusion_performed?: boolean | null; configured_weights?: Record<string, number>;
  state: FusionState; fused_score: number | null;
  flagged_at_default_threshold: boolean | null; threshold: number;
  weights_used: Record<string, number>; branch_flags_at_default_threshold: Record<string, boolean>;
  excluded_modalities: Record<string, string>; weights_validated: boolean; threshold_validated: boolean;
  calibrated: boolean; notes: string[];
}
export interface Assessment {
  status: AssessmentStatus; basis: Basis; summary: string; score: number | null; score_kind: string | null;
  threshold: number; threshold_validated: boolean; branch_decisions: Record<string, string | null>;
  modalities_used: string[]; calibrated: boolean; not_authenticity_note: string;
}
export interface Interpretation {
  score_meaning: string; calibrated: boolean; validated_on_project_data: boolean;
  flag_meaning: string; not_forensic_proof: string;
}
export interface AnalysisResult {
  schema_version: string; avdd_version: string; created_utc: string; outcome: "analyzed" | "unreadable_media";
  media: MediaInfo; modalities: Record<string, ModalityResult>; fusion: Fusion; assessment?: Assessment | null;
  models: ModelCard[]; is_test_fixture: boolean; interpretation: Interpretation; analyzed_limit_s: number; total_runtime_s: number;
}
export interface AnalysisStatus {
  id: string; status: JobStatus; stage: string | null; stages: string[]; created_utc: string;
  updated_utc: string; result: AnalysisResult | null; error: ApiErrorBody | null;
}
export interface AnalysisSummary {
  id: string; status: JobStatus; created_utc: string; duration_s: number | null; media_kind?: string | null;
  assessment_status?: string | null; basis?: string | null; audio_decision?: string | null; visual_decision?: string | null;
  fusion_state: string | null;
  fused_score: number | null; flagged: boolean | null; audio_status: string | null; audio_score: number | null;
  visual_status: string | null; visual_score: number | null; is_test_fixture: boolean | null; error_code: string | null;
}
export interface ModelStatus {
  role: "audio" | "visual"; state: "loading" | "loaded" | "failed" | "disabled";
  error_code: string | null; card: ModelCard | null;
}
export interface AcceptedKind { label: string; extensions: string[] }
export interface Limits {
  max_upload_mb: number; max_duration_s: number; max_image_megapixels?: number | null;
  accepted_containers: string[]; accepted?: Partial<Record<MediaKind, AcceptedKind>>; frames_sampled: number; max_pending_jobs: number;
}
export interface Capabilities {
  avdd_version: string; models_state: "loading" | "ready" | "unavailable"; models: ModelStatus[];
  limits: Limits; retention: string; evaluation_status: string; stages: string[];
}

export class ApiRequestError extends Error {
  constructor(public status: number, public body: ApiErrorBody) {
    super(body.message);
  }
}

export const API_BASE: string = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const BASE = API_BASE;
const REQUEST_TIMEOUT_MS = 20_000;
const NETWORK_ERROR: ApiErrorBody = { code: "network_error", message: "Could not reach the analysis server." };

function errorFrom(status: number, text: string): ApiRequestError {
  try {
    const parsed = JSON.parse(text) as { error?: ApiErrorBody };
    if (parsed.error?.code) return new ApiRequestError(status, parsed.error);
  } catch { /* not JSON */ }
  return new ApiRequestError(status, { code: "http_error", message: `The server returned HTTP ${status}.` });
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS), ...init });
  } catch (e) {
    if (e instanceof DOMException && e.name === "TimeoutError") {
      throw new ApiRequestError(0, { code: "timeout", message: "The analysis server took too long to respond." });
    }
    throw new ApiRequestError(0, NETWORK_ERROR);
  }
  const text = await res.text();
  if (!res.ok) throw errorFrom(res.status, text);
  return (text ? JSON.parse(text) : undefined) as T;
}

export const getCapabilities = (): Promise<Capabilities> => request("/api/capabilities");
export const getAnalysis = (id: string): Promise<AnalysisStatus> => request(`/api/analyses/${id}`);
export const listAnalyses = (): Promise<{ items: AnalysisSummary[] }> => request("/api/analyses");
export const deleteAnalysis = (id: string): Promise<void> => request(`/api/analyses/${id}`, { method: "DELETE" });
export const deleteAllAnalyses = (): Promise<{ deleted: number }> => request("/api/analyses", { method: "DELETE" });
export const getReady = (): Promise<{ status: string }> => request("/api/ready");
export const reportUrl = (id: string): string => `${BASE}/api/analyses/${id}/report`;

/** Upload with REAL byte-level progress (XMLHttpRequest). The body is the raw file. */
export function uploadMedia(
  file: Blob,
  onProgress: (loaded: number, total: number) => void,
  signal?: AbortSignal,
): Promise<{ id: string; status: JobStatus }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/api/analyses`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded, e.total); };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else reject(errorFrom(xhr.status, xhr.responseText));
    };
    xhr.timeout = 15 * 60_000;
    xhr.ontimeout = () => reject(new ApiRequestError(0, { code: "timeout", message: "The upload timed out." }));
    xhr.onerror = () => reject(new ApiRequestError(0, NETWORK_ERROR));
    xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(file);
  });
}

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); reject(new DOMException("Stopped", "AbortError")); });
  });

/** Poll the real job status until it finishes. Tolerates brief network hiccups. */
export async function waitForAnalysis(
  id: string, onUpdate: (s: AnalysisStatus) => void, signal?: AbortSignal, intervalMs = 800, maxWaitMs = 10 * 60_000,
): Promise<AnalysisStatus> {
  let failures = 0;
  const started = Date.now();
  for (;;) {
    if (Date.now() - started > maxWaitMs) {
      throw new ApiRequestError(0, { code: "timeout", message: "The analysis is taking longer than expected." });
    }
    if (signal?.aborted) throw new DOMException("Stopped", "AbortError");
    try {
      const s = await getAnalysis(id);
      failures = 0;
      onUpdate(s);
      if (s.status === "completed" || s.status === "failed") return s;
    } catch (e) {
      if (e instanceof ApiRequestError && e.status === 0 && ++failures < 4) { /* retry */ } else throw e;
    }
    await sleep(intervalMs, signal);
  }
}

/** Kept for the existing end-to-end test. */
export const uploadVideo = uploadMedia;
