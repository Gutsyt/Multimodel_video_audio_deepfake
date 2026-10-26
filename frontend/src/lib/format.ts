import type { ApiErrorBody, Assessment, AssessmentStatus, AnalysisResult, MediaKind, ModalityStatus } from "../api";

export const formatScore = (s: number | null | undefined): string => (s == null ? "n/a" : s.toFixed(2));
export const formatTime = (s: number | null | undefined): string => (s == null ? "n/a" : `${s.toFixed(1)} s`);
export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
export function formatWhen(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export const KIND_LABEL: Record<MediaKind, string> = { video: "Video", audio: "Audio", image: "Image" };

export const ASSESSMENT_LABEL: Record<AssessmentStatus, string> = {
  flagged: "Flagged by the model",
  not_flagged: "Not flagged by the model",
  insufficient_evidence: "Insufficient evidence",
};
export const BASIS_LABEL: Record<string, string> = {
  multimodal: "Audio and visual branches, fused",
  audio_only: "Audio branch only",
  visual_only: "Visual branch only",
  none: "No usable evidence",
};
export const STATUS_LABEL: Record<ModalityStatus, string> = {
  available: "Analyzed",
  missing: "Not in this file",
  corrupted: "Could not be decoded",
  unreliable: "Too unreliable to score",
  not_analyzed: "Not analyzed",
};
export const STATUS_EXPLANATION: Record<Exclude<ModalityStatus, "available">, string> = {
  missing: "Nothing was analyzed for this branch. A missing stream is not evidence about authenticity.",
  corrupted: "The stream exists but could not be decoded. Nothing was scored.",
  unreliable: "Quality checks say a score would not be trustworthy, so none was produced or used.",
  not_analyzed: "No analysis ran for this branch (model unavailable or inference failed). This is a system limit, not a property of the media.",
};

/** Results stored before schema 0.2 have no assessment block; derive one without inventing anything. */
export function assessmentOf(r: AnalysisResult): Assessment {
  if (r.assessment) return r.assessment;
  const f = r.fusion;
  const used = Object.entries(r.modalities).filter(([, m]) => m.status === "available").map(([k]) => k);
  const single = used.length === 1;
  const status: AssessmentStatus = used.length === 0 ? "insufficient_evidence" : f.flagged_at_default_threshold ? "flagged" : "not_flagged";
  const m = single ? r.modalities[used[0]] : null;
  return {
    status, basis: used.length === 0 ? "none" : single ? (used[0] === "audio" ? "audio_only" : "visual_only") : "multimodal",
    summary: "", score: single ? m!.score : f.fused_score, score_kind: single ? `${used[0]}_branch_raw_score` : f.fused_score == null ? null : "fused_raw_score",
    threshold: f.threshold, threshold_validated: f.threshold_validated,
    branch_decisions: Object.fromEntries(Object.entries(r.modalities).map(([k, v]) => [k, v.status === "available" && v.score != null ? (v.score >= f.threshold ? "flagged" : "not_flagged") : null])),
    modalities_used: used, calibrated: false,
    not_authenticity_note: "'Not flagged' does not mean the media is authentic.",
  };
}

const FRIENDLY: Record<string, string> = {
  file_too_large: "That file is larger than the server allows.",
  duration_exceeded: "That media is longer than the server's duration limit.",
  unreadable_media: "The file could not be read. It may be corrupted or truncated.",
  unsupported_format: "That file type is not supported. Use a video, audio file or image from the accepted list.",
  empty_upload: "The file is empty.",
  image_too_large: "That image has more pixels than the server allows.",
  unknown_duration: "The media's duration could not be determined, so it cannot be analyzed.",
  queue_full: "The server is busy with other analyses. Try again in a moment.",
  models_loading: "The models are still loading. Try again in a moment.",
  no_models_available: "No analysis model could be loaded on the server.",
  network_error: "Could not reach the analysis server. Check that the backend is running and the API address is right.",
  timeout: "The server took too long to answer.",
  upload_interrupted: "The upload was interrupted.",
  interrupted: "The server restarted before this analysis finished.",
  internal_error: "The analysis failed unexpectedly on the server.",
  tool_missing: "The server is missing ffmpeg/ffprobe, which it needs to read media.",
  analysis_in_progress: "That analysis is still running; delete it when it finishes.",
  not_found: "That analysis no longer exists.",
};
export function describeError(e: ApiErrorBody): string {
  const base = FRIENDLY[e.code] ?? e.message;
  const max = e.details?.["max_upload_mb"];
  return typeof max === "number" && e.code === "file_too_large" ? `${base} Limit: ${max} MB.` : base;
}
