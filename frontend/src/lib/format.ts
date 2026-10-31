import type { ApiErrorBody, MediaKind, ModalityStatus, Verdict } from "../api";

export const pct = (s: number | null | undefined): string => (s == null ? "–" : `${Math.round(s * 100)}%`);
export const formatTime = (s: number | null | undefined): string => (s == null ? "–" : `${s.toFixed(1)} s`);
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
export const VERDICT_LABEL: Record<Verdict, string> = {
  likely_authentic: "Likely authentic",
  inconclusive: "Inconclusive",
  likely_manipulated: "Likely manipulated",
  insufficient_evidence: "Not enough to judge",
};
export const VERDICT_SHORT: Record<Verdict, string> = {
  likely_authentic: "Authentic", inconclusive: "Inconclusive", likely_manipulated: "Manipulated", insufficient_evidence: "No evidence",
};
export const BASIS_LABEL: Record<string, string> = {
  multimodal: "Face and voice", audio_only: "Voice only", visual_only: "Face only", none: "Nothing usable",
};
export const STATUS_LABEL: Record<ModalityStatus, string> = {
  available: "Analysed", missing: "Not in this file", corrupted: "Could not be decoded",
  unreliable: "Not enough signal", not_analyzed: "Not analysed",
};

const FRIENDLY: Record<string, string> = {
  file_too_large: "That file is larger than the server allows.",
  duration_exceeded: "That media is longer than the server's duration limit.",
  unreadable_media: "The file could not be read. It may be corrupted or truncated.",
  unsupported_format: "That file type is not supported.",
  empty_upload: "The file is empty.",
  image_too_large: "That image has more pixels than the server allows.",
  unknown_duration: "The media's duration could not be determined.",
  queue_full: "The server is busy. Try again in a moment.",
  models_loading: "The models are still loading. Try again in a moment.",
  no_models_available: "No analysis model could be loaded on the server.",
  network_error: "Could not reach the analysis server. Is the backend running?",
  timeout: "The server took too long to answer.",
  interrupted: "The server restarted before this analysis finished.",
  internal_error: "The analysis failed unexpectedly on the server.",
  tool_missing: "The server is missing ffmpeg/ffprobe.",
  analysis_in_progress: "That analysis is still running.",
  not_found: "That analysis no longer exists.",
  legacy_result: "This record was made by an older model version. Run the file again.",
};
export function describeError(e: ApiErrorBody): string {
  const base = FRIENDLY[e.code] ?? e.message;
  const max = e.details?.["max_upload_mb"];
  return typeof max === "number" && e.code === "file_too_large" ? `${base} Limit: ${max} MB.` : base;
}
