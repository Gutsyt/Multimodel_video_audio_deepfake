// Presentation logic. Rule: an absent / unusable modality is never described as authentic or real.
import type { ApiErrorBody, Fusion, ModalityStatus } from "./api";

export const STATUS_LABEL: Record<ModalityStatus, string> = {
  available: "Analyzed",
  missing: "Not present in file",
  corrupted: "Could not be decoded",
  unreliable: "Too unreliable to score",
  not_analyzed: "Not analyzed",
};

export const STATUS_EXPLANATION: Record<ModalityStatus, (what: string) => string> = {
  available: (w) => `The ${w} was analyzed by a pretrained model.`,
  missing: (w) => `This file has no ${w}. Nothing was analyzed, and a missing ${w} is not evidence that the media is genuine or manipulated.`,
  corrupted: (w) => `The ${w} could not be decoded, so nothing was analyzed. This is not evidence either way.`,
  unreliable: (w) => `The ${w} failed quality checks, so no score is shown. This is not evidence either way.`,
  not_analyzed: (w) => `The ${w} was not analyzed because the model was unavailable or failed. This is not evidence either way.`,
};

export type Tone = "flagged" | "not_flagged" | "insufficient";
export interface Headline { tone: Tone; title: string; detail: string }

export function fusionHeadline(f: Fusion): Headline {
  if (f.state === "insufficient_evidence" || f.fused_score === null) {
    return {
      tone: "insufficient",
      title: "Insufficient evidence",
      detail: "No audio or visual evidence could be used, so no result is given. This is not evidence of authenticity.",
    };
  }
  const basis = f.state === "single_modality" ? "one modality only" : "audio and visual evidence";
  if (f.flagged_at_default_threshold) {
    return {
      tone: "flagged",
      title: "Flagged by the model at the default threshold",
      detail: `Based on ${basis}. The threshold is an unvalidated default; genuine media can be flagged.`,
    };
  }
  return {
    tone: "not_flagged",
    title: "Not flagged at the default threshold",
    detail: `Based on ${basis}. This is not evidence that the media is authentic; manipulated media can evade detection.`,
  };
}

export const formatScore = (s: number): string => s.toFixed(2);
export const formatTime = (s: number): string => `${s.toFixed(1)} s`;
export const formatBytes = (n: number): string =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} kB`;
export const modalityNoun = (m: string): string => (m === "audio" ? "audio track" : "video stream");

const FRIENDLY: Record<string, string> = {
  file_too_large: "That file is larger than the allowed upload size.",
  duration_exceeded: "That video is longer than the allowed duration.",
  unreadable_media: "That file could not be read as a video. It may be corrupted or not a video.",
  unsupported_format: "That file type is not supported. Use MP4, MOV, MKV, WebM or AVI.",
  empty_upload: "The selected file is empty.",
  queue_full: "The server is busy with other analyses. Please try again in a moment.",
  models_loading: "The analysis models are still loading. Please try again shortly.",
  no_models_available: "No analysis model could be loaded on the server, so analysis is unavailable.",
  network_error: "Could not reach the analysis server. Is the backend running?",
  interrupted: "The server restarted before this analysis finished.",
  internal_error: "The analysis failed unexpectedly. Nothing was concluded about this file.",
};

export function describeError(e: ApiErrorBody): { title: string; detail: string } {
  return { title: FRIENDLY[e.code] ?? e.message, detail: `Error code: ${e.code}` };
}
