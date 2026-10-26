// TEST FIXTURES for UI unit tests only: hand-written objects shaped like API responses, used to check
// rendering rules. They are NEVER imported by application code and say nothing about detection.
import type { AnalysisResult, AnalysisSummary, Assessment, Capabilities, ModalityResult } from "./api";

const card = (modality: string) => ({
  name: "AASIST", version: "v", modality, license: "MIT", source_url: "https://example.org",
  citation: "cite", trained_on: "data", limitations: ["limitation one"], is_test_fixture: false,
});

export const audioOk = (score: number): ModalityResult => ({
  status: "available", score, reasons: [], aggregation: "mean", model: card("audio"), decision: score >= 0.5 ? "flagged" : "not_flagged",
  segments: [{ start_s: 0, end_s: 4, score, logits: { spoof: 2, bona_fide: -2 } }, { start_s: 4, end_s: 8, score: 0.1 }], n_windows: 2,
  diagnostics: { output_semantics: { class_labels: ["spoof", "bona_fide"], score_class: "spoof", score_class_index: 0, reference: "ref" }, windows: { mean: score, range: 0.9 } },
});
export const visualOk = (score: number, source: "video" | "image" = "video"): ModalityResult => ({
  status: "available", score, reasons: [], source, model: card("visual"), decision: score >= 0.5 ? "flagged" : "not_flagged",
  frames: [{ frame_index: 0, t_s: source === "image" ? null : 0, n_faces: 1, score, face_box: { x: 10, y: 10, w: 50, h: 50 }, face_area_ratio: 0.2, crop_px: [80, 80] },
           { frame_index: 5, t_s: source === "image" ? null : 1, n_faces: 0, score: null }],
  n_frames_sampled: 2, n_face_frames: 1, face_detector: "haar",
  diagnostics: { output_semantics: { class_labels: ["Realism", "Deepfake"], score_class: "Deepfake", score_class_index: 1 } },
});
export const absent = (status: ModalityResult["status"], reasons: string[] = []): ModalityResult => ({ status, score: null, reasons, decision: null });

export function make(audio: ModalityResult, visual: ModalityResult, kind: "video" | "audio" | "image" = "video"): AnalysisResult {
  const usable = [["audio", audio], ["visual", visual]].filter(([, m]) => (m as ModalityResult).status === "available") as [string, ModalityResult][];
  const multi = usable.length === 2;
  const fused = multi ? (usable[0][1].score! + usable[1][1].score!) / 2 : null;
  const single = usable.length === 1 ? usable[0] : null;
  const score = multi ? fused : single ? single[1].score : null;
  const assessment: Assessment = {
    status: score == null ? "insufficient_evidence" : score >= 0.5 ? "flagged" : "not_flagged",
    basis: multi ? "multimodal" : single ? (single[0] === "audio" ? "audio_only" : "visual_only") : "none",
    summary: "Summary text.", score, score_kind: multi ? "fused_raw_score" : single ? `${single[0]}_branch_raw_score` : null,
    threshold: 0.5, threshold_validated: false, branch_decisions: { audio: audio.decision ?? null, visual: visual.decision ?? null },
    modalities_used: usable.map(([k]) => k), calibrated: false,
    not_authenticity_note: "'Not flagged' does not mean the media is authentic.",
  };
  return {
    schema_version: "0.2", avdd_version: "0.0.1", created_utc: "2026-10-25T00:00:00+00:00", outcome: "analyzed",
    media: { kind, container: "mov,mp4", duration_s: kind === "image" ? null : 8, has_video: kind !== "audio", has_audio: kind !== "image", width: 320, height: 240, fps: 25, sample_rate: 16000 },
    modalities: { audio, visual }, assessment, models: [], is_test_fixture: false,
    fusion: {
      method: "weighted mean", rule: "Applied only when both usable.", fusion_performed: multi, state: multi ? "multimodal" : single ? "single_modality" : "insufficient_evidence",
      fused_score: fused, flagged_at_default_threshold: multi ? fused! >= 0.5 : null, threshold: 0.5,
      weights_used: multi ? { audio: 0.5, visual: 0.5 } : {}, branch_flags_at_default_threshold: {}, excluded_modalities: {},
      weights_validated: false, threshold_validated: false, calibrated: false, notes: [],
    },
    interpretation: { score_meaning: "Raw model output in [0, 1].", calibrated: false, validated_on_project_data: false, flag_meaning: "Flagged means only a threshold crossing.", not_forensic_proof: "Not proof." },
    analyzed_limit_s: 60, total_runtime_s: 1.5,
  };
}

export const caps = (over: Partial<Capabilities> = {}): Capabilities => ({
  avdd_version: "0.0.1", models_state: "ready", models: [], retention: "r", evaluation_status: "e", stages: ["probing", "audio", "visual", "fusion"],
  limits: { max_upload_mb: 1, max_duration_s: 60, accepted_containers: [], accepted: { video: { label: "Video", extensions: [".mp4"] }, audio: { label: "Audio", extensions: [".wav"] }, image: { label: "Image", extensions: [".png"] } }, frames_sampled: 16, max_pending_jobs: 4 },
  ...over,
});

export const summary = (id: string, over: Partial<AnalysisSummary> = {}): AnalysisSummary => ({
  id, status: "completed", created_utc: "2026-10-25T00:00:00+00:00", duration_s: 5, media_kind: "video", assessment_status: "flagged", basis: "multimodal",
  audio_decision: "flagged", visual_decision: "flagged", fusion_state: "multimodal", fused_score: 0.7, flagged: true, audio_status: "available", audio_score: 0.8,
  visual_status: "available", visual_score: 0.6, is_test_fixture: false, error_code: null, ...over,
});
