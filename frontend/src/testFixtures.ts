// TEST FIXTURES for UI unit tests only. These are hand-written objects shaped like API responses,
// used to check rendering rules. They are NEVER imported by application code (a build check
// verifies the production bundle does not contain this file) and say nothing about detection.
import type { AnalysisResult, Fusion, ModalityResult } from "./api";

const card = (modality: string, fixture = false) => ({
  name: fixture ? "stub" : "AASIST", version: "v", modality, license: "MIT", source_url: "https://example.org",
  citation: "cite", trained_on: "data", limitations: ["limitation one"], is_test_fixture: fixture,
});

export const available = (score: number, fixture = false): ModalityResult => ({
  status: "available", score, reasons: [], aggregation: "mean of per-window raw scores", model: card("audio", fixture),
  segments: [{ start_s: 0, end_s: 4, score }], frames: null,
});
export const absent = (status: ModalityResult["status"], reasons: string[] = []): ModalityResult => ({ status, score: null, reasons });

export const fusion = (over: Partial<Fusion>): Fusion => ({
  method: "weighted mean", state: "multimodal", fused_score: 0.4, flagged_at_default_threshold: false, threshold: 0.5,
  weights_used: { audio: 0.5, visual: 0.5 }, branch_flags_at_default_threshold: {}, excluded_modalities: {},
  weights_validated: false, threshold_validated: false, calibrated: false, notes: ["Fusion weights are an unvalidated equal-weight default."], ...over,
});

export const result = (audio: ModalityResult, visual: ModalityResult, f: Fusion, fixture = false): AnalysisResult => ({
  schema_version: "0.1", avdd_version: "0.0.1", created_utc: "2026-10-25T00:00:00+00:00", outcome: "analyzed",
  media: { container: "mov,mp4", duration_s: 4, has_video: true, has_audio: true, width: 320, height: 240, fps: 25, sample_rate: 16000 },
  modalities: { audio, visual }, fusion: f, models: [], is_test_fixture: fixture,
  interpretation: { score_meaning: "Raw model output in [0, 1].", calibrated: false, validated_on_project_data: false, flag_meaning: "Flagged means only a threshold crossing.", not_forensic_proof: "Not proof." },
  analyzed_limit_s: 60, total_runtime_s: 1.5,
});
