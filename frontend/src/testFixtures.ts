// TEST FIXTURES for UI unit tests only: hand-written objects shaped like API responses.
import type { AnalysisResult, AnalysisSummary, Capabilities, ModalityResult, Verdict } from "./api";

const TH = { t_real: 0.3, t_fake: 0.7 };
const v = (s: number): Verdict => (s >= TH.t_fake ? "likely_manipulated" : s <= TH.t_real ? "likely_authentic" : "inconclusive");
const card = (modality: string) => ({
  name: `${modality} model`, version: "v", modality, license: "MIT", source_url: "https://example.org",
  citation: "cite", trained_on: "data", limitations: ["limitation one"], is_test_fixture: false,
  metrics: { layer: 8, t_real: 0.3, t_fake: 0.7, validation: { n_real: 10, n_fake: 10, auc: 0.95, eer: 0.1 },
    tests: { in_domain_test: { n_real: 50, n_fake: 50, auc: 0.93, eer: 0.12, genuine_flagged: 0.02, genuine_cleared: 0.9, fake_flagged: 0.6, fake_cleared: 0.1 } } },
});

export const audioOk = (score: number): ModalityResult => ({
  status: "available", score, reasons: [], model: card("audio"), verdict: v(score),
  segments: [{ start_s: 0, end_s: 4, score }, { start_s: 4, end_s: 8, score: 0.1 }], n_windows: 2,
});
export const visualOk = (score: number): ModalityResult => ({
  status: "available", score, reasons: [], source: "video", model: card("visual"), verdict: v(score),
  frames: [{ frame_index: 0, t_s: 0, n_faces: 1, score }, { frame_index: 5, t_s: 1, n_faces: 1, score }],
  n_frames_sampled: 2, n_face_frames: 2,
});
export const absent = (status: ModalityResult["status"], reasons: string[] = []): ModalityResult => ({ status, score: null, reasons, verdict: null });

export function make(audio: ModalityResult, visual: ModalityResult): AnalysisResult {
  const used = ([["audio", audio], ["visual", visual]] as const).filter(([, m]) => m.status === "available");
  const vs = used.map(([, m]) => m.verdict!);
  const verdict: Verdict = !vs.length ? "insufficient_evidence" : vs.includes("likely_manipulated") ? "likely_manipulated"
    : vs.every((x) => x === "likely_authentic") ? "likely_authentic" : "inconclusive";
  return {
    schema_version: "1.0", avdd_version: "0.0.1", created_utc: "2026-10-25T00:00:00+00:00", outcome: "analyzed",
    media: { kind: "video", container: "mov,mp4", duration_s: 8, has_video: true, has_audio: true, width: 320, height: 240, fps: 25, sample_rate: 16000 },
    modalities: { audio, visual },
    assessment: {
      verdict, basis: used.length === 2 ? "multimodal" : used.length ? (used[0][0] === "audio" ? "audio_only" : "visual_only") : "none",
      summary: "Summary text.", score: used.length ? Math.max(...used.map(([, m]) => m.score!)) : null,
      branch_verdicts: { audio: audio.verdict ?? null, visual: visual.verdict ?? null }, thresholds: { audio: TH, visual: TH },
      modalities_used: used.map(([k]) => k),
    },
    models: [], is_test_fixture: false, analyzed_limit_s: 60, total_runtime_s: 1.5,
  };
}

export const caps = (over: Partial<Capabilities> = {}): Capabilities => ({
  avdd_version: "0.0.1", models_state: "ready", retention: "r", evaluation_status: "e", stages: ["probing", "audio", "visual", "verdict"],
  models: [{ role: "visual", state: "loaded", error_code: null, card: card("visual") }],
  limits: { max_upload_mb: 1, max_duration_s: 60, accepted_containers: [], accepted: { video: { label: "Video", extensions: [".mp4"] }, audio: { label: "Audio", extensions: [".wav"] }, image: { label: "Image", extensions: [".png"] } }, frames_sampled: 12, max_pending_jobs: 4 },
  ...over,
});

export const summary = (id: string, over: Partial<AnalysisSummary> = {}): AnalysisSummary => ({
  id, status: "completed", created_utc: "2026-10-25T00:00:00+00:00", duration_s: 5, media_kind: "video", verdict: "likely_authentic", basis: "multimodal",
  score: 0.1, audio_status: "available", audio_score: 0.1, visual_status: "available", visual_score: 0.05, is_test_fixture: false, error_code: null, ...over,
});
