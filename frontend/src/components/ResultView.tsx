import type { AnalysisResult, ModalityResult } from "../api";
import { AudioTrace, FaceMap, FrameTrace } from "./Traces";
import { FusionDiagram, PipelineFlow } from "./Diagrams";
import { Icon } from "./Icon";
import { Meter } from "./Meter";
import { StatusMark } from "./StatusMark";
import { ASSESSMENT_LABEL, BASIS_LABEL, STATUS_EXPLANATION, STATUS_LABEL, assessmentOf, formatScore, formatTime } from "../lib/format";

const label = (k: string) => k.replace(/_/g, " ");
function KV({ data }: { data: Record<string, string | number | boolean | null> | undefined }) {
  if (!data) return null;
  return <dl className="facts">{Object.entries(data).map(([k, v]) => <div style={{ display: "contents" }} key={k}><dt>{label(k)}</dt><dd className="mono">{String(v)}</dd></div>)}</dl>;
}

function Verdict({ r }: { r: AnalysisResult }) {
  const a = assessmentOf(r);
  const glyph = a.status === "flagged" ? <path d="M16 56V10M16 10h32l-6 12 6 12H16" stroke="var(--flag)" /> : a.status === "not_flagged" ? <><circle cx="32" cy="32" r="22" stroke="var(--steel)" /><path d="M22 32h20" stroke="var(--steel)" /></> : <><circle cx="32" cy="32" r="22" stroke="var(--ink-3)" strokeDasharray="4 5" /><path d="M26 26a6 6 0 1 1 8 5c-2 1-2 3-2 4M32 44h.01" stroke="var(--ink-3)" /></>;
  return (
    <section className={`verdict ${a.status}`} aria-labelledby="verdict-h">
      <svg width="72" height="72" viewBox="0 0 64 64" fill="none" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{glyph}</svg>
      <div style={{ display: "grid", gap: 12 }}>
        <div>
          <h1 id="verdict-h">{ASSESSMENT_LABEL[a.status]}</h1>
          <div className="row"><span className="status neutral">{BASIS_LABEL[a.basis]}</span>{a.basis !== "none" && a.basis !== "multimodal" && <span className="muted">single-branch result, not a multimodal one</span>}</div>
        </div>
        {a.summary && <p>{a.summary}</p>}
        {a.score != null && <div style={{ maxWidth: 420 }}><Meter score={a.score} threshold={a.threshold} label={a.basis === "multimodal" ? "Fused raw score" : a.basis === "audio_only" ? "Audio branch raw score" : "Visual branch raw score"} /></div>}
        <p className="note">{a.not_authenticity_note} {r.interpretation.not_forensic_proof}</p>
      </div>
    </section>
  );
}

function BranchCard({ title, name, m, r }: { title: string; name: "audio" | "visual"; m: ModalityResult | undefined; r: AnalysisResult }) {
  const thr = r.fusion.threshold;
  if (!m) return null;
  const d = m.diagnostics;
  const sem = d?.output_semantics;
  const ok = m.status === "available" && m.score != null;
  const stats = name === "audio" ? d?.windows : d?.scores;
  return (
    <section className="panel" aria-label={`${title} branch`}>
      <header><h2>{title}</h2><span className={`status ${ok ? "ok" : "neutral"}`}>{STATUS_LABEL[m.status]}</span></header>
      <div style={{ display: "grid", gap: 16 }}>
        {ok ? <Meter score={m.score} threshold={thr} label="Branch raw score" /> : <Meter score={null} threshold={thr} label="Branch raw score" emptyText="not produced" />}
        {ok && m.decision && <div className="row"><StatusMark tone={m.decision === "flagged" ? "flagged" : "not_flagged"} label={`This branch alone: ${m.decision === "flagged" ? "flagged" : "not flagged"} at the default threshold`} /></div>}
        {!ok && m.status !== "missing" && m.status !== "corrupted" && m.status !== "not_analyzed" ? null : null}
        {!ok && <p className="muted">{STATUS_EXPLANATION[m.status as Exclude<typeof m.status, "available">]}</p>}
        {m.reasons.length > 0 && <ul className="lim">{m.reasons.map((x) => <li key={x}>{x}</li>)}</ul>}

        {name === "audio" && ok && m.segments && <><AudioTrace segments={m.segments} duration={r.media.duration_s} threshold={thr} /><div className="legend"><span><i className="k-sig" />window below threshold</span><span><i className="k-flag" />at or above</span><span><i style={{ borderTop: "2px dashed var(--flag)", width: 14, height: 0, borderRadius: 0 }} />default threshold {thr} (unvalidated)</span><span className="faint">{m.n_windows} window(s) of about 4 s, shown where they sit in time</span></div></>}
        {name === "visual" && ok && m.frames && <>
          <FrameTrace frames={m.frames} duration={m.source === "image" ? null : r.media.duration_s} threshold={thr} />
          <FaceMap frames={m.frames} width={r.media.width ?? 0} height={r.media.height ?? 0} threshold={thr} />
          <div className="legend"><span><i style={{ borderTop: "2px dashed var(--flag)", width: 14, height: 0, borderRadius: 0 }} />default threshold {thr} (unvalidated)</span><span>{m.n_face_frames} of {m.n_frames_sampled} {m.source === "image" ? "image" : "sampled frames"} had a face</span><span>detector: {m.face_detector}</span><span className="faint">only the largest face per frame is scored</span></div></>}
        {name === "visual" && !ok && m.frames && m.frames.length > 0 && <FrameTrace frames={m.frames} duration={r.media.duration_s} threshold={thr} />}

        {m.metrics && <details><summary>Audio input quality</summary><KV data={m.metrics} /></details>}
        {sem && <details open={ok}><summary>What this score means</summary>
          <p className="muted">The score is the softmax mass on the <b>{sem.score_class}</b> class (output {sem.score_class_index} of {sem.class_labels.join(" / ")}). {sem.reference}</p>
          <p className="muted" style={{ marginTop: 8 }}>It is a raw model output, not a calibrated probability, and the threshold is an unvalidated default.</p></details>}
        {stats && <details><summary>Score statistics across {name === "audio" ? "windows" : "frames"}</summary><KV data={stats as Record<string, string | number>} />
          <p className="faint" style={{ marginTop: 6 }}>The reported branch score is the mean. When the values spread widely, the mean hides it.</p></details>}
        {d?.input && <details><summary>Preprocessing</summary><KV data={d.input} /></details>}
        {name === "audio" && m.segments && <details><summary>Per-window values</summary><div className="tablewrap"><table><thead><tr><th>Window</th><th>Raw score</th><th>Logits</th></tr></thead><tbody>
          {m.segments.map((s, i) => <tr key={i}><td>{formatTime(s.start_s)} to {formatTime(s.end_s)}</td><td className="num">{formatScore(s.score)}</td><td className="num">{s.logits ? Object.entries(s.logits).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ") : "n/a"}</td></tr>)}</tbody></table></div></details>}
        {name === "visual" && m.frames && m.frames.some((f) => f.score != null) && <details><summary>Per-frame values</summary><div className="tablewrap"><table><thead><tr><th>Frame</th><th>Raw score</th><th>Face area</th><th>Crop px</th><th>Logits</th></tr></thead><tbody>
          {m.frames.filter((f) => f.score != null).map((f, i) => <tr key={i}><td>{f.t_s != null ? formatTime(f.t_s) : "image"}</td><td className="num">{formatScore(f.score)}</td><td className="num">{f.face_area_ratio != null ? `${(f.face_area_ratio * 100).toFixed(1)}%` : "n/a"}</td><td className="num">{f.crop_px?.join("×") ?? "n/a"}</td><td className="num">{f.logits ? Object.entries(f.logits).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ") : "n/a"}</td></tr>)}</tbody></table></div></details>}
        {m.model && <details><summary>Model and limitations</summary>
          <p><b>{m.model.name}</b> <span className="faint mono">{m.model.version}</span></p>
          <p className="muted">Trained on: {m.model.trained_on}. License: {m.model.license}. <a href={m.model.source_url} target="_blank" rel="noreferrer">Source</a></p>
          <ul className="lim">{m.model.limitations.map((l) => <li key={l}>{l}</li>)}</ul></details>}
      </div>
    </section>
  );
}

function FusionPanel({ r }: { r: AnalysisResult }) {
  const f = r.fusion;
  return (
    <section className="panel" aria-label="Fusion">
      <header><h2>Fusion</h2><span className={`status ${f.state === "multimodal" ? "ok" : "neutral"}`}>{f.state === "multimodal" ? "performed" : "not performed"}</span></header>
      <div className="grid2" style={{ alignItems: "center" }}>
        <FusionDiagram result={r} />
        <div style={{ display: "grid", gap: 12 }}>
          {f.state === "multimodal" ? <Meter score={f.fused_score} threshold={f.threshold} label="Fused raw score" /> : (
            <p>{f.state === "single_modality" ? "Only one branch produced usable evidence, so nothing was combined and there is no fused score. The assessment above is that branch's own result." : "No branch produced usable evidence, so there is nothing to combine."}</p>)}
          <dl className="facts">
            <dt>Rule</dt><dd>{f.rule ?? f.method}</dd>
            {f.state === "multimodal" && <><dt>Weights used</dt><dd className="mono">{Object.entries(f.weights_used).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ")}</dd></>}
            <dt>Threshold</dt><dd className="mono">{f.threshold} <span className="status neutral" style={{ marginLeft: 6 }}>unvalidated default</span></dd>
            <dt>Validation</dt><dd>Weights {f.weights_validated ? "validated" : "not validated"}; threshold {f.threshold_validated ? "validated" : "not validated"}; scores {f.calibrated ? "calibrated" : "not calibrated"}</dd>
          </dl>
          {f.notes.length > 0 && <ul className="lim">{f.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
        </div>
      </div>
    </section>
  );
}

export function ResultView({ result: r }: { result: AnalysisResult }) {
  const m = r.media;
  return (
    <div className="stack">
      {r.is_test_fixture && <div className="errbox" role="status"><b>Test doubles in use</b><span>These numbers come from stand-in detectors that return fixed values. They say nothing about any media.</span></div>}
      <Verdict r={r} />
      <section className="panel" aria-label="Pipeline"><header><h2>What ran</h2></header><PipelineFlow result={r} /></section>
      <div className="grid2">
        <BranchCard title="Audio" name="audio" m={r.modalities.audio} r={r} />
        <BranchCard title="Visual" name="visual" m={r.modalities.visual} r={r} />
      </div>
      <FusionPanel r={r} />
      <section className="panel" aria-label="Evidence and metadata">
        <header><h2>Processing metadata</h2></header>
        <dl className="facts">
          <dt>Media kind</dt><dd>{m.kind ?? (m.has_video ? "video" : "audio")}</dd>
          <dt>Container</dt><dd className="mono">{m.container ?? "n/a"}</dd>
          <dt>Duration</dt><dd>{m.duration_s != null ? formatTime(m.duration_s) : "n/a (still image)"}</dd>
          {m.width && <><dt>Frame size</dt><dd className="mono">{m.width}×{m.height}{m.fps ? ` @ ${m.fps.toFixed(1)} fps` : ""}{m.video_codec ? ` · ${m.video_codec}` : ""}</dd></>}
          {m.has_audio && <><dt>Audio stream</dt><dd className="mono">{m.audio_codec ?? "?"} · {m.sample_rate ?? "?"} Hz · {m.channels ?? "?"} ch</dd></>}
          <dt>Analysis limit</dt><dd>first {r.analyzed_limit_s} s</dd>
          <dt>Runtime</dt><dd>{r.total_runtime_s.toFixed(1)} s total</dd>
          <dt>Analyzed</dt><dd>{new Date(r.created_utc).toLocaleString()} · schema {r.schema_version} · avdd {r.avdd_version}</dd>
        </dl>
        <hr className="rule" />
        <p className="muted">{r.interpretation.score_meaning}</p>
        <p className="muted" style={{ marginTop: 8 }}>{r.interpretation.flag_meaning}</p>
      </section>
    </div>
  );
}
export { Icon };
