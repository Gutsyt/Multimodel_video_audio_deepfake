import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { ApiRequestError, deleteAnalysis, getAnalysis, reportUrl, waitForAnalysis, type AnalysisResult, type AnalysisStatus, type ApiErrorBody, type ModalityResult } from "../api";
import { useAppData } from "../AppData";
import { Icon } from "../components/Icon";
import { rise, stagger } from "../components/Shell";
import { useToast } from "../components/Toasts";
import { Ring, Timeline, ZoneBar } from "../components/Viz";
import { BASIS_LABEL, KIND_LABEL, STATUS_LABEL, VERDICT_LABEL, describeError, formatTime, formatWhen, pct } from "../lib/format";

const VERDICT_LINE: Record<string, string> = {
  likely_authentic: "Neither check found signs of manipulation.",
  inconclusive: "The evidence sits between the two thresholds. Treat this file with care and check its source.",
  likely_manipulated: "At least one check crossed the threshold that genuine media rarely reaches.",
  insufficient_evidence: "No face and no speech could be analysed, so the file cannot be judged.",
};

export function AnalysisPage({ id }: { id: string }) {
  const [st, setSt] = useState<AnalysisStatus | null>(null);
  const [err, setErr] = useState<ApiErrorBody | null>(null);
  useEffect(() => {
    const ctl = new AbortController();
    getAnalysis(id).then((s) => {
      setSt(s);
      if (s.status === "queued" || s.status === "processing") return waitForAnalysis(id, setSt, ctl.signal);
    }).catch((e) => { if (!(e instanceof DOMException)) setErr(e instanceof ApiRequestError ? e.body : { code: "internal_error", message: "Unexpected error." }); });
    return () => ctl.abort();
  }, [id]);

  if (err || st?.error) return <div className="card pad alert" role="alert">{describeError((err ?? st?.error)!)} <a href="#/">Analyze a file</a></div>;
  if (!st?.result) return <div className="card pad muted" role="status">{st ? "Analysis in progress…" : "Loading…"}</div>;
  return <Result id={id} r={st.result} />;
}

function Result({ id, r }: { id: string; r: AnalysisResult }) {
  const a = r.assessment;
  const { refreshHistory } = useAppData();
  const toast = useToast();
  const th = Object.values(a.thresholds);
  const strongest = a.modalities_used.length ? a.thresholds[[...a.modalities_used].sort((x, y) => (r.modalities[y].score ?? 0) - (r.modalities[x].score ?? 0))[0]] : th[0];

  async function remove() {
    try { await deleteAnalysis(id); await refreshHistory(); toast.push("Deleted."); window.location.hash = "#/history"; }
    catch (e) { toast.push(e instanceof ApiRequestError ? describeError(e.body) : "Could not delete.", "error"); }
  }

  return (
    <motion.div className="stack" style={{ display: "grid", gap: 20 }} variants={stagger} initial="hidden" animate="show">
      <motion.section variants={rise} className={`card verdict t-${a.verdict}`} aria-label="Verdict">
        <motion.div className="wash" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.2 }}
          style={{ background: "radial-gradient(120% 140% at 0% 0%, var(--cbg) 0%, transparent 55%)" }} />
        <Ring score={a.score} tReal={strongest?.t_real} tFake={strongest?.t_fake} />
        <div style={{ display: "grid", gap: 14 }}>
          <div className="row"><span className="badge">{BASIS_LABEL[a.basis]}</span><span className="faint mono" style={{ fontSize: 12 }}>{formatWhen(r.created_utc)}</span></div>
          <h1 className="word">{VERDICT_LABEL[a.verdict]}</h1>
          <p className="lead">{VERDICT_LINE[a.verdict]}</p>
          <p className="muted" style={{ fontSize: 14 }}>{a.summary}</p>
          {r.is_test_fixture && <p className="alert soft">Test doubles produced this result, not real models.</p>}
          <div className="row actions">
            <a className="btn primary" href="#/"><Icon name="analyze" size={16} /> Check another file</a>
            <a className="btn" href={reportUrl(id)}><Icon name="download" size={16} /> JSON</a>
            <button className="btn" onClick={() => window.print()}><Icon name="print" size={16} /> PDF</button>
            <button className="btn ghost danger" onClick={() => void remove()}><Icon name="trash" size={16} /></button>
          </div>
        </div>
      </motion.section>

      <div className="grid2">
        <Branch title="Face" icon="face" m={r.modalities.visual} t={a.thresholds.visual} />
        <Branch title="Voice" icon="voice" m={r.modalities.audio} t={a.thresholds.audio} />
      </div>

      <motion.section variants={rise} className="card pad">
        <h2 className="eyebrow" style={{ marginBottom: 14 }}>File</h2>
        <dl className="kv">
          <dt>Type</dt><dd>{r.media.kind ? KIND_LABEL[r.media.kind] : "–"} <span className="faint">{r.media.container?.split(",")[0]}</span></dd>
          {r.media.duration_s != null && <><dt>Duration</dt><dd>{formatTime(r.media.duration_s)}{r.media.duration_s > r.analyzed_limit_s ? ` (first ${r.analyzed_limit_s} s analysed)` : ""}</dd></>}
          {r.media.width != null && <><dt>Picture</dt><dd>{r.media.width}×{r.media.height}{r.media.fps ? ` · ${r.media.fps.toFixed(0)} fps` : ""}{r.media.video_codec ? ` · ${r.media.video_codec}` : ""}</dd></>}
          {r.media.sample_rate != null && <><dt>Sound</dt><dd>{r.media.sample_rate} Hz{r.media.audio_codec ? ` · ${r.media.audio_codec}` : ""}</dd></>}
          <dt>Took</dt><dd>{formatTime(r.total_runtime_s)}</dd>
        </dl>
      </motion.section>
    </motion.div>
  );
}

function Branch({ title, icon, m, t }: { title: string; icon: "face" | "voice"; m: ModalityResult; t: { t_real: number; t_fake: number } }) {
  const v = m.verdict ?? "insufficient_evidence";
  const frames = (m.frames ?? []).filter((f) => f.score != null);
  const pts = m.segments?.length
    ? m.segments.map((s, i) => ({ t: i, s: s.score }))
    : frames.map((f, i) => ({ t: f.t_s ?? i, s: f.score! }));
  return (
    <motion.section variants={rise} className={`card branch t-${m.status === "available" ? v : "none"}`} aria-label={`${title} check`}>
      <div className="branch-head">
        <h3><span className="ico" style={{ width: 34, height: 34, borderRadius: 10 }}><Icon name={icon} size={17} /></span>{title}</h3>
        <span className="badge">{m.status === "available" ? VERDICT_LABEL[v] : STATUS_LABEL[m.status]}</span>
      </div>
      {m.status === "available" && m.score != null ? (
        <>
          <div className="big">{pct(m.score)}<small>chance of manipulation</small></div>
          <ZoneBar score={m.score} tReal={t.t_real} tFake={t.t_fake} />
          {pts.length > 1 && <Timeline points={pts} tReal={t.t_real} tFake={t.t_fake} bars={!!m.segments?.length} label={`${title} score per ${m.segments ? "window" : "frame"}`} />}
          <dl className="kv">
            {m.n_frames_sampled != null && <><dt>Faces found</dt><dd>{m.n_face_frames} of {m.n_frames_sampled} frames</dd></>}
            {m.n_windows != null && <><dt>Speech windows</dt><dd>{m.n_windows} × 4 s</dd></>}
            <dt>Model</dt><dd>{m.model?.name}</dd>
          </dl>
        </>
      ) : (
        <p className="note">{m.reasons[0] ?? "Nothing to analyse."} This says nothing about authenticity.</p>
      )}
      {m.status === "available" && m.reasons.length > 0 && <p className="note">{m.reasons.join(" · ")}</p>}
    </motion.section>
  );
}
