import { motion, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { ApiRequestError, deleteAnalysis, getAnalysis, reportUrl, waitForAnalysis, type AnalysisResult, type AnalysisStatus, type ApiErrorBody, type ModalityResult } from "../api";
import { useAppData } from "../AppData";
import { EvidenceScale, Lanes, MediaStage, usePeaks, zoneOf, type Th } from "../components/Evidence";
import { Compare, Filmstrip, shareCard } from "../components/Extras";
import { Icon } from "../components/Icon";
import { useToast } from "../components/Toasts";
import { BASIS_LABEL, KIND_LABEL, STATUS_LABEL, VERDICT_LABEL, describeError, formatTime, formatWhen } from "../lib/format";
import { sessionMedia } from "../lib/session";

const LINE: Record<string, string> = {
  likely_authentic: "Both checks sit in the range genuine media normally falls in.",
  inconclusive: "At least one check landed between the thresholds. Check where this file came from before trusting it.",
  likely_manipulated: "A check crossed the threshold that at most 2% of genuine media reached in validation.",
  insufficient_evidence: "No face and no speech could be analysed, so this file cannot be judged either way.",
};
const STAGE: Record<string, string> = { probing: "Reading the file", audio: "Checking the voice", visual: "Checking the face", verdict: "Weighing the evidence" };

export function AnalysisPage({ id }: { id: string }) {
  const [st, setSt] = useState<AnalysisStatus | null>(null);
  const [err, setErr] = useState<ApiErrorBody | null>(null);
  useEffect(() => {
    const ctl = new AbortController();
    getAnalysis(id).then((s) => {
      setSt(s);
      if (s.status === "queued" || s.status === "processing") return waitForAnalysis(id, setSt, ctl.signal, 500);
    }).catch((e) => { if (!(e instanceof DOMException)) setErr(e instanceof ApiRequestError ? e.body : { code: "internal_error", message: "Unexpected error." }); });
    return () => ctl.abort();
  }, [id]);

  const e = err ?? st?.error;
  if (e) return <div className="alert" role="alert"><b>{describeError(e)}</b><a href="#/">Check a file</a></div>;
  if (!st?.result) return <Pending st={st} />;
  return <Result id={id} r={st.result} />;
}

function Pending({ st }: { st: AnalysisStatus | null }) {
  const stages = st?.stages.length ? st.stages : ["probing", "audio", "visual", "verdict"];
  const ci = st?.stage ? stages.indexOf(st.stage) : -1;
  return (
    <div className="pending" aria-live="polite">
      <h1 className="verdict-word muted">{st ? "Analysing…" : "Loading…"}</h1>
      <ol className="steps">
        {stages.map((k, i) => (
          <li key={k} className={i < ci ? "done" : i === ci ? "now" : ""}><span className="tick">{i < ci && <Icon name="check" size={11} />}</span><span>{STAGE[k] ?? k}</span><span /></li>
        ))}
      </ol>
      <div className="skeleton" />
    </div>
  );
}

function Result({ id, r }: { id: string; r: AnalysisResult }) {
  const a = r.assessment, vis = r.modalities.visual, aud = r.modalities.audio;
  const tv: Th = a.thresholds.visual, ta: Th = a.thresholds.audio;
  const { refreshHistory } = useAppData();
  const toast = useToast();
  const reduce = useReducedMotion();
  const [settled, setSettled] = useState(false);
  const onSettled = useCallback(() => setSettled(true), []);
  const media = sessionMedia(id);
  const kind = r.media.kind ?? "video";
  const dur = Math.min(r.media.duration_s ?? 0, r.analyzed_limit_s);
  const peaks = usePeaks(kind !== "image" && media ? media.file : null, r.analyzed_limit_s);
  const video = useRef<HTMLVideoElement>(null);
  const [playhead, setPlayhead] = useState<number | null>(null);
  const onTime = useCallback((t: number) => setPlayhead(t), []);
  const seek = (t: number) => { setPlayhead(t); if (video.current) video.current.currentTime = t; };
  const inp = (vis.diagnostics?.input ?? {}) as Record<string, number>;
  const frameSize: [number, number] | null = inp.frame_width ? [inp.frame_width, inp.frame_height] : r.media.width ? [r.media.width, r.media.height!] : null;

  async function remove() {
    try { await deleteAnalysis(id); await refreshHistory(); toast.push("Deleted."); window.location.hash = "#/history"; }
    catch (e) { toast.push(e instanceof ApiRequestError ? describeError(e.body) : "Could not delete.", "error"); }
  }
  async function copy() {
    try { await navigator.clipboard.writeText(`${VERDICT_LABEL[a.verdict]} (${BASIS_LABEL[a.basis]}). ${a.summary}`); toast.push("Summary copied.", "success"); }
    catch { toast.push("Clipboard is not available here.", "error"); }
  }

  async function share() {
    try { toast.push((await shareCard(r)) === "shared" ? "Shared." : "Card saved as PNG.", "success"); }
    catch (e) { if (!(e instanceof DOMException && e.name === "AbortError")) toast.push("Could not make the card.", "error"); }
  }

  return (
    <div className={`result t-${a.verdict}`}>
      <header className="result-head">
        <div className="result-title">
          <motion.h1 className="verdict-word" initial={reduce ? false : { filter: "blur(10px)", opacity: 0.25 }}
            animate={settled || reduce ? { filter: "blur(0px)", opacity: 1 } : { filter: "blur(10px)", opacity: 0.25 }} transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}>
            {VERDICT_LABEL[a.verdict]}
          </motion.h1>
          <p className="result-line">{LINE[a.verdict]}</p>
          <p className="meta">{BASIS_LABEL[a.basis]} · {r.media.kind ? KIND_LABEL[r.media.kind] : "File"}{r.media.duration_s ? ` · ${formatTime(r.media.duration_s)}` : ""} · {formatWhen(r.created_utc)}</p>
        </div>
        <div className="row actions">
          <a className="btn primary" href="#/"><Icon name="analyze" size={16} /> Check another</a>
          <button className="btn" onClick={() => void share()}>Share card</button>
          <button className="btn" onClick={() => void copy()}>Copy summary</button>
          <a className="btn" href={reportUrl(id)}><Icon name="download" size={16} /> JSON</a>
          <button className="btn" onClick={() => window.print()}><Icon name="print" size={16} /> PDF</button>
          <button className="btn ghost danger" aria-label="Delete this analysis" onClick={() => void remove()}><Icon name="trash" size={16} /></button>
        </div>
      </header>
      {r.is_test_fixture && <p className="alert soft">Test doubles produced this result, not the real models.</p>}

      <section className="panel" aria-label="Evidence scale">
        <EvidenceScale onSettled={onSettled} rows={[
          { key: "visual", label: "Face", score: vis.score, t: tv, note: STATUS_LABEL[vis.status] },
          { key: "audio", label: "Voice", score: aud.score, t: ta, note: STATUS_LABEL[aud.status] },
        ]} />
        <p className="legend"><i className="lg real" />genuine range <i className="lg mid" />borderline <i className="lg fake" />manipulated range <span className="faint">· bands differ per check; both were set on held-out validation data</span></p>
      </section>

      {(kind !== "audio" || media) && (
        <section className={`panel explore${media && kind !== "audio" ? " with-stage" : ""}`} aria-label="Evidence over time">
          {media && kind !== "audio" && <MediaStage url={media.url} kind={kind === "image" ? "image" : "video"} frames={vis.frames ?? []} tv={tv} frameSize={frameSize} videoRef={video} onTime={onTime} />}
          {kind === "audio" && media && <audio className="audio-player" src={media.url} controls onTimeUpdate={(e) => setPlayhead(e.currentTarget.currentTime)} ref={(el) => { (video as React.MutableRefObject<unknown>).current = el; }} />}
          {kind !== "image" && dur > 0 && <Lanes dur={dur} frames={vis.frames ?? []} segments={aud.segments ?? []} tv={tv} ta={ta} peaks={peaks} playhead={playhead} onSeek={seek} />}
          {media && kind !== "audio" && <div className="span-all"><Filmstrip url={media.url} kind={kind === "image" ? "image" : "video"} frames={vis.frames ?? []} tv={tv} onSeek={seek} /></div>}
          {!media && <p className="hint">Analyse a file in this tab to replay it here with the detected face outlined frame by frame. The server deleted its copy after scoring.</p>}
        </section>
      )}
      {kind === "audio" && !media && dur > 0 && (
        <section className="panel" aria-label="Evidence over time"><Lanes dur={dur} frames={[]} segments={aud.segments ?? []} tv={tv} ta={ta} peaks={null} playhead={null} onSeek={() => undefined} /></section>
      )}

      <div className="checks">
        <Check name="Face" m={vis} t={tv} unit="frames" />
        <Check name="Voice" m={aud} t={ta} unit="windows" />
      </div>

      <Compare id={id} r={r} />

      <dl className="file-facts">
        <div><dt>Container</dt><dd>{r.media.container?.split(",")[0] ?? "–"}</dd></div>
        {r.media.width != null && <div><dt>Picture</dt><dd>{r.media.width}×{r.media.height}{r.media.fps ? ` @ ${r.media.fps.toFixed(0)} fps` : ""}</dd></div>}
        {r.media.video_codec && <div><dt>Video codec</dt><dd>{r.media.video_codec}</dd></div>}
        {r.media.sample_rate != null && <div><dt>Audio</dt><dd>{r.media.sample_rate} Hz {r.media.audio_codec ?? ""}</dd></div>}
        <div><dt>Analysis time</dt><dd>{formatTime(r.total_runtime_s)}</dd></div>
      </dl>
    </div>
  );
}

function Check({ name, m, t, unit }: { name: string; m: ModalityResult; t: Th; unit: string }) {
  const s = ((m.diagnostics?.[unit === "frames" ? "scores" : "windows"]) ?? {}) as Record<string, number>;
  const z = m.score == null ? null : zoneOf(m.score, t);
  const margin = m.score == null ? null : z === "fake" ? m.score - t.t_fake : z === "real" ? t.t_real - m.score : Math.min(m.score - t.t_real, t.t_fake - m.score);
  return (
    <section className="check" aria-label={`${name} check`}>
      <h2>{name} <span className={`zc ${z ?? "none"}`}>{m.verdict ? VERDICT_LABEL[m.verdict] : STATUS_LABEL[m.status]}</span></h2>
      {m.score != null ? (
        <dl className="stats">
          <div><dt>Score (median)</dt><dd>{Math.round(m.score * 100)}%</dd></div>
          {s.min != null && <div><dt>Range</dt><dd>{Math.round(s.min * 100)}–{Math.round(s.max * 100)}%</dd></div>}
          {margin != null && <div><dt>{z === "mid" ? "To nearest threshold" : "Clear of threshold by"}</dt><dd>{Math.round(margin * 100)} pts</dd></div>}
          {unit === "frames" ? <div><dt>Frames with a face</dt><dd>{m.n_face_frames} / {m.n_frames_sampled}</dd></div> : <div><dt>Speech windows</dt><dd>{m.n_windows} × 4 s</dd></div>}
          <div><dt>Model</dt><dd>{m.model?.name}</dd></div>
        </dl>
      ) : <p className="hint">{(m.reasons[0] ?? "Nothing to analyse").replace(/\.?$/, ".")} That says nothing about authenticity.</p>}
      {m.score != null && m.reasons.length > 0 && <p className="hint">{m.reasons.join(" · ")}</p>}
    </section>
  );
}
