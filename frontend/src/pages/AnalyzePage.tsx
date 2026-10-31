import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiRequestError, uploadMedia, waitForAnalysis, type ApiErrorBody, type MediaKind } from "../api";
import { useAppData } from "../AppData";
import { Icon } from "../components/Icon";
import { rise, stagger } from "../components/Shell";
import { useToast } from "../components/Toasts";
import { KIND_LABEL, describeError, formatBytes } from "../lib/format";
import { rememberMedia } from "../lib/session";
import { acceptAttr, acceptedExtensions, validateFile } from "../lib/validate";

type Phase = "upload" | "queued" | "probing" | "audio" | "visual" | "verdict";
type View = { s: "idle" } | { s: "busy"; phase: Phase; progress: number } | { s: "error"; error: ApiErrorBody };

const STEP_LABEL: Record<Phase, string> = {
  upload: "Uploading", queued: "Waiting for the worker", probing: "Reading the file",
  audio: "Checking the voice", visual: "Checking the face", verdict: "Weighing the evidence",
};
const STEPS_FOR: Record<MediaKind, Phase[]> = {
  video: ["upload", "probing", "audio", "visual", "verdict"], audio: ["upload", "probing", "audio", "verdict"], image: ["upload", "probing", "visual", "verdict"],
};

export function AnalyzePage() {
  const { caps, capsError, refreshHistory } = useAppData();
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [view, setView] = useState<View>({ s: "idle" });
  const input = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);
  const check = useMemo(() => (file ? validateFile(file, caps) : null), [file, caps]);
  const url = useMemo(() => (file && check?.kind ? URL.createObjectURL(file) : null), [file, check?.kind]);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  useEffect(() => () => abort.current?.abort(), []);

  const accepted = acceptedExtensions(caps);
  const ready = caps?.models_state === "ready";
  const blocked = capsError ? "The analysis server is not reachable. Start the backend and reload."
    : caps?.models_state === "loading" ? "Models are loading. This takes about a minute after the server starts."
    : caps?.models_state === "unavailable" ? (caps.models.some((m) => m.error_code === "head_missing") ? "The trained model files are missing. Train them first: see docs/TRAINING.md." : "No analysis model could be loaded on the server.") : null;
  const busy = view.s === "busy";

  function pick(f: File | null | undefined) {
    if (!f) return;
    setFile(f); setView({ s: "idle" });
  }
  function reset() { setFile(null); setView({ s: "idle" }); if (input.current) input.current.value = ""; }

  async function run() {
    if (!file || !check?.kind || check.errors.length) return;
    const ctl = new AbortController(); abort.current = ctl;
    setView({ s: "busy", phase: "upload", progress: 0 });
    try {
      const created = await uploadMedia(file, (l, t) => setView({ s: "busy", phase: "upload", progress: l / Math.max(1, t) }), ctl.signal);
      rememberMedia(created.id, file);
      setView({ s: "busy", phase: "queued", progress: 1 });
      const final = await waitForAnalysis(created.id, (st) => {
        if (st.status === "processing" && st.stage) setView({ s: "busy", phase: (st.stage as Phase), progress: 1 });
      }, ctl.signal, 500);
      void refreshHistory();
      if (final.status === "failed" && final.error) { setView({ s: "error", error: final.error }); return; }
      window.location.hash = `#/analysis/${created.id}`;
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") { setView({ s: "idle" }); toast.push("Upload cancelled."); return; }
      setView({ s: "error", error: e instanceof ApiRequestError ? e.body : { code: "internal_error", message: "Unexpected error." } });
    }
  }

  const steps = check?.kind ? STEPS_FOR[check.kind] : [];
  const cur = view.s === "busy" ? (view.phase === "queued" ? "probing" : view.phase) : null;

  return (
    <section className="hero">
      <motion.div className="hero-copy" variants={stagger} initial="hidden" animate="show">
        <motion.h1 className="display" variants={rise}>Is it <em>real</em>,<br />or is it made?</motion.h1>
        <motion.p className="lead" variants={rise}>
          Drop a video, a voice clip or a photo. The face and the voice are checked separately by models trained on
          FaceForensics++, Celeb-DF, WildDeepfake, ASVspoof and In-the-Wild, then judged against thresholds set to keep false alarms on genuine media rare.
        </motion.p>
        <motion.div className="facts-inline" variants={rise}>
          <span><b>2</b>independent checks</span>
          <span><b>3</b>possible verdicts</span>
          <span><b>0</b>files kept after analysis</span>
        </motion.div>
        {blocked && <motion.div className="alert soft" variants={rise} role="status">{blocked}</motion.div>}
      </motion.div>

      <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}>
        <input ref={input} type="file" hidden accept={acceptAttr(caps)} onChange={(e) => pick(e.target.files?.[0])} aria-label="Choose a media file" />
        <AnimatePresence mode="wait">
          {!file ? (
            <motion.div key="drop" className={`drop card${over ? " over" : ""}`} role="button" tabIndex={0} aria-label="Choose or drop a file"
              animate={{ scale: over ? 1.02 : 1 }} exit={{ opacity: 0, scale: 0.97 }} transition={{ type: "spring", stiffness: 300, damping: 24 }}
              onClick={() => input.current?.click()} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") input.current?.click(); }}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}>
              <svg className="border" aria-hidden="true"><rect x="1" y="1" rx="21" /></svg>
              <motion.div className="orb" animate={{ y: over ? -6 : [0, -5, 0] }} transition={over ? { duration: 0.2 } : { duration: 3.2, repeat: Infinity, ease: "easeInOut" }}>
                <Icon name="analyze" size={30} />
              </motion.div>
              <div>
                <h2 className="h2">{over ? "Let go to add it" : "Drop a file to check"}</h2>
                <p className="muted" style={{ marginTop: 6 }}>or click to choose from your device</p>
              </div>
              <div className="kinds">
                {(["video", "audio", "image"] as MediaKind[]).map((k) => <span className="tag" key={k} title={accepted[k].join(" ")}><Icon name={k} size={14} />{KIND_LABEL[k]}</span>)}
              </div>
              <small className="faint">{caps ? `Up to ${Math.round(caps.limits.max_upload_mb)} MB · first ${caps.limits.max_duration_s} s analysed` : " "}</small>
            </motion.div>
          ) : (
            <motion.div key="sel" className="card selected" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <div className="preview">
                {check?.kind === "video" && url && <video src={url} muted playsInline autoPlay loop />}
                {check?.kind === "image" && url && <img src={url} alt="Selected file preview" />}
                {check?.kind === "audio" && url && <audio src={url} controls />}
                {!check?.kind && <Icon name="file" size={40} />}
                {busy && check?.kind !== "audio" && <motion.div className="scan" initial={{ top: "-30%" }} animate={{ top: "100%" }} transition={{ duration: 1.8, repeat: Infinity, ease: "linear" }} />}
              </div>
              <div className="filemeta">
                <span className="name">{file.name}</span>
                <span className="faint mono" style={{ fontSize: 13 }}>{formatBytes(file.size)}</span>
              </div>

              {check && check.errors.length > 0 && <div className="alert" role="alert">{check.errors.map((p) => <span key={p}>{p}</span>)}</div>}
              {view.s === "error" && <div className="alert" role="alert"><b>The analysis did not complete</b><span>{describeError(view.error)}</span></div>}

              {busy ? (
                <ol className="steps" aria-live="polite">
                  {steps.map((k) => {
                    const i = steps.indexOf(k), ci = steps.indexOf(cur as Phase);
                    const cls = i < ci ? "done" : i === ci ? "now" : "";
                    return (
                      <motion.li key={k} className={cls} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.06 }}>
                        <span className="tick">{cls === "done" && <Icon name="check" size={11} />}</span>
                        <span>{STEP_LABEL[k]}
                          {k === "upload" && cls === "now" && view.s === "busy" && <span className="track" style={{ display: "block", marginTop: 6 }}><motion.i animate={{ width: `${view.progress * 100}%` }} /></span>}
                        </span>
                        <span className="faint mono" style={{ fontSize: 12 }}>{cls === "done" ? "done" : ""}</span>
                      </motion.li>
                    );
                  })}
                </ol>
              ) : (
                <div className="row">
                  <button className="btn primary" disabled={!ready || !check?.kind || !!check?.errors.length} onClick={() => void run()}>
                    {view.s === "error" ? "Try again" : "Analyze"} <Icon name="arrow" size={16} />
                  </button>
                  <button className="btn ghost" onClick={reset}>Choose another</button>
                </div>
              )}
              {busy && view.s === "busy" && view.phase === "upload" && <button className="btn ghost small" onClick={() => abort.current?.abort()}>Cancel</button>}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </section>
  );
}
