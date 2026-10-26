import { useEffect, useMemo, useRef, useState } from "react";
import { ApiRequestError, uploadMedia, waitForAnalysis, type ApiErrorBody, type MediaKind } from "../api";
import { useAppData } from "../AppData";
import { Icon } from "../components/Icon";
import { PageHead } from "../components/Shell";
import { useToast } from "../components/Toasts";
import { KIND_LABEL, describeError, formatBytes } from "../lib/format";
import { acceptAttr, acceptedExtensions, validateFile } from "../lib/validate";

type View =
  | { s: "idle" }
  | { s: "ready" }
  | { s: "busy"; phase: "upload" | "queued" | "stage"; loaded: number; total: number; stage: string | null; t0: number }
  | { s: "error"; error: ApiErrorBody };

const STAGE_LABEL: Record<string, string> = { probing: "Reading the file", audio: "Audio branch", visual: "Visual branch", fusion: "Combining evidence" };
const STAGES_FOR: Record<MediaKind, string[]> = { video: ["probing", "audio", "visual", "fusion"], audio: ["probing", "audio", "fusion"], image: ["probing", "visual", "fusion"] };

export function AnalyzePage() {
  const { caps, capsError, refreshHistory } = useAppData();
  const toast = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [problems, setProblems] = useState<string[]>([]);
  const [consent, setConsent] = useState(false);
  const [over, setOver] = useState(false);
  const [view, setView] = useState<View>({ s: "idle" });
  const [now, setNow] = useState(Date.now());
  const abort = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const accepted = acceptedExtensions(caps);
  const check = useMemo(() => (file ? validateFile(file, caps) : null), [file, caps]);
  const url = useMemo(() => (file && check?.kind ? URL.createObjectURL(file) : null), [file, check?.kind]);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  useEffect(() => { if (view.s !== "busy") return; const t = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(t); }, [view.s]);
  useEffect(() => () => abort.current?.abort(), []);

  function pick(f: File | null | undefined) {
    if (!f) return;
    const c = validateFile(f, caps);
    setFile(f); setProblems(c.errors); setView(c.errors.length ? { s: "idle" } : { s: "ready" });
  }
  function clear() { setFile(null); setProblems([]); setView({ s: "idle" }); if (input.current) input.current.value = ""; }

  async function run() {
    if (!file || !check?.kind) return;
    const ctl = new AbortController(); abort.current = ctl;
    const t0 = Date.now();
    setView({ s: "busy", phase: "upload", loaded: 0, total: file.size, stage: null, t0 });
    try {
      const created = await uploadMedia(file, (loaded, total) => setView({ s: "busy", phase: "upload", loaded, total, stage: null, t0 }), ctl.signal);
      setView({ s: "busy", phase: "queued", loaded: file.size, total: file.size, stage: null, t0 });
      const final = await waitForAnalysis(created.id, (st) => {
        if (st.status === "processing") setView({ s: "busy", phase: "stage", loaded: file.size, total: file.size, stage: st.stage, t0 });
      }, ctl.signal);
      await refreshHistory();
      if (final.status === "failed" && final.error) { setView({ s: "error", error: final.error }); return; }
      toast.push("Analysis finished.", "success");
      window.location.hash = `#/analysis/${created.id}`;
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") { setView({ s: "ready" }); toast.push("Upload cancelled."); return; }
      setView({ s: "error", error: e instanceof ApiRequestError ? e.body : { code: "internal_error", message: "Unexpected error." } });
    }
  }

  const modelsReady = caps?.models_state === "ready";
  const blocked = capsError ? "The backend is not reachable." : !caps ? "Checking the backend…" : caps.models_state === "loading" ? "Models are still loading; this usually takes a minute after start." : caps.models_state === "unavailable" ? "No analysis model could be loaded on the server." : null;
  const busy = view.s === "busy";
  const stageList = check?.kind ? STAGES_FOR[check.kind] : [];

  return (
    <div className="stack">
      <PageHead title="New analysis" lead="Upload one file. It is processed from a temporary copy that the server deletes when the analysis ends." />
      {blocked && <div className="errbox" role="status"><b>Not ready</b><span>{blocked}</span></div>}

      {!file && (
        <div className={`drop${over ? " over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}>
          <svg className="glyph" viewBox="0 0 74 74" aria-hidden="true"><g fill="none" strokeWidth="3" strokeLinecap="round"><path stroke="#3b79ff" d="M12 30v14M22 22v30M32 14v46" /><rect stroke="#f4a62a" x="44" y="22" width="18" height="18" rx="3" /><path stroke="#f4a62a" d="M44 50h18M44 58h11" /></g></svg>
          <div><h2>Drop a file here</h2><p className="muted" style={{ marginTop: 4 }}>or choose one from your device</p></div>
          <button className="btn primary" onClick={() => input.current?.click()}><Icon name="analyze" /> Choose file</button>
          <div className="kinds" aria-label="Accepted formats">
            {(["video", "audio", "image"] as MediaKind[]).map((k) => <span key={k}><Icon name={k} size={15} /> <b>{KIND_LABEL[k]}</b> {accepted[k].join(" ")}</span>)}
          </div>
          <small className="faint">{caps ? `Up to ${caps.limits.max_upload_mb} MB; audio and video up to ${caps.limits.max_duration_s} s${caps.limits.max_image_megapixels ? `; images up to ${caps.limits.max_image_megapixels} MP` : ""}.` : "Limits load from the server."}</small>
        </div>
      )}
      <input ref={input} type="file" hidden accept={acceptAttr(caps)} onChange={(e) => pick(e.target.files?.[0])} aria-label="Choose a media file" />

      {file && (
        <section className="panel" aria-label="Selected file">
          <div className="preview">
            <div className="media">
              {check?.kind === "video" && url && <video src={url} controls preload="metadata" />}
              {check?.kind === "image" && url && <img src={url} alt="Preview of the selected image" />}
              {check?.kind === "audio" && url && <audio src={url} controls preload="metadata" />}
              {!check?.kind && <Icon name="file" size={40} />}
            </div>
            <div style={{ display: "grid", gap: 14 }}>
              <dl className="facts">
                <dt>File</dt><dd className="mono">{file.name}</dd>
                <dt>Size</dt><dd>{formatBytes(file.size)}</dd>
                <dt>Type</dt><dd>{check?.kind ? KIND_LABEL[check.kind] : "unsupported"} <span className="faint">({file.type || "unknown MIME"})</span></dd>
              </dl>
              {problems.length > 0 && <div className="errbox" role="alert"><b>This file cannot be analyzed</b><ul>{problems.map((p) => <li key={p}>{p}</li>)}</ul></div>}
              {!problems.length && view.s !== "error" && (
                <>
                  <label className="check"><input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} disabled={busy} />
                    <span>I have the right to analyze this file and I understand it is sent to the analysis server. Faces and voices of other people are personal data.</span></label>
                  <div className="row">
                    <button className="btn primary" disabled={!consent || busy || !modelsReady} onClick={() => void run()}>{busy ? "Analyzing…" : "Run analysis"}</button>
                    <button className="btn" disabled={busy} onClick={() => input.current?.click()}>Replace file</button>
                    <button className="btn ghost danger" disabled={busy} onClick={clear}><Icon name="close" size={15} /> Remove</button>
                  </div>
                </>
              )}
              {view.s === "error" && (
                <div className="errbox" role="alert"><b>The analysis did not complete</b><span>{describeError(view.error)}</span>
                  <small className="faint">Code: {view.error.code}</small>
                  <div className="row"><button className="btn primary" onClick={() => void run()}><Icon name="refresh" size={15} /> Retry</button><button className="btn" onClick={clear}>Choose another file</button></div></div>
              )}
              {problems.length > 0 && <div className="row"><button className="btn" onClick={() => input.current?.click()}>Choose another file</button><button className="btn ghost" onClick={clear}>Remove</button></div>}
            </div>
          </div>
        </section>
      )}

      {view.s === "busy" && (
        <section className="panel" aria-label="Progress" aria-live="polite">
          <header><h2>Progress</h2><span className="faint mono">{Math.round((now - view.t0) / 1000)} s</span></header>
          <ol className="steps">
            {(["upload", "queued", ...stageList] as string[]).map((k) => {
              const order = ["upload", "queued", ...stageList];
              const cur = view.phase === "upload" ? "upload" : view.phase === "queued" ? "queued" : view.stage ?? "probing";
              const ci = order.indexOf(cur), i = order.indexOf(k);
              const cls = i < ci ? "done" : i === ci ? "now" : "";
              return (
                <li key={k} className={cls}><span className="dot" />
                  <span>{k === "upload" ? "Uploading" : k === "queued" ? "Waiting for the analysis worker" : STAGE_LABEL[k] ?? k}
                    {k === "upload" && view.phase === "upload" && <span className="bar-track" role="progressbar" aria-label="Upload progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round((view.loaded / Math.max(1, view.total)) * 100)} style={{ display: "block", marginTop: 6 }}><i style={{ width: `${(view.loaded / Math.max(1, view.total)) * 100}%` }} /></span>}
                  </span>
                  <span className="faint mono">{k === "upload" && view.phase === "upload" ? `${formatBytes(view.loaded)} / ${formatBytes(view.total)}` : cls === "done" ? "done" : cls === "now" ? "in progress" : ""}</span>
                </li>);
            })}
          </ol>
          <div className="row" style={{ marginTop: 12 }}>
            {view.phase === "upload" ? <button className="btn" onClick={() => abort.current?.abort()}>Cancel upload</button> : <small className="faint">The server cannot cancel an analysis once it has started.</small>}
          </div>
        </section>
      )}
    </div>
  );
}
