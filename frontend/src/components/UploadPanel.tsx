import { useEffect, useRef, useState } from "react";
import type { Capabilities } from "../api";
import { formatBytes, formatTime } from "../format";

export interface Picked { file: File; durationS: number | null; width: number | null; height: number | null; previewOk: boolean }

export function UploadPanel({ caps, disabled, onAnalyze }: {
  caps: Capabilities | null; disabled: boolean; onAnalyze: (file: File) => void;
}) {
  const [picked, setPicked] = useState<Picked | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  function choose(file: File | undefined) {
    if (!file) return;
    setProblem(null);
    if (url) URL.revokeObjectURL(url);
    const u = URL.createObjectURL(file);
    setUrl(u);
    setPicked({ file, durationS: null, width: null, height: null, previewOk: true });
    if (caps && file.size > caps.limits.max_upload_mb * 1e6) {
      setProblem(`This file is ${formatBytes(file.size)}; the limit is ${caps.limits.max_upload_mb} MB.`);
    }
  }

  function onMeta(e: React.SyntheticEvent<HTMLVideoElement>) {
    const v = e.currentTarget;
    setPicked((p) => (p ? { ...p, durationS: Number.isFinite(v.duration) ? v.duration : null, width: v.videoWidth || null, height: v.videoHeight || null } : p));
    if (caps && Number.isFinite(v.duration) && v.duration > caps.limits.max_duration_s) {
      setProblem(`This video is ${formatTime(v.duration)} long; the limit is ${caps.limits.max_duration_s} s.`);
    }
  }

  const blocked = !!problem || !picked || disabled;
  return (
    <section className="card" aria-labelledby="h-upload">
      <h2 id="h-upload">1. Choose a video</h2>
      <div
        className={`drop ${drag ? "drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files[0]); }}
      >
        <p>Drag a video here, or</p>
        <button className="btn" onClick={() => input.current?.click()} disabled={disabled}>Browse files</button>
        <input ref={input} type="file" accept="video/*,.mkv,.avi,.mov,.mp4,.webm" hidden onChange={(e) => choose(e.target.files?.[0])} aria-label="Video file" />
        {caps && <p className="muted small">MP4, MOV, MKV, WebM or AVI · up to {caps.limits.max_upload_mb} MB · up to {caps.limits.max_duration_s} s. Only use videos you have the right to analyze.</p>}
      </div>
      {picked && url && (
        <div className="preview">
          <video src={url} controls preload="metadata" onLoadedMetadata={onMeta} onError={() => setPicked((p) => (p ? { ...p, previewOk: false } : p))} />
          <ul className="small">
            <li>Size: {formatBytes(picked.file.size)}</li>
            {picked.durationS !== null && <li>Duration: {formatTime(picked.durationS)}</li>}
            {picked.width && <li>Resolution: {picked.width}×{picked.height}</li>}
            {!picked.previewOk && <li className="muted">This browser cannot preview the file; analysis can still run. Duration is checked by the server.</li>}
          </ul>
        </div>
      )}
      {problem && <p className="field-error" role="alert">{problem}</p>}
      <button className="btn btn-primary" disabled={blocked} onClick={() => picked && onAnalyze(picked.file)}>Analyze video</button>
      {caps?.models_state === "loading" && <p className="muted small">Models are still loading; analysis will be available shortly.</p>}
      {caps?.models_state === "unavailable" && <p className="field-error">No analysis model could be loaded on the server. See the model status above.</p>}
    </section>
  );
}
