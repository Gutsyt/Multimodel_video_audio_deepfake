import { useState } from "react";
import { API_BASE, getReady } from "../api";
import { useAppData } from "../AppData";
import { PageHead } from "../components/Shell";
import { Icon } from "../components/Icon";
import { RETENTION_NOTE } from "./HistoryPage";

export function SystemPage() {
  const { caps, capsError, recheck } = useAppData();
  const [ready, setReady] = useState<string | null>(null);
  async function check() {
    recheck();
    try { setReady((await getReady()).status); } catch { setReady("not ready"); }
  }
  return (
    <div className="stack">
      <PageHead title="System status" lead="Everything here is read from the running backend when you open or re-check this page."
        actions={<button className="btn" onClick={() => void check()}><Icon name="refresh" size={16} /> Re-check</button>} />
      {capsError && <div className="errbox" role="alert"><b>The backend is unreachable</b><span>API address in use: <span className="mono">{API_BASE || "same origin (/api)"}</span>. Start the backend, or set VITE_API_BASE when building the frontend.</span></div>}
      {caps && <>
        <section className="panel"><header><h2>Models</h2><span className={`status ${caps.models_state === "ready" ? "ok" : "neutral"}`}>{caps.models_state}</span></header>
          {ready && <p className="muted" style={{ marginBottom: 8 }}>/api/ready answered: <b>{ready}</b></p>}
          {caps.models.map((m) => (
            <div key={m.role} style={{ padding: "12px 0", borderTop: "1px solid var(--line-soft)" }}>
              <div className="health-row" style={{ border: 0, padding: 0 }}><span className={`pulse ${m.state === "loaded" ? "up" : m.state === "failed" ? "down" : ""}`} /><b style={{ textTransform: "capitalize" }}>{m.role}: {m.card?.name ?? "no model"}</b><span className="muted">{m.state}{m.error_code ? ` (${m.error_code})` : ""}</span></div>
              {m.card && <details><summary>Model card</summary><dl className="facts"><dt>Version</dt><dd className="mono">{m.card.version}</dd><dt>License</dt><dd>{m.card.license}</dd><dt>Trained on</dt><dd>{m.card.trained_on}</dd><dt>Source</dt><dd><a href={m.card.source_url} target="_blank" rel="noreferrer">{m.card.source_url}</a></dd></dl><ul className="lim">{m.card.limitations.map((l) => <li key={l}>{l}</li>)}</ul></details>}
            </div>))}
        </section>
        <div className="grid2">
          <section className="panel"><header><h2>Limits</h2></header><dl className="facts">
            <dt>Max upload</dt><dd>{caps.limits.max_upload_mb} MB</dd><dt>Max duration</dt><dd>{caps.limits.max_duration_s} s (audio, video)</dd>
            {caps.limits.max_image_megapixels != null && <><dt>Max image</dt><dd>{caps.limits.max_image_megapixels} megapixels</dd></>}
            <dt>Frames sampled</dt><dd>{caps.limits.frames_sampled} per video</dd><dt>Queue</dt><dd>{caps.limits.max_pending_jobs} pending jobs, one at a time</dd>
            <dt>Version</dt><dd className="mono">avdd {caps.avdd_version}</dd><dt>API address</dt><dd className="mono">{API_BASE || "same origin"}</dd></dl></section>
          <section className="panel"><header><h2>Privacy and data</h2></header>
            <p className="muted">{caps.retention}</p><p className="muted" style={{ marginTop: 8 }}>{RETENTION_NOTE}</p>
            <p className="muted" style={{ marginTop: 8 }}>You can delete a record from History or Reports at any time. Models are downloaded once from Hugging Face; no media is sent to third parties.</p></section>
        </div>
        <section className="panel quiet"><h2>Evaluation status</h2><p className="muted" style={{ marginTop: 8 }}>{caps.evaluation_status} No claim of enterprise security, legal admissibility or forensic reliability is made.</p></section>
      </>}
    </div>
  );
}
