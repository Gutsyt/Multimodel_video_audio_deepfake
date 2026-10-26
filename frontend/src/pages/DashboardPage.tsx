import { useAppData, toneOf } from "../AppData";
import { CoverageMatrix, RunDots, RunStrip, StatusMix } from "../components/Dashboard";
import { Icon } from "../components/Icon";
import { StatusMark } from "../components/StatusMark";
import { KIND_LABEL, formatScore, formatWhen } from "../lib/format";
import type { MediaKind } from "../api";

function Disclaimer() {
  return (
    <section className="panel quiet" aria-label="About this prototype">
      <h2>Research prototype</h2>
      <p className="muted" style={{ marginTop: 8, maxWidth: "80ch" }}>
        AVDD runs <b>pretrained</b> open models for inference only. Nothing was trained or fine-tuned here and no accuracy evaluation has been completed, so scores are raw model outputs: not calibrated probabilities, and the 0.50 flag threshold is an unvalidated default.
        Output is not forensic or legal proof. Genuine media can be flagged and manipulated media can pass.
      </p>
    </section>
  );
}

export function DashboardPage() {
  const { history, historyLoaded, historyError, caps, capsError } = useAppData();
  const done = history.filter((h) => h.status === "completed" || h.status === "failed");
  const recent = history.slice(0, 5);
  return (
    <div className="stack">
      <section className="hero" aria-labelledby="hero-h">
        <div>
          <h1 id="hero-h">Is this media showing signs of manipulation?</h1>
          <p>Upload a video, an audio clip or a still image. Two pretrained detectors look at the voice and at the face, and each reports what it could and could not use.</p>
          <div className="row">
            <a className="btn primary" href="#/analyze"><Icon name="analyze" /> Analyze media</a>
            <a className="btn" href="#/history"><Icon name="history" /> History</a>
          </div>
        </div>
        <div aria-label="Stored analyses">
          {done.length ? (
            <div style={{ display: "grid", gap: 14 }}><RunStrip items={done} /><StatusMix items={done} /></div>
          ) : (
            <div className="kinds" style={{ justifyContent: "flex-start", gap: 12 }}>
              {(["video", "audio", "image"] as MediaKind[]).map((k) => <span key={k} style={{ padding: "18px 22px", flexDirection: "column" }}><Icon name={k} size={30} />{KIND_LABEL[k]}</span>)}
            </div>
          )}
        </div>
      </section>

      {historyError && <div className="errbox" role="alert"><b>History could not be loaded.</b><span>The backend did not answer. Charts below need stored records.</span></div>}

      {!historyLoaded ? <p className="muted" role="status">Loading stored analyses…</p> : done.length === 0 ? (
        <section className="panel"><div className="empty">
          <Icon name="analyze" size={34} />
          <h2>No analyses yet</h2>
          <p style={{ maxWidth: "52ch" }}>This looks like a fresh install. Run your first analysis and its results, branch coverage and score history will appear here. Only results are stored, never the media.</p>
          <a className="btn primary" href="#/analyze">Analyze media</a>
        </div></section>
      ) : (
        <>
          <div className="grid2">
            <section className="panel"><header><h2>Raw scores by run</h2></header><RunDots items={done} threshold={0.5} /></section>
            <section className="panel"><header><h2>Which branches ran</h2></header><CoverageMatrix items={done} /></section>
          </div>
          <section className="panel">
            <header><h2>Recent analyses</h2><a href="#/history">All records</a></header>
            <div className="tablewrap"><table>
              <thead><tr><th>When</th><th>Media</th><th>Outcome</th><th>Audio</th><th>Visual</th><th /></tr></thead>
              <tbody>{recent.map((s) => { const t = toneOf(s); return (
                <tr key={s.id}>
                  <td>{formatWhen(s.created_utc)}</td><td>{s.media_kind ? KIND_LABEL[s.media_kind as MediaKind] ?? s.media_kind : "n/a"}</td>
                  <td>{t === "running" ? <span className="status neutral">Running</span> : <StatusMark tone={t} />}</td>
                  <td className="num">{formatScore(s.audio_score)}</td><td className="num">{formatScore(s.visual_score)}</td>
                  <td><a href={`#/analysis/${s.id}`}>Open</a></td>
                </tr>); })}</tbody>
            </table></div>
          </section>
        </>
      )}

      <section className="panel" aria-label="Backend health">
        <header><h2>Backend and models</h2><span className="faint">live check of this session</span></header>
        {capsError ? <p>Backend unreachable. Start it and reload.</p> : !caps ? <p className="muted">Checking…</p> : (
          caps.models.map((m) => (
            <div className="health-row" key={m.role}>
              <span className={`pulse ${m.state === "loaded" ? "up" : m.state === "failed" ? "down" : ""}`} />
              <div><b style={{ textTransform: "capitalize" }}>{m.role} model</b> <span className="faint">{m.card ? `${m.card.name}` : ""}</span></div>
              <span className="muted">{m.state === "failed" ? `failed to load (${m.error_code})` : m.state}</span>
            </div>
          ))
        )}
      </section>
      <Disclaimer />
    </div>
  );
}
export { Disclaimer };
