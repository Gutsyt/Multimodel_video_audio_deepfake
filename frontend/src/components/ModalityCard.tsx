import type { ModalityResult } from "../api";
import { STATUS_EXPLANATION, formatScore, formatTime, modalityNoun } from "../format";
import { ScoreBar, StatusBadge } from "./Bits";
import { Timeline } from "./Timeline";

export function ModelInfo({ result }: { result: ModalityResult }) {
  const m = result.model;
  if (!m) return null;
  return (
    <details className="model-info">
      <summary>Model, license and limitations</summary>
      {m.is_test_fixture && <p className="banner banner-warn">Test fixture model: returns fixed numbers and is NOT a detector.</p>}
      <dl>
        <dt>Model</dt><dd>{m.name} <span className="muted">({m.version})</span></dd>
        <dt>License</dt><dd>{m.license}</dd>
        <dt>Source</dt><dd><a href={m.source_url} target="_blank" rel="noreferrer">{m.source_url}</a></dd>
        <dt>Trained on</dt><dd>{m.trained_on}</dd>
        <dt>Citation</dt><dd>{m.citation}</dd>
      </dl>
      <strong>Limitations</strong>
      <ul>{m.limitations.map((l) => <li key={l}>{l}</li>)}</ul>
    </details>
  );
}

export function ModalityCard({ name, result, duration }: { name: "audio" | "visual"; result: ModalityResult; duration: number | null }) {
  const title = name === "audio" ? "Audio branch" : "Visual branch";
  const noun = modalityNoun(name === "audio" ? "audio" : "video");
  const available = result.status === "available" && result.score !== null;
  return (
    <section className={`card modality status-${result.status}`} aria-labelledby={`h-${name}`}>
      <header>
        <h3 id={`h-${name}`}>{title}</h3>
        <StatusBadge status={result.status} />
      </header>
      {available ? (
        <>
          <ScoreBar label="Raw model score (not a probability)" value={result.score as number} />
          <p className="muted small">{result.aggregation}. Higher means the model finds the input more similar to its training examples of manipulated media.</p>
        </>
      ) : (
        <p>{STATUS_EXPLANATION[result.status](noun)}</p>
      )}
      {result.reasons.length > 0 && (
        <ul className="reasons">{result.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
      )}
      {available && duration && (
        <>
          <h4>Evidence over time</h4>
          <Timeline duration={duration} segments={result.segments} frames={result.frames} />
          <details>
            <summary>Show the numbers</summary>
            <table className="table">
              <thead><tr><th>Time</th><th>Detail</th><th>Raw score</th></tr></thead>
              <tbody>
                {result.segments?.map((s) => (
                  <tr key={s.start_s}><td>{formatTime(s.start_s)}–{formatTime(s.end_s)}</td><td>audio window</td><td>{formatScore(s.score)}</td></tr>
                ))}
                {result.frames?.map((f) => (
                  <tr key={f.frame_index}>
                    <td>{formatTime(f.t_s)}</td>
                    <td>{f.n_faces === 0 ? "no face found" : `${f.n_faces} face(s); largest scored`}</td>
                    <td>{f.score === null ? "not scored" : formatScore(f.score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.frames && <p className="muted small">Timestamps are nominal (frame index / frame rate). Face detector: {result.face_detector}.</p>}
          </details>
        </>
      )}
      <ModelInfo result={result} />
    </section>
  );
}
