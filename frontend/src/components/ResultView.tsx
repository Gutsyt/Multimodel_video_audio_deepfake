import type { AnalysisResult } from "../api";
import { fusionHeadline, formatScore, formatTime, STATUS_LABEL } from "../format";
import type { ModalityStatus } from "../api";
import { ScoreBar } from "./Bits";
import { ModalityCard } from "./ModalityCard";

export function ResultView({ result, analysisId, reportHref, onDelete }: {
  result: AnalysisResult; analysisId: string; reportHref: string; onDelete: () => void;
}) {
  const f = result.fusion;
  const h = fusionHeadline(f);
  const duration = result.media.duration_s;
  return (
    <div className="result">
      {result.is_test_fixture && (
        <div className="banner banner-warn" role="alert">
          <strong>TEST FIXTURE MODEL:</strong> this result comes from a stub that returns fixed numbers. It is not a detector.
        </div>
      )}
      <section className={`card fusion tone-${h.tone}`} aria-labelledby="h-fusion">
        <h2 id="h-fusion">Combined model output</h2>
        <p className="headline">{h.title}</p>
        <p>{h.detail}</p>
        {f.fused_score !== null && <ScoreBar label="Raw fused score (not a probability)" value={f.fused_score} />}
        <dl className="facts">
          <dt>Method</dt><dd>{f.method}</dd>
          {Object.keys(f.weights_used).length > 0 && (
            <><dt>Weights used</dt><dd>{Object.entries(f.weights_used).map(([k, v]) => `${k} ${formatScore(v)}`).join(", ")}{f.weights_validated ? "" : " (unvalidated default)"}</dd></>
          )}
          <dt>Threshold</dt><dd>{formatScore(f.threshold)}{f.threshold_validated ? "" : " (unvalidated default)"}</dd>
          <dt>Calibrated</dt><dd>{f.calibrated ? "yes" : "no: scores are not probabilities"}</dd>
          {Object.entries(f.excluded_modalities).map(([k, v]) => (
            <><dt key={`${k}-t`}>Not used: {k}</dt><dd key={`${k}-d`}>{STATUS_LABEL[v as ModalityStatus] ?? v}. This is neither evidence of authenticity nor of manipulation.</dd></>
          ))}
        </dl>
        {f.notes.length > 0 && <ul className="notes">{f.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
      </section>

      <div className="grid2">
        {(["audio", "visual"] as const).map((k) => {
          const m = result.modalities[k];
          return m ? <ModalityCard key={k} name={k} result={m} duration={duration} /> : null;
        })}
      </div>

      <section className="card" aria-labelledby="h-read">
        <h3 id="h-read">How to read this</h3>
        <ul>
          <li>{result.interpretation.score_meaning}</li>
          <li>{result.interpretation.flag_meaning}</li>
          <li>{result.interpretation.not_forensic_proof}</li>
          <li>No performance evaluation of this system has been completed.</li>
        </ul>
        <p className="muted small">
          Media: {result.media.container ?? "unknown container"}{duration ? `, ${formatTime(duration)}` : ""}
          {result.media.width ? `, ${result.media.width}×${result.media.height}` : ""}. Analyzed up to {formatTime(result.analyzed_limit_s)};
          processing took {formatTime(result.total_runtime_s)}. avdd {result.avdd_version}, schema {result.schema_version}. Analysis {analysisId.slice(0, 8)}.
        </p>
        <div className="actions">
          <a className="btn" href={reportHref} download>Download JSON report</a>
          <button className="btn btn-quiet" onClick={onDelete}>Delete this analysis</button>
        </div>
      </section>
    </div>
  );
}
