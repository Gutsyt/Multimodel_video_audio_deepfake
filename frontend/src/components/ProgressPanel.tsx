import { useEffect, useState } from "react";
import type { AnalysisStatus } from "../api";
import { formatBytes } from "../format";

const STAGE_LABEL: Record<string, string> = {
  probing: "Reading and checking the media",
  audio: "Audio model",
  visual: "Face detection and visual model",
  fusion: "Combining the evidence",
};

export type Progress =
  | { kind: "uploading"; loaded: number; total: number }
  | { kind: "analyzing"; status: AnalysisStatus | null };

/** Steps come from the backend's real stage events; nothing here is simulated. */
export function ProgressPanel({ progress, stages, onCancel }: { progress: Progress; stages: string[]; onCancel?: () => void }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const t = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(t);
  }, []);

  const steps = ["upload", "queued", ...stages];
  let active = 0;
  if (progress.kind === "analyzing") {
    const s = progress.status;
    active = !s || s.status === "queued" ? 1 : Math.max(2, 2 + stages.indexOf(s.stage ?? ""));
  }
  const label = (id: string) =>
    id === "upload" ? "Uploading the file" : id === "queued" ? "Waiting in the queue" : STAGE_LABEL[id] ?? id;

  return (
    <section className="card" aria-live="polite" aria-labelledby="h-progress">
      <h2 id="h-progress">Analysis in progress</h2>
      <ol className="steps">
        {steps.map((id, i) => (
          <li key={id} className={i < active ? "done" : i === active ? "active" : ""} aria-current={i === active ? "step" : undefined}>
            <span className="dot" aria-hidden="true">{i < active ? "✓" : i + 1}</span>
            <span>
              {label(id)}
              {i === active && id === "upload" && progress.kind === "uploading" && (
                <span className="muted"> {formatBytes(progress.loaded)} of {formatBytes(progress.total)} ({Math.round((progress.loaded / Math.max(1, progress.total)) * 100)}%)</span>
              )}
              {i === active && id !== "upload" && <span className="spinner" aria-hidden="true" />}
            </span>
          </li>
        ))}
      </ol>
      <p className="muted small">
        Elapsed {elapsed} s. Steps that do not apply (for example the audio model when the file has no audio track) are skipped by the server and noted in the result.
        {progress.kind === "analyzing" && " Analysis cannot be cancelled once upload has finished; you can leave this page and find it in History."}
      </p>
      {onCancel && progress.kind === "uploading" && <button className="btn btn-quiet" onClick={onCancel}>Cancel upload</button>}
    </section>
  );
}
