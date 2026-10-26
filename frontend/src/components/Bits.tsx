import type { ModalityStatus } from "../api";
import { STATUS_LABEL, formatScore } from "../format";

export function StatusBadge({ status }: { status: ModalityStatus }) {
  return <span className={`badge badge-${status}`}>{STATUS_LABEL[status]}</span>;
}

/** Bar showing a raw score. Neutral blue only: colour never implies real/fake. */
export function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="scorebar">
      <div className="scorebar-label">
        <span>{label}</span>
        <strong>{formatScore(value)}</strong>
      </div>
      <div className="scorebar-track" role="img" aria-label={`${label}: ${formatScore(value)} on a 0 to 1 scale`}>
        <div className="scorebar-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

export function ErrorBanner({ title, detail, onRetry }: { title: string; detail?: string; onRetry?: () => void }) {
  return (
    <div className="banner banner-error" role="alert">
      <div>
        <strong>{title}</strong>
        {detail && <div className="muted small">{detail}</div>}
      </div>
      {onRetry && <button className="btn" onClick={onRetry}>Try again</button>}
    </div>
  );
}
