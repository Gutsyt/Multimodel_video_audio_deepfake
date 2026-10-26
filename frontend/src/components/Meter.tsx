import { formatScore } from "../lib/format";

/** A raw score on its 0-1 scale with the threshold marked. Never shows a percentage. */
export function Meter({ score, threshold, label, emptyText }: { score: number | null; threshold: number; label: string; emptyText?: string }) {
  const hot = score != null && score >= threshold;
  const W = 300;
  const x = (v: number) => 4 + v * (W - 8);
  return (
    <div className="meter">
      <div className="cap">
        <span>{label}</span>
        <span className="mono" style={{ color: score == null ? undefined : "var(--ink)" }}>{score == null ? emptyText ?? "no score" : formatScore(score)}</span>
      </div>
      <svg viewBox={`0 0 ${W} 40`} role="img" aria-label={score == null ? `${label}: no score` : `${label}: ${formatScore(score)} on a 0 to 1 scale, threshold ${threshold}`}>
        <defs>
          <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="var(--raised)" /><line x1="0" y1="0" x2="0" y2="6" stroke="var(--line)" strokeWidth="2" /></pattern>
        </defs>
        <rect x="4" y="12" width={W - 8} height="10" rx="5" className={score == null ? "hatch-track" : undefined} fill={score == null ? undefined : "var(--line-soft)"} />
        {score != null && <rect x="4" y="12" width={Math.max(10, x(score) - 4)} height="10" rx="5" fill={hot ? "var(--flag)" : "var(--signal)"} />}
        {score != null && <circle cx={x(score)} cy="17" r="7" fill="var(--surface)" stroke={hot ? "var(--flag)" : "var(--signal)"} strokeWidth="3" />}
        <line x1={x(threshold)} x2={x(threshold)} y1="6" y2="28" stroke="var(--flag-hi)" strokeWidth="1.5" strokeDasharray="2 2" />
        <text x="4" y="38">0</text>
        <text x={x(threshold)} y="38" textAnchor="middle">{threshold} threshold (unvalidated)</text>
        <text x={W - 4} y="38" textAnchor="end">1</text>
      </svg>
    </div>
  );
}
