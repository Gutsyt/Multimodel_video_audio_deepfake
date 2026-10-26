import type { AnalysisSummary } from "../api";
import { toneOf, type RunTone } from "../AppData";
import { formatScore } from "../lib/format";

const TONES: RunTone[] = ["flagged", "not_flagged", "insufficient_evidence", "failed"];
export const TONE_TEXT: Record<RunTone, string> = { flagged: "Flagged", not_flagged: "Not flagged", insufficient_evidence: "Insufficient evidence", failed: "Failed" };

export function countsOf(items: AnalysisSummary[]): Record<RunTone, number> {
  const c: Record<RunTone, number> = { flagged: 0, not_flagged: 0, insufficient_evidence: 0, failed: 0 };
  for (const s of items) { const t = toneOf(s); if (t !== "running") c[t]++; }
  return c;
}

/** One square per stored analysis, newest first. The picture is the count. */
export function RunStrip({ items }: { items: AnalysisSummary[] }) {
  return (
    <div className="runstrip" role="list" aria-label="Stored analyses, newest first">
      {items.slice(0, 60).map((s) => {
        const t = toneOf(s);
        return <a role="listitem" key={s.id} href={`#/analysis/${s.id}`} className={`t-${t}`} title={t === "running" ? "Running" : TONE_TEXT[t]} aria-label={`Analysis ${s.id.slice(0, 8)}: ${t === "running" ? "running" : TONE_TEXT[t]}`} />;
      })}
    </div>
  );
}

export function StatusMix({ items }: { items: AnalysisSummary[] }) {
  const c = countsOf(items);
  const total = TONES.reduce((a, t) => a + c[t], 0);
  return (
    <div style={{ display: "grid", gap: 10 }}>
      <div className="segbar" role="img" aria-label={TONES.map((t) => `${c[t]} ${TONE_TEXT[t]}`).join(", ")}>
        {TONES.filter((t) => c[t]).map((t) => <i key={t} className={`s-${t}`} style={{ flex: c[t] }} />)}
      </div>
      <div className="legend">
        {TONES.map((t) => <span key={t}><i className={t === "flagged" ? "k-flag" : t === "not_flagged" ? "k-steel" : t === "failed" ? "k-fail" : "k-hatch"} />{TONE_TEXT[t]} <b className="mono" style={{ color: "var(--ink)" }}>{c[t]}</b></span>)}
        <span className="faint">of {total}</span>
      </div>
    </div>
  );
}

const CELL: Record<string, string> = { available: "c-available", unreliable: "c-unreliable", missing: "c-missing", corrupted: "c-corrupted", not_analyzed: "c-not_analyzed" };
/** Which branches ran for each recent record: filled = analyzed, hatched = unreliable, dashed = absent. */
export function CoverageMatrix({ items }: { items: AnalysisSummary[] }) {
  const done = items.filter((s) => s.status === "completed").slice(0, 24);
  const row = (label: string, pick: (s: AnalysisSummary) => string | null) => (
    <div style={{ display: "grid", gap: 6 }}>
      <span className="faint" style={{ fontSize: "0.8rem" }}>{label}</span>
      <div className="cover" role="img" aria-label={`${label} status for the last ${done.length} analyses`}>
        {done.map((s) => <i key={s.id} className={CELL[pick(s) ?? "missing"]} title={`${label}: ${pick(s) ?? "n/a"}`} />)}
      </div>
    </div>
  );
  return (
    <div style={{ display: "grid", gap: 12 }}>
      {row("Audio branch", (s) => s.audio_status)}
      {row("Visual branch", (s) => s.visual_status)}
      <div className="legend"><span><i className="k-sig" />analyzed</span><span><i className="k-hatch" />unreliable</span><span><i className="k-steel" style={{ borderStyle: "dashed" }} />not in file</span><span><i className="k-fail" />failed to decode</span></div>
    </div>
  );
}

/** Raw scores per stored run. Different marks per branch; the fused score appears only for runs that fused two branches. */
export function RunDots({ items, threshold = 0.5 }: { items: AnalysisSummary[]; threshold?: number }) {
  const runs = items.filter((s) => s.status === "completed").slice(0, 24).reverse();
  if (!runs.length) return null;
  const W = 640, H = 150, L = 30, R = 8, T = 8, B = 20;
  const x = (i: number) => L + ((i + 0.5) / runs.length) * (W - L - R);
  const y = (v: number) => T + (1 - v) * (H - T - B);
  return (
    <div style={{ display: "grid", gap: 10 }}>
      <svg className="trace" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Raw scores for the last ${runs.length} analyses, oldest to newest`}>
        {[0, 0.5, 1].map((v) => <g key={v}><line className="grid" x1={L} x2={W - R} y1={y(v)} y2={y(v)} /><text x={L - 6} y={y(v) + 3} textAnchor="end">{v}</text></g>)}
        <line className="thr" x1={L} x2={W - R} y1={y(threshold)} y2={y(threshold)} />
        {runs.map((s, i) => (
          <g key={s.id}>
            {s.audio_score != null && <circle cx={x(i) - 5} cy={y(s.audio_score)} r="4.5" fill="var(--signal)"><title>{`Audio ${formatScore(s.audio_score)}`}</title></circle>}
            {s.visual_score != null && <rect x={x(i) + 1} y={y(s.visual_score) - 4} width="9" height="9" rx="1.5" fill="var(--signal-hi)"><title>{`Visual ${formatScore(s.visual_score)}`}</title></rect>}
            {s.fusion_state === "multimodal" && s.fused_score != null && <circle cx={x(i)} cy={y(s.fused_score)} r="7" fill="none" stroke="var(--flag)" strokeWidth="2"><title>{`Fused ${formatScore(s.fused_score)}`}</title></circle>}
          </g>
        ))}
        <text x={L} y={H - 4}>oldest</text><text x={W - R} y={H - 4} textAnchor="end">newest</text>
      </svg>
      <div className="legend"><span><i className="k-sig" style={{ borderRadius: 5 }} />audio branch</span><span><i style={{ background: "var(--signal-hi)" }} />visual branch</span><span><i style={{ border: "2px solid var(--flag)", borderRadius: 5 }} />fused (two branches only)</span><span className="faint">raw scores, not probabilities</span></div>
    </div>
  );
}
