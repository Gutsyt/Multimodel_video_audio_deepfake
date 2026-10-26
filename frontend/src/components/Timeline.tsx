import type { AudioSegment, FrameEvidence } from "../api";
import { formatScore } from "../format";

const tint = (score: number) => `rgba(37, 99, 235, ${0.15 + 0.85 * score})`;

/** Time axis with the real per-segment / per-frame evidence. Blue intensity = raw score magnitude. */
export function Timeline({ duration, segments, frames }: {
  duration: number; segments?: AudioSegment[] | null; frames?: FrameEvidence[] | null;
}) {
  const pct = (t: number) => `${Math.min(100, Math.max(0, (t / duration) * 100))}%`;
  return (
    <div className="timeline">
      {segments && segments.length > 0 && (
        <div className="track" aria-label="Audio window scores over time">
          {segments.map((s) => (
            <div key={s.start_s} className="seg" title={`${s.start_s.toFixed(1)}–${s.end_s.toFixed(1)} s: raw score ${formatScore(s.score)}`}
              style={{ left: pct(s.start_s), width: `${Math.max(1, ((s.end_s - s.start_s) / duration) * 100)}%`, background: tint(s.score) }} />
          ))}
        </div>
      )}
      {frames && frames.length > 0 && (
        <div className="track" aria-label="Sampled frame scores over time">
          {frames.map((f) => (
            <div key={f.frame_index}
              className={f.score === null ? "tick tick-noface" : "tick"}
              title={f.score === null ? `${f.t_s.toFixed(1)} s: no face found, frame not scored` : `${f.t_s.toFixed(1)} s: raw score ${formatScore(f.score)}`}
              style={{ left: pct(f.t_s), background: f.score === null ? undefined : tint(f.score) }} />
          ))}
        </div>
      )}
      <div className="axis"><span>0 s</span><span>{duration.toFixed(1)} s</span></div>
    </div>
  );
}
