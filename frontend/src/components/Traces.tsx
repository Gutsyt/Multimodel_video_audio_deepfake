import { useState } from "react";
import type { AudioSegment, FrameEvidence } from "../api";
import { formatScore } from "../lib/format";

interface Tip { x: number; y: number; text: string }
function useTip() {
  const [tip, setTip] = useState<Tip | null>(null);
  const show = (text: string) => (e: { clientX: number; clientY: number }) => setTip({ x: e.clientX + 12, y: e.clientY + 12, text });
  const at = (text: string) => (e: React.FocusEvent<Element>) => { const r = e.currentTarget.getBoundingClientRect(); setTip({ x: r.left, y: r.bottom + 6, text }); };
  return { tip, show, at, hide: () => setTip(null) };
}

const W = 640, L = 34, R = 8, T = 8;
const plot = (h: number) => ({ top: T, bottom: T + h });

function Axes({ h, threshold, duration }: { h: number; threshold: number; duration: number | null }) {
  const y = (v: number) => T + h - v * h;
  return (
    <g>
      {[0, 0.5, 1].map((v) => <g key={v}><line className="grid" x1={L} x2={W - R} y1={y(v)} y2={y(v)} /><text x={L - 6} y={y(v) + 3} textAnchor="end">{v}</text></g>)}
      <line className="thr" x1={L} x2={W - R} y1={y(threshold)} y2={y(threshold)} />
      {duration != null && <><text x={L} y={T + h + 40}>0 s</text><text x={W - R} y={T + h + 40} textAnchor="end">{duration.toFixed(1)} s</text></>}
    </g>
  );
}

/** One bar per ANALYZED audio window, placed where that window sits in time. */
export function AudioTrace({ segments, duration, threshold }: { segments: AudioSegment[]; duration: number | null; threshold: number }) {
  const { tip, show, at, hide } = useTip();
  const h = 112;
  const end = Math.max(duration ?? 0, ...segments.map((s) => s.end_s), 1);
  const x = (t: number) => L + (t / end) * (W - L - R);
  return (
    <figure style={{ margin: 0 }}>
      <svg className="trace" viewBox={`0 0 ${W} ${h + 48}`} role="img" aria-label={`Audio window scores: ${segments.map((s) => formatScore(s.score)).join(", ")}`}>
        <Axes h={h} threshold={threshold} duration={end} />
        {segments.map((s, i) => {
          const bh = Math.max(2, s.score * h);
          const text = `Window ${i + 1}: ${s.start_s.toFixed(1)}–${s.end_s.toFixed(1)} s · raw score ${formatScore(s.score)}`;
          return <rect key={i} className={`bar${s.score >= threshold ? " hot" : ""}`} x={x(s.start_s) + 1} width={Math.max(2, x(s.end_s) - x(s.start_s) - 2)} y={T + h - bh} height={bh} rx="3"
            tabIndex={0} aria-label={text} onMouseMove={show(text)} onMouseLeave={hide} onFocus={at(text)} onBlur={hide} />;
        })}
      </svg>
      {tip && <div className="tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </figure>
  );
}

/** One dot per sampled frame. Frames with no detected face are marked on their own row, never plotted as 0. */
export function FrameTrace({ frames, duration, threshold }: { frames: FrameEvidence[]; duration: number | null; threshold: number }) {
  const { tip, show, at, hide } = useTip();
  const h = 112;
  const timed = frames.every((f) => f.t_s != null);
  const end = Math.max(duration ?? 0, ...frames.map((f) => f.t_s ?? 0), 1);
  const x = (f: FrameEvidence, i: number) => (timed ? L + ((f.t_s ?? 0) / end) * (W - L - R - 10) + 5 : L + ((i + 0.5) / frames.length) * (W - L - R));
  const y = (v: number) => T + h - v * h;
  const row = plot(h).bottom + 12;
  return (
    <figure style={{ margin: 0 }}>
      <svg className="trace" viewBox={`0 0 ${W} ${h + 52}`} role="img" aria-label={`Visual frame scores: ${frames.filter((f) => f.score != null).map((f) => formatScore(f.score)).join(", ") || "none"}`}>
        <Axes h={h} threshold={threshold} duration={timed ? end : null} />
        <text x={L - 6} y={row + 4} textAnchor="end">no face</text>
        {frames.map((f, i) => {
          const label = f.t_s == null ? "Image" : `Frame at ${f.t_s.toFixed(1)} s`;
          if (f.score == null) {
            const text = `${label}: ${f.n_faces ? "face found" : "no face detected"}, not scored`;
            return <g key={i} tabIndex={0} aria-label={text} onMouseMove={show(text)} onMouseLeave={hide} onFocus={at(text)} onBlur={hide}>
              <path className="miss" d={`M${x(f, i) - 4} ${row - 3} l8 8 M${x(f, i) + 4} ${row - 3} l-8 8`} /></g>;
          }
          const text = `${label} · raw score ${formatScore(f.score)} · face covers ${f.face_area_ratio != null ? (f.face_area_ratio * 100).toFixed(0) + "% of the frame" : "n/a"}`;
          return <circle key={i} className={`dot${f.score >= threshold ? " hot" : ""}`} cx={x(f, i)} cy={y(f.score)} r="6" tabIndex={0} aria-label={text}
            onMouseMove={show(text)} onMouseLeave={hide} onFocus={at(text)} onBlur={hide} />;
        })}
      </svg>
      {tip && <div className="tip" style={{ left: tip.x, top: tip.y }}>{tip.text}</div>}
    </figure>
  );
}

/** Where the scored face sat in each sampled frame, drawn from the returned boxes (no pixels are stored). */
export function FaceMap({ frames, width, height, threshold }: { frames: FrameEvidence[]; width: number; height: number; threshold: number }) {
  const shown = frames.filter((f) => f.face_box);
  if (!shown.length || !width || !height) return null;
  return (
    <div className="facemap" aria-label="Detected face position per sampled frame">
      {shown.map((f, i) => (
        <figure key={i}>
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Face box ${f.t_s != null ? "at " + f.t_s.toFixed(1) + " s" : "in image"}`}>
            <rect x={f.face_box!.x} y={f.face_box!.y} width={f.face_box!.w} height={f.face_box!.h} fill="none" strokeWidth={Math.max(width, height) / 40}
              stroke={f.score == null ? "#8191b8" : f.score >= threshold ? "#f4a62a" : "#3b79ff"} />
          </svg>
          <span>{f.t_s != null ? `${f.t_s.toFixed(1)} s` : "image"}</span>
        </figure>
      ))}
    </div>
  );
}
