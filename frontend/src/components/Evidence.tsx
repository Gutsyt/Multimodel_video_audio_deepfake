import { motion, useReducedMotion } from "motion/react";
import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import type { AudioSegment, FrameEvidence } from "../api";

export type Th = { t_real: number; t_fake: number };
export type Zone = "real" | "mid" | "fake";
export const zoneOf = (s: number, t: Th): Zone => (s >= t.t_fake ? "fake" : s <= t.t_real ? "real" : "mid");
export const clock = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const EASE = [0.16, 1, 0.3, 1] as const;

/* ---------------------------------------------------------------- evidence scale (focal) */
export interface ScaleRow { key: string; label: string; score: number | null; t: Th; note?: string }

export function EvidenceScale({ rows, onSettled }: { rows: ScaleRow[]; onSettled: () => void }) {
  const reduce = useReducedMotion();
  const [settled, setSettled] = useState(false);
  const live = rows.filter((r) => r.score != null).length;
  const done = useRef(0);
  const settle = () => { if (++done.current >= live) { setSettled(true); onSettled(); } };
  useEffect(() => { if (!live) { setSettled(true); onSettled(); } }, [live, onSettled]);
  return (
    <div className="scale" role="group" aria-label="Where each check landed against its thresholds">
      <div className="scale-axis" aria-hidden="true">{[0, 25, 50, 75, 100].map((v) => <span key={v} style={{ left: `${v}%` }}>{v}</span>)}</div>
      {rows.map((r) => {
        const z = r.score == null ? null : zoneOf(r.score, r.t);
        return (
          <div className="scale-row" key={r.key}>
            <span className="scale-label">{r.label}</span>
            <div className={`scale-track${settled ? " settled" : ""}`} data-zone={z ?? "none"}>
              <i className="z real" style={{ width: `${r.t.t_real * 100}%` }} />
              <i className="z mid" style={{ left: `${r.t.t_real * 100}%`, width: `${(r.t.t_fake - r.t.t_real) * 100}%` }} />
              <i className="z fake" style={{ left: `${r.t.t_fake * 100}%`, width: `${(1 - r.t.t_fake) * 100}%` }} />
              <span className="thr" style={{ left: `${r.t.t_real * 100}%` }}>{Math.round(r.t.t_real * 100)}</span>
              <span className="thr" style={{ left: `${r.t.t_fake * 100}%` }}>{Math.round(r.t.t_fake * 100)}</span>
              {r.score != null ? (
                <motion.span className={`pin ${z}`} initial={{ left: reduce ? `${r.score * 100}%` : "0%" }} animate={{ left: `${r.score * 100}%` }}
                  transition={{ duration: reduce ? 0 : 1, ease: EASE, delay: 0.15 }} onAnimationComplete={settle}>
                  <b>{Math.round(r.score * 100)}</b>
                </motion.span>
              ) : <span className="scale-none">{r.note}</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- waveform from the in-session file */
export function usePeaks(file: File | null, limit: number, n = 700): number[] | null {
  const [peaks, setPeaks] = useState<number[] | null>(null);
  useEffect(() => {
    if (!file || !("AudioContext" in window)) return;
    let stop = false;
    const ctx = new AudioContext();
    file.arrayBuffer().then((b) => ctx.decodeAudioData(b)).then((buf) => {
      const ch = buf.getChannelData(0), len = Math.min(ch.length, Math.floor(limit * buf.sampleRate)), step = Math.max(1, Math.floor(len / n));
      const out: number[] = [];
      for (let i = 0; i < len; i += step) { let m = 0; for (let j = i; j < Math.min(len, i + step); j++) m = Math.max(m, Math.abs(ch[j])); out.push(m); }
      const top = Math.max(1e-6, ...out);
      if (!stop) setPeaks(out.map((v) => v / top));
    }).catch(() => undefined).finally(() => void ctx.close());
    return () => { stop = true; };
  }, [file, limit, n]);
  return peaks;
}

/* ---------------------------------------------------------------- synced lanes */
interface LanesProps {
  dur: number; frames: FrameEvidence[]; segments: AudioSegment[]; tv: Th; ta: Th; peaks: number[] | null;
  playhead: number | null; onSeek: (t: number) => void;
}

export function Lanes({ dur, frames, segments, tv, ta, peaks, playhead, onSeek }: LanesProps) {
  const W = 1000, H = 84;
  const [hover, setHover] = useState<number | null>(null);
  const x = (t: number) => (t / dur) * W;
  const y = (s: number) => H - 6 - s * (H - 14);
  const at = hover ?? playhead;
  const face = useMemo(() => frames.filter((f) => f.t_s != null), [frames]);
  const nearFrame = at == null || !face.length ? null : face.reduce((a, b) => (Math.abs(b.t_s! - at) < Math.abs(a.t_s! - at) ? b : a));
  const win = at == null ? null : [...segments].reverse().find((s) => s.start_s <= at && at < s.end_s + 1e-6) ?? null;
  const move = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setHover(Math.max(0, Math.min(dur, ((e.clientX - r.left) / r.width) * dur)));
  };
  const cross = at == null ? null : <line x1={x(at)} x2={x(at)} y1="0" y2={H} className="cross" />;
  const thr = (t: Th) => <>{[t.t_real, t.t_fake].map((v) => <line key={v} x1="0" x2={W} y1={y(v)} y2={y(v)} className="thr-line" />)}</>;

  return (
    <div className="lanes">
      <div className="readout" aria-live="polite">
        <span className="mono">{at == null ? "Hover or play to inspect a moment" : clock(at)}</span>
        {at != null && <>
          <span>Face <b className={nearFrame?.score != null ? `zc ${zoneOf(nearFrame.score, tv)}` : ""}>{nearFrame?.score != null ? `${Math.round(nearFrame.score * 100)}%` : nearFrame ? "no face" : "–"}</b></span>
          <span>Voice <b className={win ? `zc ${zoneOf(win.score, ta)}` : ""}>{win ? `${Math.round(win.score * 100)}%` : "–"}</b></span>
        </>}
      </div>
      <div className="lane-grid" onPointerMove={move} onPointerLeave={() => setHover(null)} onClick={() => hover != null && onSeek(hover)}
        role="slider" tabIndex={0} aria-label="Timeline" aria-valuemin={0} aria-valuemax={dur} aria-valuenow={at ?? 0}
        onKeyDown={(e) => { if (e.key === "ArrowRight" || e.key === "ArrowLeft") onSeek(Math.max(0, Math.min(dur, (playhead ?? 0) + (e.key === "ArrowRight" ? 1 : -1)))); }}>
        <span className="lane-label">Face</span>
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="lane">
          {thr(tv)}
          {face.map((f, i) => f.score == null
            ? <circle key={i} cx={x(f.t_s!)} cy={H - 6} r="3" className="noface" />
            : <motion.line key={i} x1={x(f.t_s!)} x2={x(f.t_s!)} y2={H - 6} className={`stem ${zoneOf(f.score, tv)}`}
                initial={{ y1: H - 6 }} animate={{ y1: y(f.score) }} transition={{ duration: 0.6, ease: EASE, delay: 0.5 + i * 0.03 }} />)}
          {cross}
        </svg>
        <span className="lane-label">Voice</span>
        {segments.length || peaks ? (
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="lane">
          {peaks && <path className="wave" d={peaks.map((p, i) => `M${(i / peaks.length) * W},${H / 2 - p * (H / 2 - 4)}V${H / 2 + p * (H / 2 - 4)}`).join("")} />}
          {thr(ta)}
          {segments.map((s, i) => (
            <motion.rect key={i} x={x(s.start_s) + 1} width={Math.max(2, x(s.end_s) - x(s.start_s) - 2)} rx="3" className={`seg ${zoneOf(s.score, ta)}`}
              initial={{ y: H - 6, height: 0 }} animate={{ y: y(s.score), height: H - 6 - y(s.score) }} transition={{ duration: 0.6, ease: EASE, delay: 0.6 + i * 0.05 }} />
          ))}
          {cross}
        </svg>
        ) : <div className="lane-empty">No speech analysed</div>}
        <span />
        <div className="ticks mono">{[0, 0.25, 0.5, 0.75, 1].map((f) => <span key={f} style={{ left: `${f * 100}%` }}>{clock(f * dur)}</span>)}</div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- the real file, with the detected face box */
export function MediaStage({ url, kind, frames, tv, frameSize, videoRef, onTime }: {
  url: string; kind: "video" | "image"; frames: FrameEvidence[]; tv: Th; frameSize: [number, number] | null;
  videoRef: React.RefObject<HTMLVideoElement | null>; onTime: (t: number) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState<[number, number]>([0, 0]);
  const [t, setT] = useState(0);
  const [ready, setReady] = useState(kind === "image");
  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver(([e]) => setSize([e.contentRect.width, e.contentRect.height]));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let raf = 0;
    const tick = () => { setT(v.currentTime); onTime(v.currentTime); if (!v.paused) raf = requestAnimationFrame(tick); };
    const on = () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(tick); };
    v.addEventListener("play", on); v.addEventListener("seeked", on); v.addEventListener("timeupdate", on);
    return () => { cancelAnimationFrame(raf); v.removeEventListener("play", on); v.removeEventListener("seeked", on); v.removeEventListener("timeupdate", on); };
  }, [videoRef, onTime]);

  const scored = frames.filter((f) => f.face_box && f.score != null);
  const gap = scored.length > 1 ? (scored[scored.length - 1].t_s! - scored[0].t_s!) / (scored.length - 1) : Infinity;
  const f = kind === "image" ? scored[0] : scored.find((s) => Math.abs((s.t_s ?? 0) - t) <= gap / 2);
  let rect: { left: number; top: number; width: number; height: number } | null = null;
  if (f?.face_box && frameSize && size[0]) {
    const k = Math.min(size[0] / frameSize[0], size[1] / frameSize[1]);
    const ox = (size[0] - frameSize[0] * k) / 2, oy = (size[1] - frameSize[1] * k) / 2;
    rect = { left: ox + f.face_box.x * k, top: oy + f.face_box.y * k, width: f.face_box.w * k, height: f.face_box.h * k };
  }
  return (
    <div className="stage" ref={box}>
      {kind === "video" ? <video ref={videoRef} src={url} controls playsInline muted preload="auto" onLoadedData={() => setReady(true)} /> : <img src={url} alt="The analysed image" />}
      {ready && rect && f && (
        <motion.div className={`facebox ${zoneOf(f.score!, tv)}`} animate={rect} transition={{ duration: 0.18, ease: "easeOut" }}>
          <span>{Math.round(f.score! * 100)}%</span>
        </motion.div>
      )}
    </div>
  );
}
