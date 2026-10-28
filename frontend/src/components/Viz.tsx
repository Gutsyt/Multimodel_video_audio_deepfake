import { animate, motion, useMotionValue, useTransform } from "motion/react";
import { useEffect, useState } from "react";

const EASE = [0.22, 1, 0.36, 1] as const;

/** Number that counts up to its value. */
export function CountUp({ value, suffix = "%" }: { value: number; suffix?: string }) {
  const mv = useMotionValue(0);
  const [shown, setShown] = useState(0);
  useEffect(() => {
    const c = animate(mv, value, { duration: 1.1, ease: EASE });
    const u = mv.on("change", (v) => setShown(Math.round(v)));
    return () => { c.stop(); u(); };
  }, [mv, value]);
  return <>{shown}{suffix}</>;
}

/** Ring that fills to ``score`` with tick marks at the two decision thresholds. */
export function Ring({ score, tReal, tFake, size = 200 }: { score: number | null; tReal?: number; tFake?: number; size?: number }) {
  const r = 84, c = 2 * Math.PI * r;
  const p = useMotionValue(0);
  const dash = useTransform(p, (v) => `${v * c} ${c}`);
  useEffect(() => { const a = animate(p, score ?? 0, { duration: 1.3, ease: EASE, delay: 0.15 }); return () => a.stop(); }, [p, score]);
  const tick = (t: number) => {
    const a = -Math.PI / 2 + t * 2 * Math.PI;
    return <line x1={100 + 74 * Math.cos(a)} y1={100 + 74 * Math.sin(a)} x2={100 + 95 * Math.cos(a)} y2={100 + 95 * Math.sin(a)} stroke="var(--ink)" strokeWidth="1.5" strokeLinecap="round" opacity="0.5" />;
  };
  return (
    <svg width={size} height={size} viewBox="0 0 200 200" role="img" aria-label={score == null ? "No score" : `Strongest manipulation probability ${Math.round(score * 100)} percent`}>
      <circle cx="100" cy="100" r={r} fill="none" stroke="var(--sunk)" strokeWidth="14" />
      <motion.circle cx="100" cy="100" r={r} fill="none" stroke="var(--c)" strokeWidth="14" strokeLinecap="round" style={{ strokeDasharray: dash }} transform="rotate(-90 100 100)" />
      {tReal != null && tick(tReal)}
      {tFake != null && tick(tFake)}
      <text x="100" y="104" textAnchor="middle" style={{ font: "400 52px var(--serif)", fill: "var(--ink)" }}>{score == null ? "–" : <CountUp value={Math.round(score * 100)} suffix="" />}</text>
      <text x="100" y="132" textAnchor="middle" style={{ font: "500 10.5px var(--mono)", fill: "var(--ink-3)", letterSpacing: "0.1em" }}>{score == null ? "NO SCORE" : "% FAKE"}</text>
    </svg>
  );
}

/** Horizontal bar split into authentic / inconclusive / manipulated zones with an animated score marker. */
export function ZoneBar({ score, tReal, tFake }: { score: number; tReal: number; tFake: number }) {
  return (
    <div>
      <div className="zones" aria-hidden="true">
        <span style={{ width: `${tReal * 100}%`, background: "var(--real-bg)" }} />
        <span style={{ width: `${(tFake - tReal) * 100}%`, background: "var(--mid-bg)" }} />
        <span style={{ width: `${(1 - tFake) * 100}%`, background: "var(--fake-bg)" }} />
        <motion.i className="marker" initial={{ left: "0%" }} animate={{ left: `${score * 100}%` }} transition={{ duration: 1.1, ease: EASE, delay: 0.25 }} />
      </div>
      <div className="zone-labels" style={{ marginTop: 8 }}><span>genuine</span><span>borderline</span><span>manipulated</span></div>
    </div>
  );
}

/** Score over time (frames or audio windows) with the two thresholds drawn as bands. */
export function Timeline({ points, tReal, tFake, label, bars = false }: { points: { t: number; s: number }[]; tReal: number; tFake: number; label: string; bars?: boolean }) {
  const W = 520, H = 150, P = 26;
  const tMax = Math.max(1e-6, ...points.map((p) => p.t));
  const x = (t: number) => P + (points.length < 2 && !bars ? (W - 2 * P) / 2 : (t / tMax) * (W - 2 * P));
  const y = (s: number) => H - P - s * (H - 2 * P);
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.s).toFixed(1)}`).join(" ");
  const slot = (W - 2 * P) / Math.max(points.length, 1), bw = Math.min(28, Math.max(6, slot - 8));
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      <rect x={P} y={y(1)} width={W - 2 * P} height={y(tFake) - y(1)} fill="var(--fake-bg)" />
      <rect x={P} y={y(tReal)} width={W - 2 * P} height={y(0) - y(tReal)} fill="var(--real-bg)" />
      {[0, 0.5, 1].map((v) => <text key={v} x={P - 6} y={y(v) + 3} textAnchor="end">{v * 100}</text>)}
      <line x1={P} x2={W - P} y1={y(0)} y2={y(0)} stroke="var(--line-2)" />
      {bars ? points.map((p, i) => (
        <motion.rect key={i} x={P + i * slot + (slot - bw) / 2} width={bw} rx="4" fill={p.s >= tFake ? "var(--fake)" : p.s <= tReal ? "var(--real)" : "var(--mid)"} initial={{ y: y(0), height: 0 }}
          animate={{ y: y(p.s), height: y(0) - y(p.s) }} transition={{ duration: 0.7, delay: 0.3 + i * 0.05, ease: EASE }} />
      )) : (
        <>
          <motion.path d={d} fill="none" stroke="var(--ink)" strokeWidth="2" strokeLinejoin="round" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.2, ease: EASE, delay: 0.2 }} />
          {points.map((p, i) => <motion.circle key={i} cx={x(p.t)} cy={y(p.s)} r="3.5" fill="var(--surface)" stroke="var(--ink)" strokeWidth="1.8" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.3 + i * 0.06 }} />)}
        </>
      )}
      <text x={W - P} y={H - 6} textAnchor="end">{bars ? "window" : `${tMax.toFixed(1)} s`}</text>
    </svg>
  );
}
