import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { getAnalysis, type AnalysisResult, type FrameEvidence } from "../api";
import { useAppData } from "../AppData";
import { VERDICT_LABEL, VERDICT_SHORT, formatWhen } from "../lib/format";
import { clock, zoneOf, type Th } from "./Evidence";

const EASE = [0.16, 1, 0.3, 1] as const;

/* ---------------------------------------------------------------- filmstrip of the scored face crops */
type Crop = { t: number | null; score: number; src: string };

async function cutCrops(url: string, kind: "video" | "image", frames: FrameEvidence[]): Promise<Crop[]> {
  const scored = frames.filter((f) => f.face_box && f.score != null);
  const c = document.createElement("canvas");
  c.width = c.height = 168;
  const g = c.getContext("2d")!;
  const crop = (src: CanvasImageSource, f: FrameEvidence): Crop => {
    const b = f.face_box!, side = Math.max(b.w, b.h) * 1.6;
    g.drawImage(src, b.x + b.w / 2 - side / 2, b.y + b.h / 2 - side / 2, side, side, 0, 0, 168, 168);
    return { t: f.t_s, score: f.score!, src: c.toDataURL("image/jpeg", 0.82) };
  };
  if (kind === "image") {
    const img = new Image();
    img.src = url;
    await img.decode();
    return scored.map((f) => crop(img, f));
  }
  const v = document.createElement("video");
  v.muted = true; v.preload = "auto"; v.src = url;
  await new Promise((ok, no) => { v.onloadeddata = ok; v.onerror = no; });
  const out: Crop[] = [];
  for (const f of scored) {
    v.currentTime = f.t_s ?? 0;
    await new Promise((ok) => { v.onseeked = ok; });
    out.push(crop(v, f));
  }
  v.removeAttribute("src"); v.load();
  return out;
}

export function Filmstrip({ url, kind, frames, tv, onSeek }: { url: string; kind: "video" | "image"; frames: FrameEvidence[]; tv: Th; onSeek: (t: number) => void }) {
  const [crops, setCrops] = useState<Crop[] | null>(null);
  const n = frames.filter((f) => f.face_box && f.score != null).length;
  useEffect(() => { cutCrops(url, kind, frames).then(setCrops).catch(() => setCrops([])); }, [url, kind, frames]);
  if (!n || crops?.length === 0) return null;
  return (
    <div className="filmstrip" role="list" aria-label="Scored face crops">
      {(crops ?? Array.from({ length: n }, () => null)).map((c, i) => (
        <motion.button key={i} role="listitem" className={`crop ${c ? zoneOf(c.score, tv) : "loading"}`} disabled={!c || c.t == null}
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, ease: EASE, delay: i * 0.025 }}
          onClick={() => c?.t != null && onSeek(c.t)} aria-label={c ? `Face at ${c.t == null ? "image" : clock(c.t)}, ${Math.round(c.score * 100)}% fake` : "Loading face crop"}>
          {c && <img src={c.src} alt="" />}
          {c && <span className="crop-meta" title={c.t == null ? undefined : clock(c.t)}><b>{Math.round(c.score * 100)}</b></span>}
        </motion.button>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- compare with another analysis */
export function Compare({ id, r }: { id: string; r: AnalysisResult }) {
  const { history } = useAppData();
  const others = history.filter((h) => h.id !== id && h.status === "completed" && h.verdict);
  const [other, setOther] = useState("");
  const [b, setB] = useState<AnalysisResult | null>(null);
  useEffect(() => {
    if (!other) { setB(null); return; }
    let stop = false;
    getAnalysis(other).then((s) => { if (!stop) setB(s.result); }).catch(() => setB(null));
    return () => { stop = true; };
  }, [other]);
  if (!others.length) return null;
  const rows = [["visual", "Face"], ["audio", "Voice"]] as const;
  return (
    <section className="panel compare" aria-label="Compare with another analysis">
      <div className="compare-head">
        <h2>Compare</h2>
        <select value={other} onChange={(e) => setOther(e.target.value)} aria-label="Analysis to compare with">
          <option value="">Pick an earlier analysis…</option>
          {others.map((h) => <option key={h.id} value={h.id}>{VERDICT_SHORT[h.verdict!]} · {h.media_kind} · {formatWhen(h.created_utc)}</option>)}
        </select>
      </div>
      <AnimatePresence>
        {b && (
          <motion.div className="compare-body" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.3, ease: EASE }}>
            <p className="compare-key"><span className="key a" />This file: <b>{VERDICT_LABEL[r.assessment.verdict]}</b><span className="key b" />Other: <b>{VERDICT_LABEL[b.assessment.verdict]}</b></p>
            {rows.map(([k, label]) => {
              const sa = r.modalities[k].score, sb = b.modalities[k].score, t = r.assessment.thresholds[k];
              return (
                <div className="cmp-row" key={k}>
                  <span className="scale-label">{label}</span>
                  <div className="cmp-track">
                    <i className="z real" style={{ width: `${t.t_real * 100}%` }} />
                    <i className="z mid" style={{ left: `${t.t_real * 100}%`, width: `${(t.t_fake - t.t_real) * 100}%` }} />
                    <i className="z fake" style={{ left: `${t.t_fake * 100}%`, width: `${(1 - t.t_fake) * 100}%` }} />
                    {sa != null && sb != null && <span className="cmp-gap" style={{ left: `${Math.min(sa, sb) * 100}%`, width: `${Math.abs(sa - sb) * 100}%` }} />}
                    {sa != null && <motion.span className="dot-a" initial={{ left: 0 }} animate={{ left: `${sa * 100}%` }} transition={{ duration: 0.7, ease: EASE }} />}
                    {sb != null && <motion.span className="dot-b" initial={{ left: 0 }} animate={{ left: `${sb * 100}%` }} transition={{ duration: 0.7, ease: EASE, delay: 0.08 }} />}
                  </div>
                  <span className="cmp-delta mono">{sa != null && sb != null ? `${sa >= sb ? "+" : "−"}${Math.round(Math.abs(sa - sb) * 100)} pts` : "–"}</span>
                </div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </section>
  );
}

/* ---------------------------------------------------------------- share card (no media, no face) */
const COL = { real: "#1f7a4d", mid: "#a86d12", fake: "#d2462b", none: "#6b6f78" } as const;

export async function shareCard(r: AnalysisResult): Promise<"shared" | "saved"> {
  await document.fonts.ready;
  const W = 1200, H = 630, c = document.createElement("canvas");
  c.width = W; c.height = H;
  const g = c.getContext("2d")!;
  const a = r.assessment;
  const vc = a.verdict === "likely_authentic" ? COL.real : a.verdict === "inconclusive" ? COL.mid : a.verdict === "likely_manipulated" ? COL.fake : COL.none;
  g.fillStyle = "#f7f5f0"; g.fillRect(0, 0, W, H);
  g.fillStyle = "#15171c"; g.font = "600 26px 'Inter Tight Variable'"; g.fillText("AVDD lens", 80, 92);
  g.fillStyle = vc; g.font = "400 120px 'Instrument Serif'"; g.fillText(VERDICT_LABEL[a.verdict], 76, 250);
  g.fillStyle = "#4a4d55"; g.font = "400 26px 'Inter Tight Variable'"; g.fillText(formatWhen(r.created_utc), 80, 300);
  ([["visual", "Face"], ["audio", "Voice"]] as const).forEach(([k, label], i) => {
    const y = 380 + i * 92, x0 = 220, w = 900, t = a.thresholds[k], s = r.modalities[k].score;
    g.fillStyle = "#15171c"; g.font = "600 28px 'Inter Tight Variable'"; g.fillText(label, 80, y + 34);
    [[0, t.t_real, "#e4f1e8"], [t.t_real, t.t_fake, "#f8ecd5"], [t.t_fake, 1, "#fbe4dd"]].forEach(([p, q, col]) => {
      g.fillStyle = col as string; g.fillRect(x0 + (p as number) * w, y, ((q as number) - (p as number)) * w, 48);
    });
    if (s == null) { g.fillStyle = "#85888f"; g.font = "400 22px 'Inter Tight Variable'"; g.fillText("not analysed", x0 + 20, y + 32); return; }
    const z = zoneOf(s, t), px = x0 + s * w;
    g.fillStyle = COL[z]; g.beginPath(); g.roundRect(px - 34, y + 4, 68, 40, 8); g.fill();
    g.fillStyle = "#fff"; g.font = "600 24px 'JetBrains Mono Variable'"; g.textAlign = "center"; g.fillText(String(Math.round(s * 100)), px, y + 33); g.textAlign = "left";
  });
  g.fillStyle = "#85888f"; g.font = "400 20px 'Inter Tight Variable'"; g.fillText("Model evidence, not forensic proof.", 80, 590);
  const blob = await new Promise<Blob>((ok) => c.toBlob((b) => ok(b!), "image/png"));
  const file = new File([blob], "avdd-verdict.png", { type: "image/png" });
  if (navigator.canShare?.({ files: [file] })) { await navigator.share({ files: [file], title: VERDICT_LABEL[a.verdict] }); return "shared"; }
  const u = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = u; link.download = file.name; link.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
  return "saved";
}
