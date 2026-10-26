import type { AnalysisResult, ModalityResult } from "../api";
import { STATUS_LABEL, formatScore, formatTime } from "../lib/format";

function nodeClass(m: ModalityResult | undefined): string {
  if (!m) return "off";
  return m.status === "available" ? "ok" : m.status === "corrupted" || m.status === "not_analyzed" ? "bad" : "off";
}

/** The four pipeline stages with the status each really ended in. */
export function PipelineFlow({ result }: { result: AnalysisResult }) {
  const { audio, visual } = result.modalities;
  const f = result.fusion;
  const kind = result.media.kind ?? (result.media.has_video ? "video" : "audio");
  const fusionText = f.state === "multimodal" ? "Fused (unvalidated rule)" : f.state === "single_modality" ? "Not performed: one branch" : "Not performed: no evidence";
  const stages = [
    { name: "Read media", cls: result.outcome === "analyzed" ? "ok" : "bad", text: `${kind}${result.media.duration_s ? ", " + formatTime(result.media.duration_s) : ""}` },
    { name: "Audio branch", cls: nodeClass(audio), text: audio ? (audio.reasons[0] && audio.status !== "available" ? `${STATUS_LABEL[audio.status]}: ${audio.reasons[0]}` : STATUS_LABEL[audio.status]) : "n/a" },
    { name: "Visual branch", cls: nodeClass(visual), text: visual ? (visual.reasons[0] && visual.status !== "available" ? `${STATUS_LABEL[visual.status]}: ${visual.reasons[0]}` : STATUS_LABEL[visual.status]) : "n/a" },
    { name: "Fusion", cls: f.state === "multimodal" ? "ok" : "off", text: fusionText },
  ];
  return (
    <div className="flow" role="list" aria-label="Analysis pipeline">
      {stages.map((s) => (
        <div className="gap" role="listitem" key={s.name}>
          <div className={`node ${s.cls}`}><b>{s.name}</b><span>{s.text}</span><i className="link" /></div>
        </div>
      ))}
    </div>
  );
}

/** Wire thickness is the weight actually used. With one branch there is no fusion node, only a dashed stub. */
export function FusionDiagram({ result }: { result: AnalysisResult }) {
  const f = result.fusion;
  const { audio, visual } = result.modalities;
  const used = (m?: ModalityResult) => m?.status === "available";
  const w = (k: string) => f.weights_used[k] ?? 0;
  const fused = f.state === "multimodal";
  const wire = (y: number, weight: number, on: boolean) => <path className={`wire${on ? "" : " off"}`} d={`M190 ${y} C 280 ${y}, 290 100, 380 100`} strokeWidth={on ? 2 + weight * 12 : 1.5} />;
  return (
    <svg className="fusion" viewBox="0 0 560 200" role="img" aria-label={fused ? "Audio and visual scores fused into one score" : f.state === "single_modality" ? "One branch usable: no fusion performed" : "No usable branch: no fusion performed"}>
      <rect className={`box ${used(audio) ? "on" : "off"}`} x="10" y="30" width="180" height="40" rx="10" />
      <text x="24" y="48">Audio branch</text><text className="sub" x="24" y="62">{used(audio) ? `raw ${formatScore(audio!.score)}` : STATUS_LABEL[audio?.status ?? "missing"].toLowerCase()}</text>
      <rect className={`box ${used(visual) ? "on" : "off"}`} x="10" y="130" width="180" height="40" rx="10" />
      <text x="24" y="148">Visual branch</text><text className="sub" x="24" y="162">{used(visual) ? `raw ${formatScore(visual!.score)}` : STATUS_LABEL[visual?.status ?? "missing"].toLowerCase()}</text>
      {wire(50, w("audio"), fused && used(audio))}
      {wire(150, w("visual"), fused && used(visual))}
      <rect className={`box ${fused ? "on" : "off"}`} x="380" y="70" width="170" height="60" rx="12" />
      {fused ? (
        <><text x="396" y="94">Fused score</text><text className="sub" x="396" y="112">raw {formatScore(f.fused_score)} · w {w("audio").toFixed(2)}/{w("visual").toFixed(2)}</text></>
      ) : (
        <><text x="396" y="96">No fusion</text><text className="sub" x="396" y="113">{f.state === "single_modality" ? "single-branch result" : "no usable evidence"}</text></>
      )}
    </svg>
  );
}
