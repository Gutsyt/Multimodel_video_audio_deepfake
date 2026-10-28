import { motion } from "motion/react";
import type { ModelStatus, SetMetrics } from "../api";
import { useAppData } from "../AppData";
import { CountUp } from "../components/Viz";
import { rise, stagger } from "../components/Shell";
import { pct } from "../lib/format";

const TEST_LABEL: Record<string, string> = {
  in_domain_test: "Held-out test (unseen videos / speakers)",
  wilddeepfake_test: "WildDeepfake test (internet deepfakes)",
  celebdf_test: "Celeb-DF test frames",
};

export function ModelsPage() {
  const { caps } = useAppData();
  return (
    <motion.div style={{ display: "grid", gap: 28 }} variants={stagger} initial="hidden" animate="show">
      <motion.div variants={rise} style={{ display: "grid", gap: 10 }}>
        <span className="eyebrow">How it decides</span>
        <h1 className="h2" style={{ fontSize: 44 }}>Two independent checks, measured.</h1>
        <p className="lead">Each check is a large pretrained network that stays frozen, plus a small classifier trained on labelled real and fake media. Every number below was measured on data the classifier never saw.</p>
      </motion.div>
      <motion.ol variants={rise} className="pipeline" style={{ padding: 0, margin: 0 }}>
        <li><b>Read</b>Up to 12 frames and the speech are taken from the first 60 s.</li>
        <li><b>Find</b>YuNet locates the face in each frame; speech is cut into 4 s windows.</li>
        <li><b>Score</b>CLIP ViT-L/14 and XLS-R features go through calibrated classifiers.</li>
        <li><b>Decide</b>Median per check vs. two thresholds: below → genuine, above → manipulated, between → inconclusive.</li>
      </motion.ol>
      <div className="grid2">
        {[...(caps?.models ?? [])].sort((a, b) => a.role.localeCompare(b.role)).reverse().map((m) => <ModelCard key={m.role} m={m} />)}
      </div>
    </motion.div>
  );
}

function ModelCard({ m }: { m: ModelStatus }) {
  const c = m.card, met = c?.metrics;
  const head: SetMetrics | undefined = met ? Object.values(met.tests)[0] : undefined;
  return (
    <motion.section variants={rise} className="card pad" style={{ display: "grid", gap: 20, alignContent: "start" }}>
      <div style={{ display: "grid", gap: 6 }}>
        <span className="eyebrow">{m.role === "visual" ? "Face check" : "Voice check"} · {m.state}</span>
        <h2 style={{ font: "600 19px var(--sans)" }}>{c?.name ?? "Not loaded"}</h2>
        {c && <p className="faint" style={{ fontSize: 13 }}>Trained on {c.trained_on}</p>}
      </div>
      {head && met && (
        <>
          <div className="metrics">
            <div className="metric"><span className="v"><CountUp value={Math.round(head.auc * 100)} suffix="" /></span><span className="faint" style={{ fontSize: 12.5 }}>AUC × 100</span></div>
            <div className="metric"><span className="v"><CountUp value={Math.round((head.genuine_flagged ?? 0) * 1000) / 10} /></span><span className="faint" style={{ fontSize: 12.5 }}>genuine flagged</span></div>
            <div className="metric"><span className="v"><CountUp value={Math.round((head.fake_flagged ?? 0) * 100)} /></span><span className="faint" style={{ fontSize: 12.5 }}>fakes flagged</span></div>
          </div>
          <table className="t">
            <thead><tr><th>Test set</th><th className="n">AUC</th><th className="n">EER</th><th className="n">Real flagged</th><th className="n">Fake caught</th></tr></thead>
            <tbody>
              {Object.entries(met.tests).map(([k, s]) => (
                <tr key={k}><td>{TEST_LABEL[k] ?? k}<div className="faint" style={{ fontSize: 12 }}>{s.n_real} real · {s.n_fake} fake</div></td>
                  <td className="n">{s.auc.toFixed(3)}</td><td className="n">{pct(s.eer)}</td><td className="n">{pct(s.genuine_flagged)}</td><td className="n">{pct(s.fake_flagged)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="faint" style={{ fontSize: 12.5 }}>Thresholds: genuine ≤ {pct(met.t_real)}, manipulated ≥ {pct(met.t_fake)}.</p>
        </>
      )}
      {c && <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13.5, color: "var(--ink-2)", display: "grid", gap: 4 }}>{c.limitations.map((l) => <li key={l}>{l}</li>)}</ul>}
      {m.error_code && <p className="alert">Not loaded: {m.error_code}</p>}
    </motion.section>
  );
}
