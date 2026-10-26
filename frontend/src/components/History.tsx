import type { AnalysisSummary } from "../api";
import { formatScore, STATUS_LABEL } from "../format";
import type { ModalityStatus } from "../api";

const mod = (status: string | null, score: number | null) =>
  status === null ? "–" : status === "available" && score !== null ? formatScore(score) : STATUS_LABEL[status as ModalityStatus] ?? status;

export function History({ items, onOpen, onDelete }: { items: AnalysisSummary[]; onOpen: (id: string) => void; onDelete: (id: string) => void }) {
  return (
    <section className="card" aria-labelledby="h-history">
      <h2 id="h-history">History</h2>
      {items.length === 0 ? <p className="muted">No analyses stored yet. Results are kept locally only until you delete them; videos are never stored.</p> : (
        <div className="scroll">
          <table className="table">
            <thead><tr><th>When (UTC)</th><th>ID</th><th>Status</th><th>Combined</th><th>Audio</th><th>Visual</th><th /></tr></thead>
            <tbody>
              {items.map((i) => (
                <tr key={i.id}>
                  <td>{i.created_utc.replace("T", " ").replace("+00:00", "")}</td>
                  <td><code>{i.id.slice(0, 8)}</code>{i.is_test_fixture ? " (test fixture)" : ""}</td>
                  <td>{i.status}{i.error_code ? ` (${i.error_code})` : ""}</td>
                  <td>{i.fusion_state === "insufficient_evidence" ? "Insufficient evidence" : i.fused_score !== null ? `${formatScore(i.fused_score)} raw${i.flagged ? ", flagged" : ", not flagged"}` : "–"}</td>
                  <td>{mod(i.audio_status, i.audio_score)}</td>
                  <td>{mod(i.visual_status, i.visual_score)}</td>
                  <td className="nowrap">
                    {i.status === "completed" && <button className="btn btn-small" onClick={() => onOpen(i.id)}>View</button>}{" "}
                    {(i.status === "completed" || i.status === "failed") && <button className="btn btn-small btn-quiet" onClick={() => onDelete(i.id)}>Delete</button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
