import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { ApiRequestError, deleteAllAnalyses, deleteAnalysis, type Verdict } from "../api";
import { useAppData } from "../AppData";
import { Icon } from "../components/Icon";
import { useToast } from "../components/Toasts";
import { VERDICT_SHORT, describeError, formatTime, formatWhen, pct } from "../lib/format";

const FILTERS: (Verdict | "all")[] = ["all", "likely_authentic", "inconclusive", "likely_manipulated"];

export function HistoryPage() {
  const { history, historyLoaded, refreshHistory } = useAppData();
  const toast = useToast();
  const [f, setF] = useState<Verdict | "all">("all");
  const rows = history.filter((h) => f === "all" || h.verdict === f);

  async function act(p: Promise<unknown>, ok: string) {
    try { await p; await refreshHistory(); toast.push(ok); }
    catch (e) { toast.push(e instanceof ApiRequestError ? describeError(e.body) : "Failed.", "error"); }
  }

  return (
    <div style={{ display: "grid", gap: 24 }}>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "end" }}>
        <h1 className="h2" style={{ fontSize: 44 }}>History</h1>
        {history.length > 0 && <button className="btn ghost danger small" onClick={() => { if (confirm("Delete all finished analyses?")) void act(deleteAllAnalyses(), "History cleared."); }}><Icon name="trash" size={14} /> Clear all</button>}
      </div>
      <div className="filters" role="group" aria-label="Filter by verdict">
        {FILTERS.map((k) => <button key={k} aria-pressed={f === k} onClick={() => setF(k)}>{k === "all" ? `All · ${history.length}` : VERDICT_SHORT[k]}</button>)}
      </div>
      {historyLoaded && rows.length === 0 ? (
        <div className="card empty"><Icon name="history" size={28} /><p className="muted">{history.length ? "Nothing matches this filter." : "No analyses yet."}</p><a className="btn primary" href="#/">Analyze a file</a></div>
      ) : (
        <motion.div className="list" layout>
          <AnimatePresence initial={true}>
            {rows.map((h, i) => {
              const v = h.status === "failed" ? "none" : h.verdict ?? "none";
              return (
                <motion.a layout key={h.id} href={`#/analysis/${h.id}`} className={`card item t-${v}`}
                  initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0, transition: { delay: Math.min(i, 10) * 0.04 } }} exit={{ opacity: 0, x: -30 }}>
                  <span className="ico"><Icon name={(h.media_kind as "video" | "audio" | "image") ?? "file"} size={20} /></span>
                  <span style={{ display: "grid", gap: 3, minWidth: 0 }}>
                    <b style={{ fontWeight: 600 }}>{h.status === "failed" ? "Failed" : h.status !== "completed" ? "Running…" : h.verdict ? VERDICT_SHORT[h.verdict] : "Old format"}</b>
                    <span className="faint" style={{ fontSize: 13 }}>{formatWhen(h.created_utc)}{h.duration_s ? ` · ${formatTime(h.duration_s)}` : ""}</span>
                  </span>
                  <span className="mini" title={`Strongest: ${pct(h.score)}`}><motion.i initial={{ width: 0 }} animate={{ width: `${(h.score ?? 0) * 100}%` }} transition={{ duration: 0.8, delay: 0.2 }} /></span>
                  <button className="btn ghost small" aria-label="Delete" onClick={(e) => { e.preventDefault(); void act(deleteAnalysis(h.id), "Deleted."); }}><Icon name="trash" size={15} /></button>
                </motion.a>
              );
            })}
          </AnimatePresence>
        </motion.div>
      )}
    </div>
  );
}
