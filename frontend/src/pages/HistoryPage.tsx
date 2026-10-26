import { useMemo, useState } from "react";
import { ApiRequestError, deleteAllAnalyses, deleteAnalysis, getAnalysis, type AnalysisSummary } from "../api";
import { toneOf, useAppData, type RunTone } from "../AppData";
import { TONE_TEXT } from "../components/Dashboard";
import { Icon } from "../components/Icon";
import { PageHead } from "../components/Shell";
import { StatusMark } from "../components/StatusMark";
import { useToast } from "../components/Toasts";
import { KIND_LABEL, describeError, formatScore, formatWhen } from "../lib/format";
import { downloadJson } from "../lib/download";
import type { MediaKind } from "../api";

export const RETENTION_NOTE = "Stored per analysis: scores, timestamps, the media's technical metadata and model diagnostics. Never stored: the media file and its file name (the upload is a temporary copy that the server deletes when the analysis ends).";

type Sort = "new" | "old";
export function filterHistory(items: AnalysisSummary[], q: string, tone: RunTone | "all", kind: MediaKind | "all", sort: Sort): AnalysisSummary[] {
  const needle = q.trim().toLowerCase();
  const out = items.filter((s) => {
    const t = toneOf(s);
    if (tone !== "all" && t !== tone) return false;
    if (kind !== "all" && s.media_kind !== kind) return false;
    if (!needle) return true;
    const hay = [s.id, s.status, t, TONE_TEXT[t as RunTone] ?? "", s.media_kind ?? "", s.created_utc, s.error_code ?? ""].join(" ").toLowerCase();
    return hay.includes(needle);
  });
  return sort === "old" ? [...out].reverse() : out;
}

export function HistoryPage() {
  const { history, historyLoaded, historyError, refreshHistory } = useAppData();
  const toast = useToast();
  const [q, setQ] = useState(""); const [tone, setTone] = useState<RunTone | "all">("all");
  const [kind, setKind] = useState<MediaKind | "all">("all"); const [sort, setSort] = useState<Sort>("new");
  const rows = useMemo(() => filterHistory(history, q, tone, kind, sort), [history, q, tone, kind, sort]);

  async function del(id: string) {
    if (!window.confirm("Delete this analysis record? This cannot be undone.")) return;
    try { await deleteAnalysis(id); await refreshHistory(); toast.push("Record deleted.", "success"); }
    catch (e) { toast.push(e instanceof ApiRequestError ? describeError(e.body) : "Delete failed.", "error"); }
  }
  async function clearAll() {
    if (!window.confirm("Delete ALL finished analysis records? This cannot be undone.")) return;
    try { const r = await deleteAllAnalyses(); await refreshHistory(); toast.push(`Deleted ${r.deleted} record(s).`, "success"); }
    catch (e) { toast.push(e instanceof ApiRequestError ? describeError(e.body) : "Delete failed.", "error"); }
  }
  async function exportOne(id: string) {
    try { const s = await getAnalysis(id); downloadJson(`avdd-analysis-${id.slice(0, 8)}.json`, { report_type: "avdd-analysis-report", analysis_id: id, result: s.result, error: s.error }); }
    catch (e) { toast.push(e instanceof ApiRequestError ? describeError(e.body) : "Export failed.", "error"); }
  }

  return (
    <div className="stack">
      <PageHead title="Analysis history" lead={RETENTION_NOTE}
        actions={history.length > 0 ? <button className="btn danger" onClick={() => void clearAll()}><Icon name="trash" size={16} /> Delete all finished</button> : undefined} />
      <div className="row" role="search">
        <input className="field" type="search" placeholder="Search id, outcome, media…" aria-label="Search history" value={q} onChange={(e) => setQ(e.target.value)} style={{ minWidth: 220 }} />
        <select className="field" aria-label="Filter by outcome" value={tone} onChange={(e) => setTone(e.target.value as RunTone | "all")}>
          <option value="all">All outcomes</option>{(Object.keys(TONE_TEXT) as RunTone[]).map((t) => <option key={t} value={t}>{TONE_TEXT[t]}</option>)}</select>
        <select className="field" aria-label="Filter by media kind" value={kind} onChange={(e) => setKind(e.target.value as MediaKind | "all")}>
          <option value="all">All media</option>{(["video", "audio", "image"] as MediaKind[]).map((k) => <option key={k} value={k}>{KIND_LABEL[k]}</option>)}</select>
        <select className="field" aria-label="Sort" value={sort} onChange={(e) => setSort(e.target.value as Sort)}><option value="new">Newest first</option><option value="old">Oldest first</option></select>
      </div>
      {historyError && <div className="errbox" role="alert"><b>History could not be loaded.</b><button className="btn" onClick={() => void refreshHistory()}>Retry</button></div>}
      {historyLoaded && history.length === 0 && !historyError && <div className="panel"><div className="empty"><Icon name="history" size={32} /><h2>No records stored</h2><p>Finished analyses appear here until you delete them.</p><a className="btn primary" href="#/analyze">Analyze media</a></div></div>}
      {history.length > 0 && (
        <div className="tablewrap"><table>
          <thead><tr><th>When</th><th>Media</th><th>Outcome</th><th>Basis</th><th>Audio</th><th>Visual</th><th>Fused</th><th><span className="sr">Actions</span></th></tr></thead>
          <tbody>
            {rows.map((s) => { const t = toneOf(s); return (
              <tr key={s.id}>
                <td>{formatWhen(s.created_utc)}<div className="faint mono" style={{ fontSize: "0.75rem" }}>{s.id.slice(0, 8)}{s.is_test_fixture ? " · test doubles" : ""}</div></td>
                <td>{s.media_kind ? KIND_LABEL[s.media_kind as MediaKind] ?? s.media_kind : "n/a"}</td>
                <td>{t === "running" ? <span className="status neutral">{s.status}</span> : <StatusMark tone={t} />}{s.error_code && <div className="faint mono" style={{ fontSize: "0.75rem" }}>{s.error_code}</div>}</td>
                <td className="muted">{s.basis ? s.basis.replace("_", " ") : "n/a"}</td>
                <td className="num">{formatScore(s.audio_score)}</td><td className="num">{formatScore(s.visual_score)}</td>
                <td className="num">{s.fusion_state === "multimodal" ? formatScore(s.fused_score) : "n/a"}</td>
                <td><div className="row" style={{ gap: 4, flexWrap: "nowrap" }}>
                  <a className="btn small" href={`#/analysis/${s.id}`}>Open</a>
                  {s.status === "completed" && <button className="btn small ghost" aria-label={`Export JSON for ${s.id.slice(0, 8)}`} onClick={() => void exportOne(s.id)}><Icon name="download" size={15} /></button>}
                  {(s.status === "completed" || s.status === "failed") && <button className="btn small ghost danger" aria-label={`Delete ${s.id.slice(0, 8)}`} onClick={() => void del(s.id)}><Icon name="trash" size={15} /></button>}
                </div></td>
              </tr>); })}
            {rows.length === 0 && <tr><td colSpan={8} className="muted" style={{ textAlign: "center", padding: 24 }}>No records match these filters.</td></tr>}
          </tbody>
        </table></div>
      )}
      <p className="faint">Fused scores exist only for analyses where two branches produced usable evidence. Scores are raw model outputs, not probabilities.</p>
    </div>
  );
}
