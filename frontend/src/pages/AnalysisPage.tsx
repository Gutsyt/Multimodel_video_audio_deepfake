import { useEffect, useState } from "react";
import { ApiRequestError, deleteAnalysis, getAnalysis, reportUrl, waitForAnalysis, type AnalysisStatus, type ApiErrorBody } from "../api";
import { useAppData } from "../AppData";
import { Icon } from "../components/Icon";
import { ResultView } from "../components/ResultView";
import { PageHead } from "../components/Shell";
import { useToast } from "../components/Toasts";
import { describeError, formatWhen } from "../lib/format";

export function AnalysisPage({ id }: { id: string }) {
  const [st, setSt] = useState<AnalysisStatus | null>(null);
  const [err, setErr] = useState<ApiErrorBody | null>(null);
  const { refreshHistory } = useAppData();
  const toast = useToast();

  useEffect(() => {
    const ctl = new AbortController();
    setSt(null); setErr(null);
    getAnalysis(id).then((s) => {
      setSt(s);
      if (s.status === "queued" || s.status === "processing") waitForAnalysis(id, setSt, ctl.signal).then(setSt).catch(() => undefined);
    }).catch((e: unknown) => setErr(e instanceof ApiRequestError ? e.body : { code: "network_error", message: "" }));
    return () => ctl.abort();
  }, [id]);

  async function remove() {
    if (!window.confirm("Delete this analysis record? This cannot be undone.")) return;
    try { await deleteAnalysis(id); await refreshHistory(); toast.push("Record deleted.", "success"); window.location.hash = "#/history"; }
    catch (e) { toast.push(e instanceof ApiRequestError ? describeError(e.body) : "Delete failed.", "error"); }
  }

  const done = st?.status === "completed" && st.result;
  return (
    <div className="stack">
      <PageHead title={done ? "Analysis report" : "Analysis"} lead={st ? `Record ${id.slice(0, 8)} · ${formatWhen(st.created_utc)}` : undefined}
        actions={done ? <>
          <a className="btn" href={reportUrl(id)} download><Icon name="download" /> Download JSON</a>
          <button className="btn" onClick={() => window.print()}><Icon name="print" /> Print / save as PDF</button>
          <button className="btn danger" onClick={() => void remove()}><Icon name="trash" /> Delete</button></> : undefined} />
      {err && <div className="errbox" role="alert"><b>Could not load this analysis</b><span>{describeError(err)}</span><div className="row"><a className="btn" href="#/history">Back to history</a></div></div>}
      {!err && !st && <p className="muted" role="status">Loading…</p>}
      {st && (st.status === "queued" || st.status === "processing") && <p role="status" className="muted">This analysis is {st.status}{st.stage ? ` (${st.stage})` : ""}. The page updates when it finishes.</p>}
      {st?.status === "failed" && st.error && <div className="errbox" role="alert"><b>Analysis failed</b><span>{describeError(st.error)}</span><small className="faint">Code: {st.error.code}</small>
        <div className="row"><a className="btn primary" href="#/analyze">Try another file</a><button className="btn danger" onClick={() => void remove()}>Delete record</button></div></div>}
      {done && <ResultView result={st.result!} />}
    </div>
  );
}
