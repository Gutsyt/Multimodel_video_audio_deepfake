import { reportUrl } from "../api";
import { toneOf, useAppData } from "../AppData";
import { Icon } from "../components/Icon";
import { PageHead } from "../components/Shell";
import { StatusMark } from "../components/StatusMark";
import { KIND_LABEL, formatWhen } from "../lib/format";
import type { MediaKind } from "../api";

export function ReportsPage() {
  const { history, historyLoaded } = useAppData();
  const done = history.filter((h) => h.status === "completed");
  return (
    <div className="stack">
      <PageHead title="Reports" lead="A report is the stored result of one analysis, exactly as the backend returned it. JSON is the complete record. Open a report to print it or save it as PDF from your browser." />
      {historyLoaded && done.length === 0 && <div className="panel"><div className="empty"><Icon name="reports" size={32} /><h2>No reports yet</h2><p>Completed analyses can be exported here.</p><a className="btn primary" href="#/analyze">Analyze media</a></div></div>}
      <div className="grid2">
        {done.map((s) => { const t = toneOf(s); return (
          <section className="panel" key={s.id}>
            <header><h3>{s.media_kind ? KIND_LABEL[s.media_kind as MediaKind] ?? s.media_kind : "Media"} analysis</h3>{t !== "running" && <StatusMark tone={t} />}</header>
            <p className="muted">{formatWhen(s.created_utc)} · record <span className="mono">{s.id.slice(0, 8)}</span></p>
            <div className="row" style={{ marginTop: 14 }}>
              <a className="btn" href={reportUrl(s.id)} download><Icon name="download" size={16} /> JSON</a>
              <a className="btn" href={`#/analysis/${s.id}`}><Icon name="print" size={16} /> Open / print</a>
            </div>
          </section>); })}
      </div>
    </div>
  );
}
