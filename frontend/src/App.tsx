import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiRequestError, deleteAnalysis, getAnalysis, getCapabilities, listAnalyses, reportUrl, uploadVideo, waitForAnalysis,
} from "./api";
import type { AnalysisResult, AnalysisSummary, ApiErrorBody, Capabilities } from "./api";
import { describeError } from "./format";
import { ErrorBanner } from "./components/Bits";
import { History } from "./components/History";
import { ProgressPanel } from "./components/ProgressPanel";
import type { Progress } from "./components/ProgressPanel";
import { ResultView } from "./components/ResultView";
import { UploadPanel } from "./components/UploadPanel";

type View =
  | { kind: "idle" }
  | { kind: "busy"; progress: Progress }
  | { kind: "done"; id: string; result: AnalysisResult }
  | { kind: "error"; error: ApiErrorBody };

const toBody = (e: unknown): ApiErrorBody =>
  e instanceof ApiRequestError ? e.body : { code: "internal_error", message: "Unexpected error." };
const isAbort = (e: unknown) => e instanceof DOMException && e.name === "AbortError";

export default function App() {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [capsError, setCapsError] = useState<ApiErrorBody | null>(null);
  const [view, setView] = useState<View>({ kind: "idle" });
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const abort = useRef<AbortController | null>(null);

  const refreshHistory = useCallback(() => { listAnalyses().then((r) => setHistory(r.items)).catch(() => undefined); }, []);

  useEffect(() => {
    let stop = false;
    const load = () =>
      getCapabilities()
        .then((c) => { if (!stop) { setCaps(c); setCapsError(null); if (c.models_state === "loading") setTimeout(load, 2000); } })
        .catch((e) => { if (!stop) setCapsError(toBody(e)); });
    load();
    refreshHistory();
    return () => { stop = true; };
  }, [refreshHistory]);

  async function analyze(file: File) {
    const ctl = new AbortController();
    abort.current = ctl;
    setView({ kind: "busy", progress: { kind: "uploading", loaded: 0, total: file.size } });
    try {
      const { id } = await uploadVideo(file, (loaded, total) => setView({ kind: "busy", progress: { kind: "uploading", loaded, total } }), ctl.signal);
      setView({ kind: "busy", progress: { kind: "analyzing", status: null } });
      const final = await waitForAnalysis(id, (status) => setView({ kind: "busy", progress: { kind: "analyzing", status } }), ctl.signal);
      if (final.status === "completed" && final.result) setView({ kind: "done", id, result: final.result });
      else setView({ kind: "error", error: final.error ?? { code: "internal_error", message: "Analysis failed." } });
    } catch (e) {
      setView(isAbort(e) ? { kind: "idle" } : { kind: "error", error: toBody(e) });
    } finally {
      abort.current = null;
      refreshHistory();
    }
  }

  async function open(id: string) {
    try {
      const s = await getAnalysis(id);
      if (s.result) setView({ kind: "done", id, result: s.result });
    } catch (e) { setView({ kind: "error", error: toBody(e) }); }
  }

  async function remove(id: string) {
    try { await deleteAnalysis(id); } catch (e) { setView({ kind: "error", error: toBody(e) }); return; }
    setView((v) => (v.kind === "done" && v.id === id ? { kind: "idle" } : v));
    refreshHistory();
  }

  const busy = view.kind === "busy";
  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>avdd <span className="muted">audio-visual deepfake analysis</span></h1>
          <p className="muted">Research prototype · pretrained-model inference only</p>
        </div>
        <ModelStatusChips caps={caps} />
      </header>

      <div className="banner banner-info" role="note">
        Scores are <strong>raw model outputs</strong>: not calibrated probabilities and not validated on any dataset by this project. No training or performance evaluation has been completed. Results are not forensic or legal proof.
      </div>
      {capsError && <ErrorBanner {...describeError(capsError)} onRetry={() => location.reload()} />}

      <main>
        {view.kind !== "done" && <UploadPanel caps={caps} disabled={busy || caps?.models_state !== "ready"} onAnalyze={analyze} />}
        {view.kind === "busy" && <ProgressPanel progress={view.progress} stages={caps?.stages ?? []} onCancel={() => abort.current?.abort()} />}
        {view.kind === "error" && <ErrorBanner {...describeError(view.error)} onRetry={() => setView({ kind: "idle" })} />}
        {view.kind === "done" && (
          <>
            <button className="btn" onClick={() => setView({ kind: "idle" })}>← Analyze another video</button>
            <ResultView result={view.result} analysisId={view.id} reportHref={reportUrl(view.id)} onDelete={() => remove(view.id)} />
          </>
        )}
        <History items={history} onOpen={open} onDelete={remove} />
      </main>

      <footer className="muted small">
        {caps?.retention} {caps?.evaluation_status}
      </footer>
    </div>
  );
}

function ModelStatusChips({ caps }: { caps: Capabilities | null }) {
  if (!caps) return <div className="chips muted small">Connecting to the server…</div>;
  return (
    <div className="chips" aria-label="Model status">
      {caps.models.map((m) => (
        <span key={m.role} className={`chip chip-${m.state}`} title={m.card ? `${m.card.name} (${m.card.version})` : undefined}>
          {m.role}: {m.state}{m.error_code ? ` (${m.error_code})` : ""}{m.card?.is_test_fixture ? " · TEST FIXTURE" : ""}
        </span>
      ))}
    </div>
  );
}
