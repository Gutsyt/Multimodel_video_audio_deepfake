import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { getCapabilities, listAnalyses, type AnalysisSummary, type Capabilities } from "./api";

interface AppData {
  caps: Capabilities | null; capsError: boolean; recheck: () => void;
  history: AnalysisSummary[]; historyLoaded: boolean; historyError: boolean; refreshHistory: () => Promise<void>;
}
const Ctx = createContext<AppData | null>(null);
export function useAppData(): AppData {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAppData outside AppDataProvider");
  return v;
}

/** Live backend state: capabilities (model health, limits) and stored history. Nothing is cached as "online". */
export function AppDataProvider({ children }: { children: ReactNode }) {
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [capsError, setCapsError] = useState(false);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const [historyError, setHistoryError] = useState(false);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      try {
        const c = await getCapabilities();
        if (stop) return;
        setCaps(c); setCapsError(false);
        if (c.models_state === "loading") timer = setTimeout(load, 2000);
      } catch {
        if (!stop) { setCapsError(true); setCaps(null); }
      }
    };
    void load();
    return () => { stop = true; if (timer) clearTimeout(timer); };
  }, [tick]);

  const refreshHistory = useCallback(async () => {
    try {
      setHistory((await listAnalyses()).items);
      setHistoryError(false);
    } catch {
      setHistoryError(true);
    } finally {
      setHistoryLoaded(true);
    }
  }, []);
  useEffect(() => { void refreshHistory(); }, [refreshHistory]);

  const value = useMemo(() => ({ caps, capsError, recheck: () => setTick((t) => t + 1), history, historyLoaded, historyError, refreshHistory }),
    [caps, capsError, history, historyLoaded, historyError, refreshHistory]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export type RunTone = "flagged" | "not_flagged" | "insufficient_evidence" | "failed";
/** Overall state of a stored record, from the summary only (older records fall back to fusion fields). */
export function toneOf(s: AnalysisSummary): RunTone | "running" {
  if (s.status === "failed") return "failed";
  if (s.status !== "completed") return "running";
  if (s.assessment_status === "flagged" || s.assessment_status === "not_flagged" || s.assessment_status === "insufficient_evidence") return s.assessment_status;
  if (s.fusion_state === "insufficient_evidence") return "insufficient_evidence";
  return s.flagged ? "flagged" : "not_flagged";
}
