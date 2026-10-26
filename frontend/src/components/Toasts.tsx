import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { Icon } from "./Icon";

type Tone = "info" | "success" | "error";
interface Toast { id: number; tone: Tone; text: string }
interface ToastApi { push: (text: string, tone?: Tone) => void }
const Ctx = createContext<ToastApi>({ push: () => undefined });
export const useToast = (): ToastApi => useContext(Ctx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const push = useCallback((text: string, tone: Tone = "info") => {
    const id = Date.now() + Math.random();
    setItems((cur) => [...cur.slice(-3), { id, tone, text }]);
    setTimeout(() => setItems((cur) => cur.filter((t) => t.id !== id)), tone === "error" ? 9000 : 5000);
  }, []);
  const api = useMemo(() => ({ push }), [push]);
  return (
    <Ctx.Provider value={api}>
      {children}
      <div className="toasts" role="region" aria-label="Notifications" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className={`toast ${t.tone}`} role={t.tone === "error" ? "alert" : "status"}>
            <span>{t.text}</span>
            <button className="btn ghost small" aria-label="Dismiss notification" onClick={() => setItems((c) => c.filter((x) => x.id !== t.id))}><Icon name="close" size={14} /></button>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}
