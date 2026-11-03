import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";
import type { Capabilities } from "../api";
import type { Route } from "../lib/route";
import { BrandMark } from "./Icon";

const NAV = [{ href: "#/", label: "Analyze", match: ["analyze", "analysis"] }, { href: "#/history", label: "History", match: ["history"] }, { href: "#/models", label: "Models", match: ["models"] }];

function ModelChip({ caps, capsError }: { caps: Capabilities | null; capsError: boolean }) {
  const [cls, text] = capsError ? ["bad", "Backend offline"] : !caps || caps.models_state === "loading" ? ["wait", "Loading models"]
    : caps.models_state === "ready" ? ["ok", "Models ready"] : ["bad", "Models unavailable"];
  return <span className="chip" role="status"><span className={`dot ${cls}`} />{text}</span>;
}

export function Shell({ route, caps, capsError, children }: { route: Route; caps: Capabilities | null; capsError: boolean; children: ReactNode }) {
  const key = route.name === "analysis" ? route.id : route.name;
  return (
    <>
      <header className="top">
        <div className="top-in">
          <a className="brand" href="#/"><BrandMark /> <span>AVDD <em>lens</em></span></a>
          <nav className="nav" aria-label="Main">
            {NAV.map((n) => {
              const on = n.match.includes(route.name);
              return (
                <a key={n.href} href={n.href} aria-current={on ? "page" : undefined}>
                  {on && <motion.span layoutId="navpill" className="pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
                  {n.label}
                </a>
              );
            })}
          </nav>
          <ModelChip caps={caps} capsError={capsError} />
        </div>
      </header>
      <main>
        <AnimatePresence mode="wait">
          <motion.div key={key} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}>
            {children}
          </motion.div>
        </AnimatePresence>
      </main>
      <footer className="foot">Results are model evidence, not forensic proof. Media is deleted right after analysis; only the result is stored locally.</footer>
    </>
  );
}

/** Staggered reveal for a list of children. */
export const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.07, delayChildren: 0.05 } } };
export const rise = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] as const } } };
