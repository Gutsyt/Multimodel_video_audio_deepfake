import type { ReactNode } from "react";
import type { Capabilities } from "../api";
import { hrefFor, type Route } from "../lib/route";
import { BrandMark, Icon, type IconName } from "./Icon";

const NAV: { route: Route; label: string; icon: IconName }[] = [
  { route: { name: "dashboard" }, label: "Dashboard", icon: "dashboard" },
  { route: { name: "analyze" }, label: "New analysis", icon: "analyze" },
  { route: { name: "history" }, label: "History", icon: "history" },
  { route: { name: "reports" }, label: "Reports", icon: "reports" },
  { route: { name: "system" }, label: "System", icon: "system" },
];
const activeName = (r: Route) => (r.name === "analysis" ? "history" : r.name);

export function ModelDots({ caps, error }: { caps: Capabilities | null; error: boolean }) {
  if (error) return <div className="row"><span className="pulse down" /> <span>Backend unreachable</span></div>;
  if (!caps) return <div className="row"><span className="pulse" /> <span>Checking backend…</span></div>;
  return (
    <>
      {caps.models.map((m) => (
        <div className="row" key={m.role} style={{ gap: 8 }}>
          <span className={`pulse ${m.state === "loaded" ? "up" : m.state === "failed" ? "down" : ""}`} />
          <span style={{ textTransform: "capitalize" }}>{m.role}</span>
          <span className="faint">{m.state === "failed" ? `failed (${m.error_code ?? "error"})` : m.state}</span>
        </div>
      ))}
    </>
  );
}

export function Shell({ route, caps, capsError, children }: { route: Route; caps: Capabilities | null; capsError: boolean; children: ReactNode }) {
  return (
    <div className="shell">
      <aside className="side">
        <a className="brand" href="#/" aria-label="AVDD home">
          <BrandMark />
          <div><b>AVDD</b><span>Audio-Visual Deepfake Detection</span></div>
        </a>
        <nav className="nav" aria-label="Main">
          {NAV.map((n) => (
            <a key={n.route.name} href={hrefFor(n.route)} aria-current={activeName(route) === n.route.name ? "page" : undefined}>
              <Icon name={n.icon} /> {n.label}
            </a>
          ))}
        </nav>
        <div className="health" aria-label="Backend status">
          <b style={{ fontSize: "0.78rem" }} className="muted">Models (live check)</b>
          <ModelDots caps={caps} error={capsError} />
        </div>
      </aside>
      <main className="main" id="main">
        <a className="topbrand brand" href="#/" aria-label="AVDD home"><BrandMark size={28} /><div><b>AVDD</b></div></a>
        {children}
      </main>
    </div>
  );
}

export function PageHead({ title, lead, actions }: { title: string; lead?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="pagehead">
      <div><h1>{title}</h1>{lead ? <p>{lead}</p> : null}</div>
      {actions ? <div className="row noprint">{actions}</div> : null}
    </div>
  );
}
