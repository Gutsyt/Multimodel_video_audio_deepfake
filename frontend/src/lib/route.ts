import { useEffect, useState } from "react";

export type Route =
  | { name: "dashboard" } | { name: "analyze" } | { name: "history" } | { name: "reports" } | { name: "system" }
  | { name: "analysis"; id: string };

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#\/?/, "");
  const [a, b] = path.split("/");
  if (a === "analysis" && b && /^[0-9a-f]{32}$/.test(b)) return { name: "analysis", id: b };
  if (a === "analyze" || a === "history" || a === "reports" || a === "system") return { name: a };
  return { name: "dashboard" };
}
export const hrefFor = (r: Route): string => (r.name === "dashboard" ? "#/" : r.name === "analysis" ? `#/analysis/${r.id}` : `#/${r.name}`);

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
