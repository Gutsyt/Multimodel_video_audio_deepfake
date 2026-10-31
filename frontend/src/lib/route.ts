import { useEffect, useState } from "react";

export type Route = { name: "analyze" } | { name: "history" } | { name: "models" } | { name: "analysis"; id: string };

export function parseHash(hash: string): Route {
  const [a, b] = hash.replace(/^#\/?/, "").split("/");
  if (a === "analysis" && b && /^[0-9a-f]{32}$/.test(b)) return { name: "analysis", id: b };
  if (a === "history" || a === "models") return { name: a };
  return { name: "analyze" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));
  useEffect(() => {
    const on = () => { setRoute(parseHash(window.location.hash)); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}
