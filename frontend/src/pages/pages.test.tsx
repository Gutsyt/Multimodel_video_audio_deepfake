import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { absent, audioOk, caps, make, summary, visualOk } from "../testFixtures";

const json = (b: unknown) => new Response(JSON.stringify(b), { status: 200 });
function serve(result: unknown, items = [summary("a".repeat(32))]) {
  vi.stubGlobal("fetch", vi.fn((url: string) => Promise.resolve(json(
    url.includes("/capabilities") ? caps() : url.endsWith("/api/analyses") ? { items } : { id: "a".repeat(32), status: "completed", stages: [], created_utc: "", updated_utc: "", result, error: null },
  ))));
}
afterEach(() => { vi.restoreAllMocks(); window.location.hash = ""; });

describe("pages", () => {
  it("shows the verdict from the backend and both checks", async () => {
    serve(make(audioOk(0.1), visualOk(0.85)));
    window.location.hash = `#/analysis/${"a".repeat(32)}`;
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Likely manipulated" })).toBeTruthy();
    expect(screen.getByRole("region", { name: "Face check" }).textContent).toMatch(/85%/);
    expect(screen.getByRole("region", { name: "Voice check" }).textContent).toMatch(/Likely authentic/);
  });
  it("never invents a score for a missing stream", async () => {
    serve(make(absent("missing", ["no audio track in this file"]), visualOk(0.1)));
    window.location.hash = `#/analysis/${"a".repeat(32)}`;
    render(<App />);
    const voice = await screen.findByRole("region", { name: "Voice check" });
    expect(voice.textContent).toMatch(/Not in this file/);
    expect(voice.textContent).not.toMatch(/%/);
  });
  it("lists history with verdicts and shows measured model metrics", async () => {
    serve(null, [summary("b".repeat(32), { verdict: "inconclusive" })]);
    window.location.hash = "#/history";
    const { unmount } = render(<App />);
    expect(await screen.findByText("Inconclusive", { selector: "b" })).toBeTruthy();
    unmount();
    window.location.hash = "#/models";
    render(<App />);
    expect(await screen.findByText(/0.930/)).toBeTruthy();
  });
});
