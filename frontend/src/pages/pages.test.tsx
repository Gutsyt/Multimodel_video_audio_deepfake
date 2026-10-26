// Page-level tests with a MOCKED backend (fetch is stubbed with fixture data, not model output).
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppDataProvider } from "../AppData";
import { ToastProvider } from "../components/Toasts";
import { caps, summary } from "../testFixtures";
import { filterHistory, HistoryPage } from "./HistoryPage";
import { DashboardPage } from "./DashboardPage";
import { AnalyzePage } from "./AnalyzePage";

type Item = ReturnType<typeof summary>;
function mockBackend(items: Item[]) {
  const state = { items: [...items], deleted: [] as string[] };
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    if (url.endsWith("/api/capabilities")) return new Response(JSON.stringify(caps()));
    if (url.endsWith("/api/analyses") && method === "GET") return new Response(JSON.stringify({ items: state.items }));
    if (url.endsWith("/api/analyses") && method === "DELETE") { const n = state.items.length; state.items = []; return new Response(JSON.stringify({ deleted: n })); }
    if (method === "DELETE") { const id = url.split("/").pop()!; state.deleted.push(id); state.items = state.items.filter((i) => i.id !== id); return new Response(null, { status: 204 }); }
    return new Response("{}", { status: 404 });
  }));
  return state;
}
const wrap = (ui: React.ReactNode) => render(<ToastProvider><AppDataProvider>{ui}</AppDataProvider></ToastProvider>);
beforeEach(() => vi.spyOn(window, "confirm").mockReturnValue(true));
afterEach(() => vi.restoreAllMocks());

const A = "a".repeat(32), B = "b".repeat(32), C = "c".repeat(32);
const data = [
  summary(A),
  summary(B, { media_kind: "audio", basis: "audio_only", assessment_status: "not_flagged", audio_decision: "not_flagged", visual_decision: null, fusion_state: "single_modality", fused_score: null, visual_status: "missing", visual_score: null, audio_score: 0.2 }),
  summary(C, { media_kind: "image", basis: "none", assessment_status: "insufficient_evidence", fusion_state: "insufficient_evidence", fused_score: null, audio_score: null, visual_score: null, audio_status: "missing", visual_status: "unreliable" }),
];

describe("history", () => {
  it("filters by outcome, media kind and free text (pure function)", () => {
    expect(filterHistory(data, "", "flagged", "all", "new").map((s) => s.id)).toEqual([A]);
    expect(filterHistory(data, "", "all", "image", "new").map((s) => s.id)).toEqual([C]);
    expect(filterHistory(data, "bbbb", "all", "all", "new").map((s) => s.id)).toEqual([B]);
    expect(filterHistory(data, "", "all", "all", "old").map((s) => s.id)).toEqual([C, B, A]);
  });
  it("lists records with status labels and shows a fused score only for multimodal ones", async () => {
    mockBackend(data);
    wrap(<HistoryPage />);
    const rows = await screen.findAllByRole("row");
    expect(rows).toHaveLength(4);
    expect(within(rows[1]).getByText("Flagged by the model")).toBeInTheDocument();
    expect(within(rows[2]).getByText("Not flagged by the model")).toBeInTheDocument();
    expect(within(rows[3]).getByText("Insufficient evidence")).toBeInTheDocument();
    expect(within(rows[2]).getAllByText("n/a").length).toBeGreaterThan(0); // no fused score for single-branch
    expect(within(rows[1]).getByText("0.70")).toBeInTheDocument();
  });
  it("search narrows the table and shows an empty-result row", async () => {
    mockBackend(data);
    wrap(<HistoryPage />);
    await screen.findAllByRole("row");
    fireEvent.change(screen.getByLabelText("Search history"), { target: { value: "zzzz" } });
    expect(screen.getByText("No records match these filters.")).toBeInTheDocument();
  });
  it("deletes one record through the API after confirmation", async () => {
    const be = mockBackend(data);
    wrap(<HistoryPage />);
    fireEvent.click(await screen.findByLabelText(`Delete ${B.slice(0, 8)}`));
    await waitFor(() => expect(be.deleted).toEqual([B]));
    await waitFor(() => expect(screen.queryByLabelText(`Delete ${B.slice(0, 8)}`)).toBeNull());
  });
  it("states what is stored and that media is not retained", async () => {
    mockBackend([]);
    wrap(<HistoryPage />);
    expect(await screen.findByText(/Never stored: the media file and its file name/)).toBeInTheDocument();
    expect(await screen.findByText("No records stored")).toBeInTheDocument();
  });
});

describe("dashboard", () => {
  it("shows an honest empty state on a fresh install (no invented statistics)", async () => {
    mockBackend([]);
    wrap(<DashboardPage />);
    expect(await screen.findByText("No analyses yet")).toBeInTheDocument();
    expect(screen.getByText(/Research prototype/)).toBeInTheDocument();
  });
  it("derives counts from stored records only", async () => {
    mockBackend(data);
    wrap(<DashboardPage />);
    const bar = await screen.findByRole("img", { name: /1 Flagged, 1 Not flagged, 1 Insufficient evidence, 0 Failed/ });
    expect(bar).toBeInTheDocument();
  });
  it("model health comes from the capabilities endpoint", async () => {
    mockBackend(data);
    vi.stubGlobal("fetch", vi.fn(async (url: string) => url.endsWith("/api/capabilities")
      ? new Response(JSON.stringify(caps({ models: [{ role: "audio", state: "failed", error_code: "weights_incompatible", card: null }] })))
      : new Response(JSON.stringify({ items: [] }))));
    wrap(<DashboardPage />);
    expect(await screen.findByText(/failed to load \(weights_incompatible\)/)).toBeInTheDocument();
  });
  it("says the backend is unreachable instead of claiming models are online", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("down")));
    wrap(<DashboardPage />);
    expect(await screen.findByText(/Backend unreachable/)).toBeInTheDocument();
  });
});

describe("analyze page validation", () => {
  it("rejects an unsupported file before upload and offers to replace it", async () => {
    mockBackend([]);
    const { container } = wrap(<AnalyzePage />);
    await screen.findByText(/Up to 1 MB/);
    const input = container.querySelector("input[type=file]") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["hello"], "notes.txt", { type: "text/plain" })] } });
    expect(await screen.findByText("This file cannot be analyzed")).toBeInTheDocument();
    expect(screen.getByText(/Unsupported file type/)).toBeInTheDocument();
    expect(screen.queryByText("Run analysis")).toBeNull();
  });
  it("requires consent before the analysis can start", async () => {
    mockBackend([]);
    const { container } = wrap(<AnalyzePage />);
    await screen.findByText(/Up to 1 MB/);
    fireEvent.change(container.querySelector("input[type=file]") as HTMLInputElement, { target: { files: [new File(["x"], "a.wav", { type: "audio/wav" })] } });
    const run = await screen.findByText("Run analysis");
    expect(run).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox"));
    await waitFor(() => expect(run).not.toBeDisabled());
  });
  it("lists accepted formats from the backend capabilities", async () => {
    mockBackend([]);
    wrap(<AnalyzePage />);
    expect(await screen.findByText(".mp4")).toBeInTheDocument();
    expect(screen.getByText(".wav")).toBeInTheDocument();
    expect(screen.getByText(".png")).toBeInTheDocument();
  });
});
