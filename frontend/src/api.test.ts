// MOCKED transport: fetch and XMLHttpRequest are replaced, so these test the client's handling of
// success and failure responses. See e2e.test.ts for the client against a real running backend.
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiRequestError, deleteAllAnalyses, getCapabilities, uploadMedia, waitForAnalysis } from "./api";
import { caps } from "./testFixtures";

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("returns parsed data on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(caps())));
    expect((await getCapabilities()).models_state).toBe("ready");
  });
  it("turns a structured backend error into ApiRequestError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ error: { code: "queue_full", message: "busy" } }, 429)));
    await expect(deleteAllAnalyses()).rejects.toMatchObject({ status: 429, body: { code: "queue_full" } });
  });
  it("falls back to http_error for a non-JSON failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("<html>bad gateway</html>", { status: 502 })));
    await expect(getCapabilities()).rejects.toMatchObject({ status: 502, body: { code: "http_error" } });
  });
  it("reports an unreachable server as network_error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("failed")));
    await expect(getCapabilities()).rejects.toMatchObject({ status: 0, body: { code: "network_error" } });
  });
  it("reports a request timeout as timeout", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new DOMException("t", "TimeoutError")));
    await expect(getCapabilities()).rejects.toMatchObject({ body: { code: "timeout" } });
  });
  it("polls until the job finishes and tolerates brief network failures", async () => {
    const st = (status: string) => json({ id: "a".repeat(32), status, stage: status === "processing" ? "audio" : null, stages: [], created_utc: "", updated_utc: "", result: null, error: null });
    const f = vi.fn().mockRejectedValueOnce(new TypeError("blip")).mockResolvedValueOnce(st("queued")).mockResolvedValueOnce(st("processing")).mockResolvedValueOnce(st("completed"));
    vi.stubGlobal("fetch", f);
    const seen: string[] = [];
    const final = await waitForAnalysis("a".repeat(32), (s) => seen.push(s.status), undefined, 1);
    expect(final.status).toBe("completed");
    expect(seen).toEqual(["queued", "processing", "completed"]);
  });
  it("gives up polling after the maximum wait", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(json({ id: "a", status: "queued", stage: null, stages: [], created_utc: "", updated_utc: "", result: null, error: null }))));
    await expect(waitForAnalysis("a".repeat(32), () => undefined, undefined, 1, 30)).rejects.toBeInstanceOf(ApiRequestError);
  });
});

describe("uploadMedia", () => {
  class FakeXhr {
    static status = 202; static body = '{"id":"b","status":"queued"}';
    upload: { onprogress: ((e: { lengthComputable: boolean; loaded: number; total: number }) => void) | null } = { onprogress: null };
    onload: (() => void) | null = null; onerror: (() => void) | null = null; onabort: (() => void) | null = null; ontimeout: (() => void) | null = null;
    status = FakeXhr.status; responseText = FakeXhr.body; timeout = 0;
    open() {} setRequestHeader() {} abort() { this.onabort?.(); }
    send() { this.upload.onprogress?.({ lengthComputable: true, loaded: 5, total: 10 }); this.onload?.(); }
  }
  it("reports real progress and resolves with the job id", async () => {
    vi.stubGlobal("XMLHttpRequest", FakeXhr);
    FakeXhr.status = 202; FakeXhr.body = '{"id":"b","status":"queued"}';
    const seen: number[] = [];
    const r = await uploadMedia(new Blob(["x"]), (l) => seen.push(l));
    expect(r.id).toBe("b"); expect(seen).toEqual([5]);
  });
  it("rejects with the backend's validation error", async () => {
    vi.stubGlobal("XMLHttpRequest", FakeXhr);
    FakeXhr.status = 415; FakeXhr.body = '{"error":{"code":"unsupported_format","message":"nope"}}';
    await expect(uploadMedia(new Blob(["x"]), () => undefined)).rejects.toMatchObject({ status: 415, body: { code: "unsupported_format" } });
  });
});
