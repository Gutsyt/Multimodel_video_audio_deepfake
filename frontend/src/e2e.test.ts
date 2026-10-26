// End-to-end wiring test: the REAL api.ts client against a RUNNING backend over real HTTP.
// Skipped unless AVDD_E2E_URL and AVDD_E2E_VIDEO are set (see docs/runbook_local.md). The backend
// decides what is real: AVDD_E2E_EXPECT=doubles means the server was started with test doubles.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ApiRequestError, deleteAnalysis, getCapabilities, listAnalyses, uploadVideo, waitForAnalysis } from "./api";
import type { AnalysisStatus } from "./api";

const url = process.env.AVDD_E2E_URL;
const video = process.env.AVDD_E2E_VIDEO;
const expectMode = process.env.AVDD_E2E_EXPECT ?? "real";

describe.skipIf(!url || !video)(`e2e against a running backend (${expectMode})`, () => {
  it("capabilities, upload, real stage polling, result, history, delete", async () => {
    const caps = await getCapabilities();
    expect(caps.models_state).toBe("ready");

    const blob = new Blob([readFileSync(video as string)]);
    let progressEvents = 0;
    const created = await uploadVideo(blob, () => { progressEvents++; });
    expect(created.id).toMatch(/^[0-9a-f]{32}$/);

    const stages = new Set<string>();
    const final = await waitForAnalysis(created.id, (s: AnalysisStatus) => { if (s.stage) stages.add(s.stage); }, undefined, 40);
    console.log(`observed stages: ${[...stages].join(" > ") || "(finished between polls)"}; upload progress events: ${progressEvents}`);
    expect(final.status).toBe("completed");
    const r = final.result!;
    expect(r.interpretation.calibrated).toBe(false);
    expect(r.fusion.calibrated).toBe(false);

    if (expectMode === "doubles") {
      expect(r.is_test_fixture).toBe(true);
      expect(r.fusion.state).toBe("multimodal");
      expect(r.modalities.audio!.segments!.length).toBeGreaterThan(0);
      expect(r.modalities.visual!.frames!.some((f) => f.score !== null && f.t_s >= 0)).toBe(true);
    } else {
      expect(r.is_test_fixture).toBe(false);
      expect(r.modalities.audio!.status).toBe("available");
      expect(r.modalities.audio!.model!.license).toMatch(/MIT/);
      expect(stages.has("audio")).toBe(true);
    }

    expect((await listAnalyses()).items.some((i) => i.id === created.id)).toBe(true);
    await deleteAnalysis(created.id);
    await expect(deleteAnalysis(created.id)).rejects.toMatchObject({ status: 404 });
  }, 180_000);  // real model inference on CPU can take several seconds

  it("surfaces a structured error for a non-video upload", async () => {
    const err = await uploadVideo(new Blob([new TextEncoder().encode("this is not a video")]), () => undefined).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiRequestError);
    expect((err as ApiRequestError).body.code).toBe("unreadable_media");
  });
});
