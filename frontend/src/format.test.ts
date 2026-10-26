import { describe, expect, it } from "vitest";
import { describeError, fusionHeadline, STATUS_EXPLANATION, STATUS_LABEL } from "./format";
import { fusion } from "./testFixtures";
import type { ModalityStatus } from "./api";
import { claimsAuthenticity } from "./testUtils";

describe("claimsAuthenticity helper (negative control)", () => {
  it("flags real claims but not explicit negations", () => {
    expect(claimsAuthenticity("The media appears authentic.")).toBe(true);
    expect(claimsAuthenticity("Verified genuine.")).toBe(true);
    expect(claimsAuthenticity("Nothing found, so the clip is real.")).toBe(true);
    expect(claimsAuthenticity("This is not evidence that the media is authentic.")).toBe(false);
  });
});

describe("fusionHeadline never turns absence or a low score into an authenticity claim", () => {
  const cases = [
    fusion({ state: "insufficient_evidence", fused_score: null, flagged_at_default_threshold: null }),
    fusion({ state: "single_modality", fused_score: 0.1, flagged_at_default_threshold: false }),
    fusion({ state: "multimodal", fused_score: 0.05, flagged_at_default_threshold: false }),
    fusion({ state: "multimodal", fused_score: 0.9, flagged_at_default_threshold: true }),
  ];
  it.each(cases.map((c) => [c.state, c.fused_score, c]))("%s / %s", (_s, _f, c) => {
    const h = fusionHeadline(c as ReturnType<typeof fusion>);
    expect(claimsAuthenticity(`${h.title}. ${h.detail}`)).toBe(false);
  });
  it("states that 'not flagged' is not evidence of authenticity", () => {
    expect(fusionHeadline(cases[1]!).detail).toMatch(/not evidence that the media is authentic/i);
    expect(fusionHeadline(cases[0]!).tone).toBe("insufficient");
  });
});

describe("modality status wording", () => {
  const statuses = Object.keys(STATUS_LABEL) as ModalityStatus[];
  it.each(statuses)("%s is never described as authentic", (s) => {
    expect(claimsAuthenticity(`${STATUS_LABEL[s]}. ${STATUS_EXPLANATION[s]("audio track")}`)).toBe(false);
  });
  it.each(["missing", "corrupted", "unreliable", "not_analyzed"] as const)("%s says it is not evidence either way", (s) => {
    expect(STATUS_EXPLANATION[s]("audio track")).toMatch(/not evidence/i);
  });
});

describe("describeError", () => {
  it("maps known codes and keeps the code visible", () => {
    const e = describeError({ code: "duration_exceeded", message: "raw" });
    expect(e.title).toMatch(/longer than the allowed/i);
    expect(e.detail).toContain("duration_exceeded");
  });
  it("falls back to the server message for unknown codes", () => {
    expect(describeError({ code: "weird", message: "Server said this" }).title).toBe("Server said this");
  });
});
