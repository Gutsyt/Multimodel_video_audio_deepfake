import { describe, expect, it } from "vitest";
import { caps, make, absent, audioOk, visualOk } from "../testFixtures";
import { assessmentOf, describeError } from "./format";
import { acceptedExtensions, detectKind, validateFile } from "./validate";
import { parseHash } from "./route";
import { claimsAuthenticity } from "../testUtils";

describe("upload validation (client side)", () => {
  const c = caps();
  it("detects the kind from the backend-provided extensions", () => {
    const acc = acceptedExtensions(c);
    expect(detectKind("clip.MP4", acc)).toBe("video");
    expect(detectKind("voice.wav", acc)).toBe("audio");
    expect(detectKind("photo.png", acc)).toBe("image");
    expect(detectKind("notes.txt", acc)).toBeNull();
  });
  it("rejects empty, unsupported and oversized files with readable messages", () => {
    expect(validateFile({ name: "a.mp4", size: 100 }, c).errors).toEqual([]);
    expect(validateFile({ name: "a.mp4", size: 0 }, c).errors[0]).toMatch(/empty/);
    expect(validateFile({ name: "a.txt", size: 5 }, c).errors[0]).toMatch(/Unsupported/);
    expect(validateFile({ name: "a.mp4", size: 2_000_000 }, c).errors[0]).toMatch(/up to 1 MB/);
  });
  it("uses the accepted list the server reported, not a hardcoded one", () => {
    expect(validateFile({ name: "a.avi", size: 5 }, c).kind).toBeNull();
  });
});

describe("assessment fallback for results stored before schema 0.2", () => {
  it("derives a single-branch assessment without inventing a fused score", () => {
    const r = make(audioOk(0.9), absent("missing"));
    r.assessment = null;
    r.fusion.fused_score = 0.9; // legacy shape
    const a = assessmentOf(r);
    expect(a.basis).toBe("audio_only");
    expect(a.score_kind).toBe("audio_branch_raw_score");
  });
  it("reports insufficient evidence when nothing was usable", () => {
    const r = make(absent("missing"), absent("missing"));
    r.assessment = null;
    expect(assessmentOf(r).status).toBe("insufficient_evidence");
  });
});

describe("copy", () => {
  it("maps backend error codes to readable messages and keeps unknown ones", () => {
    expect(describeError({ code: "unreadable_media", message: "x" })).toMatch(/corrupted/);
    expect(describeError({ code: "file_too_large", message: "x", details: { max_upload_mb: 100 } })).toMatch(/100 MB/);
    expect(describeError({ code: "weird", message: "server said this" })).toBe("server said this");
  });
  it("the authenticity-claim detector catches claims and ignores negations (negative control)", () => {
    expect(claimsAuthenticity("This video is authentic.")).toBe(true);
    expect(claimsAuthenticity("This is not evidence that the media is authentic.")).toBe(false);
  });
  it("routes", () => {
    expect(parseHash("#/history")).toEqual({ name: "history" });
    expect(parseHash("#/analysis/" + "a".repeat(32))).toEqual({ name: "analysis", id: "a".repeat(32) });
    expect(parseHash("#/analysis/../x")).toEqual({ name: "dashboard" });
  });
  it("visual fixture is image-shaped", () => expect(visualOk(0.5, "image").frames![0].t_s).toBeNull());
});
