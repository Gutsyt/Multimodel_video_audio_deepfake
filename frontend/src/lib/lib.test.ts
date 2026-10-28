import { describe, expect, it } from "vitest";
import { caps } from "../testFixtures";
import { describeError } from "./format";
import { parseHash } from "./route";
import { acceptedExtensions, detectKind, validateFile } from "./validate";

describe("upload validation", () => {
  const c = caps();
  it("detects the kind from the backend-provided extensions", () => {
    const acc = acceptedExtensions(c);
    expect(detectKind("clip.MP4", acc)).toBe("video");
    expect(detectKind("voice.wav", acc)).toBe("audio");
    expect(detectKind("notes.txt", acc)).toBeNull();
  });
  it("rejects empty, unsupported and oversized files", () => {
    expect(validateFile({ name: "a.mp4", size: 100 }, c).errors).toEqual([]);
    expect(validateFile({ name: "a.mp4", size: 0 }, c).errors[0]).toMatch(/empty/);
    expect(validateFile({ name: "a.txt", size: 5 }, c).errors[0]).toMatch(/Unsupported/);
    expect(validateFile({ name: "a.mp4", size: 2_000_000 }, c).errors[0]).toMatch(/up to 1 MB/);
  });
});

describe("copy and routes", () => {
  it("maps error codes and keeps unknown ones", () => {
    expect(describeError({ code: "file_too_large", message: "x", details: { max_upload_mb: 100 } })).toMatch(/100 MB/);
    expect(describeError({ code: "weird", message: "server said this" })).toBe("server said this");
  });
  it("routes", () => {
    expect(parseHash("#/history")).toEqual({ name: "history" });
    expect(parseHash("#/analysis/" + "a".repeat(32))).toEqual({ name: "analysis", id: "a".repeat(32) });
    expect(parseHash("#/analysis/../x")).toEqual({ name: "analyze" });
  });
});
