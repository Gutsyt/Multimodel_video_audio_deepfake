// UI rendering tests over HAND-WRITTEN fixtures (not real inference): they check what the interface
// claims for each modality combination, never whether a detector is right.
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { absent, audioOk, make, visualOk } from "../testFixtures";
import { claimsAuthenticity } from "../testUtils";
import { ResultView } from "./ResultView";

const text = (c: HTMLElement) => c.textContent ?? "";

describe("ResultView missing-modality behaviour", () => {
  it("video + audio + face: both branches and a fused score", () => {
    const { container } = render(<ResultView result={make(audioOk(0.8), visualOk(0.6))} />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Flagged by the model");
    expect(within(screen.getByLabelText("Fusion")).getByText("performed")).toBeInTheDocument();
    expect(text(container)).toContain("Fused raw score");
    expect(claimsAuthenticity(text(container))).toBe(false);
  });

  it("video without usable audio: visual-only, no fused score, not called multimodal", () => {
    const { container } = render(<ResultView result={make(absent("missing", ["no audio track in this file"]), visualOk(0.3))} />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Not flagged by the model");
    expect(text(container)).toContain("Visual branch only");
    expect(text(container)).toContain("single-branch result, not a multimodal one");
    expect(within(screen.getByLabelText("Fusion")).getByText("not performed")).toBeInTheDocument();
    expect(text(container)).not.toContain("Fused raw score");
    expect(text(container)).toContain("no audio track in this file");
  });

  it("video without a face: audio-only", () => {
    const { container } = render(<ResultView result={make(audioOk(0.9), absent("unreliable", ["faces found in 0 of 16 sampled frames"]))} />);
    expect(text(container)).toContain("Audio branch only");
    expect(within(screen.getByLabelText(/visual branch/i)).getByText("Too unreliable to score")).toBeInTheDocument();
    expect(text(container)).not.toContain("Fused raw score");
  });

  it("audio-only upload shows no visual score", () => {
    const { container } = render(<ResultView result={make(audioOk(0.2), absent("missing", ["this audio file has no video stream"]), "audio")} />);
    expect(text(container)).toContain("this audio file has no video stream");
    expect(within(screen.getByLabelText(/visual branch/i)).getByText("not produced")).toBeInTheDocument();
  });

  it("image-only upload shows the visual branch and no audio score", () => {
    const { container } = render(<ResultView result={make(absent("missing", ["this image has no audio"]), visualOk(0.7, "image"), "image")} />);
    expect(text(container)).toContain("this image has no audio");
    expect(text(container)).toContain("Visual branch only");
    expect(text(container)).toContain("n/a (still image)");
  });

  it("no usable modality: insufficient evidence, never authentic", () => {
    const { container } = render(<ResultView result={make(absent("missing"), absent("unreliable", ["no face"]))} />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Insufficient evidence");
    expect(text(container)).toContain("No usable evidence");
    expect(claimsAuthenticity(text(container))).toBe(false);
  });

  it("never presents raw scores as probabilities or percentages", () => {
    const { container } = render(<ResultView result={make(audioOk(0.8), visualOk(0.6))} />);
    expect(text(container)).not.toMatch(/\b\d+(\.\d+)?\s?%\s*(probab|confiden|likel|fake|real)/i);
    expect(text(container)).toContain("not a calibrated probability");
    expect(text(container)).toContain("unvalidated");
  });

  it("surfaces the output semantics and a corrupted branch's reason", () => {
    render(<ResultView result={make(absent("corrupted", ["decode_failed: audio stream could not be decoded"]), visualOk(0.4))} />);
    expect(screen.getAllByText(/decode_failed/)[0]).toBeInTheDocument();
    expect(screen.getByText(/softmax mass on the/)).toBeInTheDocument();
  });
});
