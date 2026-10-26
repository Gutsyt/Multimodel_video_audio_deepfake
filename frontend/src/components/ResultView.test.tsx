import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ResultView } from "./ResultView";
import { absent, available, fusion, result } from "../testFixtures";
import { claimsAuthenticity } from "../testUtils";

const renderIt = (r: ReturnType<typeof result>) =>
  render(<ResultView result={r} analysisId={"a".repeat(32)} reportHref="/r" onDelete={() => undefined} />);

describe("ResultView (UI rendering rules, using test fixtures)", () => {
  it("missing audio: no audio score is shown and absence is not called authentic", () => {
    renderIt(result(absent("missing"), available(0.2), fusion({
      state: "single_modality", fused_score: 0.2, excluded_modalities: { audio: "missing" },
    })));
    const audio = screen.getByRole("region", { name: /audio branch/i });
    expect(within(audio).getByText("Not present in file")).toBeInTheDocument();
    expect(within(audio).getByText(/not evidence that the media is genuine or manipulated/i)).toBeInTheDocument();
    expect(within(audio).queryByText(/raw model score/i)).toBeNull();
    expect(screen.getByText(/Not used: audio/)).toBeInTheDocument();
    expect(claimsAuthenticity(document.body.textContent ?? "")).toBe(false);
  });

  it("insufficient evidence: no fused score bar and an explicit non-authenticity statement", () => {
    renderIt(result(absent("corrupted", ["decode_failed"]), absent("unreliable", ["faces found in 0 of 16"]),
      fusion({ state: "insufficient_evidence", fused_score: null, flagged_at_default_threshold: null, weights_used: {} })));
    expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
    expect(screen.queryByText(/raw fused score/i)).toBeNull();
    expect(screen.getAllByText(/not evidence/i).length).toBeGreaterThan(0);
    expect(claimsAuthenticity(document.body.textContent ?? "")).toBe(false);
  });

  it("available scores are labelled raw / not a probability, and calibration is 'no'", () => {
    renderIt(result(available(0.7), available(0.3), fusion({ fused_score: 0.5, flagged_at_default_threshold: true })));
    expect(screen.getAllByText(/not a probability/i).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/no: scores are not probabilities/i)).toBeInTheDocument();
    expect(screen.getAllByText(/unvalidated default/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/Flagged by the model at the default threshold/)).toBeInTheDocument();
  });

  it("shows the test-fixture banner when the result came from a stub", () => {
    renderIt(result(available(0.7, true), available(0.3, true), fusion({}), true));
    expect(screen.getByText(/TEST FIXTURE MODEL/)).toBeInTheDocument();
  });

  it("shows model limitations and attribution", () => {
    renderIt(result(available(0.7), absent("missing"), fusion({ state: "single_modality" })));
    expect(screen.getByText("limitation one")).toBeInTheDocument();
    expect(screen.getByText("MIT")).toBeInTheDocument();
  });
});
