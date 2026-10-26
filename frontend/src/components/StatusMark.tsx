import type { AssessmentStatus } from "../api";
import { ASSESSMENT_LABEL } from "../lib/format";
import { Icon } from "./Icon";

export type MarkTone = AssessmentStatus | "failed" | "ok" | "neutral";

const ICON = { flagged: "flag", not_flagged: "dash", insufficient_evidence: "unknown", failed: "alert", ok: "dash", neutral: "dash" } as const;

/** One glyph per state, so meaning is never carried by colour alone. */
export function StatusMark({ tone, label }: { tone: MarkTone; label?: string }) {
  const text = label ?? (tone === "failed" ? "Analysis failed" : tone === "ok" || tone === "neutral" ? "" : ASSESSMENT_LABEL[tone]);
  return (
    <span className={`status ${tone}`}>
      <Icon name={ICON[tone]} size={14} />
      {text}
    </span>
  );
}
