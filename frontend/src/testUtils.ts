// Test helper: does a piece of UI text CLAIM the media is authentic?
// Sentences that explicitly negate it ("not evidence that the media is authentic") are the
// wording we WANT, so they are removed before the check.
const CLAIM = /\b(is|are|looks?|appears?|seems?|was|were)\s+(authentic|real|genuine|unmanipulated)\b|\bverified\b|\bproven\b/i;

export function claimsAuthenticity(text: string): boolean {
  const withoutNegations = text
    .split(/(?<=[.!?])\s+/)
    .filter((sentence) => !/\bnot evidence\b|\bneither evidence\b|\bnot proof\b|\bnot mean\b/i.test(sentence))
    .join(" ");
  return CLAIM.test(withoutNegations);
}
