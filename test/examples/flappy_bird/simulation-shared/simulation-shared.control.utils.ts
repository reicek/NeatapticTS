/**
 * Resolves flap/no-flap decision from network outputs.
 *
 * Educational note:
 * The shared control layer accepts both two-output competitive policies
 * (`no flap` vs `flap`) and simpler single-output thresholded policies. That
 * flexibility makes the helper reusable across experiments without forcing every
 * caller to reshape its outputs first.
 *
 * @param rawOutputs - Activation output payload.
 * @param flapThreshold - Scalar threshold for single-output policies.
 * @returns True when flap should trigger.
 */
export function resolveFlapDecision(
  rawOutputs: unknown,
  flapThreshold = 0.5,
): boolean {
  if (
    Array.isArray(rawOutputs) &&
    typeof rawOutputs[0] === 'number' &&
    typeof rawOutputs[1] === 'number'
  ) {
    return rawOutputs[1] > rawOutputs[0];
  }

  if (Array.isArray(rawOutputs) && typeof rawOutputs[0] === 'number') {
    return rawOutputs[0] > flapThreshold;
  }

  return typeof rawOutputs === 'number' ? rawOutputs > flapThreshold : false;
}
