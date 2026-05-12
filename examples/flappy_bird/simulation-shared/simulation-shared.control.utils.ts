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
  const numericOutputs = resolveNumericOutputs(rawOutputs);

  if (
    numericOutputs &&
    typeof numericOutputs[0] === 'number' &&
    typeof numericOutputs[1] === 'number'
  ) {
    return numericOutputs[1] > numericOutputs[0];
  }

  if (numericOutputs && typeof numericOutputs[0] === 'number') {
    return numericOutputs[0] > flapThreshold;
  }

  return typeof rawOutputs === 'number' ? rawOutputs > flapThreshold : false;
}

function resolveNumericOutputs(
  rawOutputs: unknown,
): ArrayLike<number> | undefined {
  if (Array.isArray(rawOutputs)) {
    return rawOutputs;
  }

  const typedArrayOutputs = rawOutputs as { length?: unknown } | undefined;

  if (
    ArrayBuffer.isView(rawOutputs) &&
    typeof typedArrayOutputs?.length === 'number'
  ) {
    return rawOutputs as unknown as ArrayLike<number>;
  }

  return undefined;
}
