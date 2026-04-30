import type { SharedObservationFeatures } from '../simulation-shared.types';

/**
 * Converts observation features to the canonical 9-value network input vector.
 *
 * Educational note:
 * This module owns the network-shape projection so feature semantics can change
 * independently from how the policy input is ordered.
 *
 * The 9-value vector is grouped into three semantic families:
 *
 * **Bird state (indices 0–1):** normalized height and vertical velocity.
 *
 * **Next gap (indices 2–5):** distance to pipe exit, signed offset from gap
 * center, normalized gap top and bottom boundaries.
 *
 * **Look-ahead (indices 6–8):** signed distance to pipe entrance (negative
 * while inside the pipe body), signed in-gap clearance (how centered the bird
 * is right now), and signed offset from the second upcoming gap center.
 *
 * Its ordering is stable on purpose: once a network topology has evolved
 * against one input layout, silent channel reshuffles would invalidate learned
 * behavior.
 *
 * @param features - Structured feature object.
 * @returns Ordered feature vector.
 * @example
 * ```ts
 * const features = resolveObservationFeatures(input);
 * const networkInput = resolveObservationVectorFromFeatures(features);
 * // networkInput.length === 9
 * ```
 */
export function resolveObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[] {
  return [
    // Bird state
    features.normalizedBirdY,
    features.normalizedVelocity,
    // Next gap
    features.normalizedDistanceToNextPipe,
    features.normalizedDeltaToNextGap,
    features.normalizedNextGapTop,
    features.normalizedNextGapBottom,
    // Look-ahead
    features.normalizedDistanceToPipeEntrance,
    features.normalizedNextGapClearance,
    features.normalizedDeltaToSecondGap,
  ];
}

/**
 * Resolves the compact per-frame vector retained for compatibility bookkeeping.
 *
 * The core intentionally keeps directly observed kinematic and geometric
 * channels while dropping some derived one-step predictors. If an opt-in
 * experiment wants external history again, this is the narrower slice worth
 * carrying between steps.
 *
 * Under the current default controller contract, however, the active network
 * input uses `resolveObservationVectorFromFeatures(features)` directly and does
 * not stack these core frames.
 *
 * @param features - Structured observation features.
 * @returns Core per-frame vector.
 * @example
 * ```ts
 * const coreFrame = resolveCoreObservationVectorFromFeatures(features);
 * console.log(coreFrame.length);
 * ```
 */
export function resolveCoreObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[] {
  return resolveObservationVectorFromFeatures(features);
}
