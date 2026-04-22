import type { SharedObservationFeatures } from '../simulation-shared.types';

/**
 * Converts observation features to the canonical 6-value network input vector.
 *
 * Educational note:
 * This module owns the network-shape projection so feature semantics can change
 * independently from how the policy input is ordered.
 *
 * The 6-value vector is the compact feed-forward policy input used by the main
 * evaluation and training flow. Its ordering is stable on purpose: once a
 * network topology has evolved against one input layout, silent channel
 * reshuffles would invalidate learned behavior.
 *
 * @param features - Structured feature object.
 * @returns Ordered feature vector.
 * @example
 * ```ts
 * const features = resolveObservationFeatures(input);
 * const networkInput = resolveObservationVectorFromFeatures(features);
 * ```
 */
export function resolveObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[] {
  return [
    features.normalizedBirdY,
    features.normalizedVelocity,
    features.normalizedDistanceToNextPipe,
    features.normalizedDeltaToNextGap,
    features.normalizedNextGapTop,
    features.normalizedNextGapBottom,
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
