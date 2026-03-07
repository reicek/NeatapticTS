import type { SharedObservationFeatures } from '../simulation-shared.types';

/**
 * Converts observation features to the canonical 12-value network input vector.
 *
 * Educational note:
 * This module owns the network-shape projection so feature semantics can change
 * independently from how the policy input is ordered.
 *
 * @param features - Structured feature object.
 * @returns Ordered feature vector.
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
    features.normalizedDistanceToSecondPipe,
    features.normalizedDeltaToSecondGap,
    features.normalizedTimeToNextPipe,
    features.normalizedNextGapClearance,
    features.normalizedRequiredVerticalVelocityToNextGap,
    features.normalizedNextToSecondGapTransition,
  ];
}

/**
 * Resolves the compact core vector used for temporal stacking.
 *
 * The core intentionally keeps directly observed kinematic and geometric
 * channels while dropping derived one-step predictors that become redundant
 * once short-term temporal memory is available.
 *
 * @param features - Structured observation features.
 * @returns Core per-frame vector.
 */
export function resolveCoreObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[] {
  return [
    features.normalizedBirdY,
    features.normalizedVelocity,
    features.normalizedDistanceToNextPipe,
    features.normalizedDeltaToNextGap,
    features.normalizedNextGapTop,
    features.normalizedNextGapBottom,
    features.normalizedDistanceToSecondPipe,
    features.normalizedDeltaToSecondGap,
    features.normalizedNextGapClearance,
    features.normalizedRequiredVerticalVelocityToNextGap,
    features.normalizedEntryUrgency,
    features.normalizedOneFlapReachabilityAtGapEntry,
  ];
}
