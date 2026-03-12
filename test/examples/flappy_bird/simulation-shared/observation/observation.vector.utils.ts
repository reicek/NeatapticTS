import type { SharedObservationFeatures } from '../simulation-shared.types';

/**
 * Converts observation features to the canonical 12-value network input vector.
 *
 * Educational note:
 * This module owns the network-shape projection so feature semantics can change
 * independently from how the policy input is ordered.
 *
 * The 12-value vector is the compact feed-forward policy input used by the main
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
 * This is the representation used when the example wants a short history of raw
 * observation slices. The idea is similar to frame stacking in reinforcement
 * learning: a feed-forward policy can recover some sense of motion by looking
 * at several recent compact frames at once.
 *
 * The Wikipedia article on "frame stacking" is a useful conceptual reference.
 *
 * @param features - Structured observation features.
 * @returns Core per-frame vector.
 * @example
 * ```ts
 * const coreFrame = resolveCoreObservationVectorFromFeatures(features);
 * observationMemoryState.previousCoreFrames.push(coreFrame);
 * ```
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
