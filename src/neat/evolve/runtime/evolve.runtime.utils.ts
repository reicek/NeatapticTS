/*
 * ESLint configuration for intentional `any` usage in NEAT evolution runtime utils
 *
 * This file mirrors the evolution module's runtime metadata handling,
 * where dynamic properties are attached to genomes/species at runtime.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import Network from '../../../architecture/network';
import type { NeatControllerForEvolution } from '../evolve.types';

/**
 * Resolve the start time for an evolution step.
 * @returns timestamp in milliseconds or high-resolution units.
 */
export function resolveStartTime(): number {
  // Step 1: Prefer high-resolution timer when available.
  if (
    typeof performance !== 'undefined' &&
    typeof (performance as unknown as { now?: () => number }).now === 'function'
  ) {
    return (performance as unknown as { now: () => number }).now();
  }
  // Step 2: Fall back to wall-clock time.
  return Date.now();
}

/**
 * Ensure the population is evaluated before evolution operations.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export async function ensurePopulationEvaluated(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Guard against empty populations.
  const lastGenome = internal.population.at(-1);
  if (!lastGenome) return;
  // Step 2: Evaluate only if scores are missing.
  if (lastGenome.score === undefined) await internal.evaluate();
}

/**
 * Update generation-level best score tracking.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function updateGlobalBestTracking(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Read the current best score safely.
  try {
    const currentBest = internal.population[0]?.score;
    // Step 2: Update tracking if improvement detected.
    if (
      typeof currentBest === 'number' &&
      (internal._bestScoreLastGen === undefined ||
        currentBest > internal._bestScoreLastGen)
    ) {
      internal._bestScoreLastGen = currentBest;
      internal._lastGlobalImproveGeneration = internal.generation;
    }
  } catch {
    // Empty catch: score tracking may fail if population is empty.
  }
}

/**
 * Track global best improvement for stagnation logic.
 * @param internal - NEAT controller instance.
 * @param snapshot - Best network snapshot.
 * @returns void.
 */
export function trackGlobalImprovement(
  internal: NeatControllerForEvolution,
  snapshot: Network,
): void {
  // Step 1: Compare to prior global best.
  if ((snapshot.score ?? -Infinity) > internal._bestGlobalScore) {
    internal._bestGlobalScore = snapshot.score ?? -Infinity;
    internal._lastGlobalImproveGeneration = internal.generation;
  }
}

/**
 * Compute elapsed time since the start of evolve().
 * @param startTimestamp - Start time resolved earlier.
 * @returns elapsed time.
 */
export function computeElapsedTime(startTimestamp: number): number {
  // Step 1: Resolve end time.
  const endTime =
    typeof performance !== 'undefined' && (performance as any).now
      ? (performance as any).now()
      : Date.now();
  // Step 2: Return delta.
  return endTime - startTimestamp;
}

/**
 * Clear genome scores to force re-evaluation.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function clearPopulationScores(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Reset scores across the population.
  internal.population.forEach((genome: any) => (genome.score = undefined));
}

/**
 * Build a cloned Network from the current best genome.
 * @param internal - NEAT controller instance.
 * @returns best network snapshot.
 */
export function buildFittestSnapshot(
  internal: NeatControllerForEvolution,
): Network {
  // Step 1: Pick the best genome after sorting.
  const firstGenome = internal.population[0];
  // Step 2: Clone or create a new network if missing.
  const cloned = firstGenome
    ? Network.fromJSON(firstGenome.toJSON?.() ?? {})
    : new Network(internal.input, internal.output);
  cloned.score = firstGenome?.score;
  return cloned;
}
