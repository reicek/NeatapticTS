/**
 * Strategy-divergence analytics for racing coevolution.
 *
 * Accumulates per-generation team-level observables (aggregate fitness,
 * pit-lap distributions, reproduction-mode mix) and classifies the resulting
 * time series to detect whether the two teams' strategies are diverging in an
 * alternating arms-race pattern or one team is consistently dominant.
 *
 * Key concepts:
 * - **Advantage**: `teamAFitness - teamBFitness` at each generation boundary.
 *   Positive means Team A leads; negative means Team B leads.
 * - **isAlternating**: true when every consecutive advantage pair flips sign,
 *   indicating a balanced coevolution arms race rather than one-team dominance.
 * - **dominantPeriod**: estimated oscillation period of the advantage signal.
 *   2 when alternating (advantage flips each generation), 1 when one team
 *   dominates (no sign flip).
 * - **advantageAmplitude**: mean of absolute advantage values across all
 *   recorded generations. Measures how far apart the teams' fitness is on
 *   average.
 * - **divergenceScore**: normalised advantage amplitude divided by the maximum
 *   fitness observed, clamped to [0, 1]. A higher score means the teams'
 *   strategies are diverging more strongly.
 *
 * These metrics are observability-only — they do NOT change fitness or
 * reproduction. The host can use them to decide whether to adjust curriculum
 * parameters, but the analytics module itself has no side effects.
 *
 * ## Analytics flow
 *
 * The diagram below shows how per-generation race results flow through the
 * tracker and classifier to produce observability metrics. The analytics
 * module has no side effects — it records and classifies, but never changes
 * fitness or reproduction.
 *
 * ```mermaid
 * flowchart LR
 *     A["Race completes"] --> B["Extract team fitness<br/>+ pit-lap distributions"]
 *     B --> C["recordSnapshot()"]
 *     C --> D["Accumulate trajectory"]
 *     D --> E["classify()"]
 *     E --> F["isAlternating?"]
 *     E --> G["divergenceScore"]
 *     E --> H["dominantPeriod"]
 *     E --> I["advantageAmplitude"]
 *     F --> J["Observability metrics<br/>(no side effects)"]
 *     G --> J
 *     H --> J
 *     I --> J
 * ```
 *
 * The alternating advantage pattern this classifier detects is the signature of
 * a balanced competitive coevolution arms race. See
 * [Coevolution (Wikipedia)](https://en.wikipedia.org/wiki/Coevolution)
 * for background on why sign-flipping advantage indicates neither team has
 * collapsed into a fixed-point equilibrium.
 */

import type {
  StrategyDivergenceClassifierResult,
  StrategyDivergenceSnapshot,
  StrategyDivergenceTracker,
} from './simulation-worker.evolution.types.js';

// ───────────────────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────────────────

/** Default minimum generations before classification produces non-zero output. */
const DEFAULT_MIN_GENERATIONS = 2;

// ───────────────────────────────────────────────────────────────────────────
// Public API
// ───────────────────────────────────────────────────────────────────────────

/**
 * Create a strategy-divergence tracker that accumulates per-generation
 * snapshots and classifies the team-fitness time series.
 *
 * The tracker is stateful but side-effect-free: it only records snapshots
 * and computes read-only classifier results. It does not modify the
 * snapshots it receives.
 *
 * @param config - Configuration object
 * @param config.teamSize - Number of cars per team (e.g. 3 for a 6-car pack)
 * @param config.minGenerations - Minimum snapshots before classify() returns
 *   a non-zero result. Defaults to 2 when omitted or less than 2.
 * @returns A `StrategyDivergenceTracker` with recordSnapshot, classify, and
 *   getTrajectory methods
 *
 * @example
 * ```ts
 * const tracker = createStrategyDivergenceTracker({ teamSize: 3, minGenerations: 2 });
 * tracker.recordSnapshot({ generation: 0, teamAFitness: 10, teamBFitness: 8, ... });
 * const result = tracker.classify();
 * console.log(result.isAlternating, result.divergenceScore);
 * ```
 */
export function createStrategyDivergenceTracker(config: {
  readonly teamSize: number;
  readonly minGenerations?: number;
}): StrategyDivergenceTracker {
  const minGenerations = Math.max(
    DEFAULT_MIN_GENERATIONS,
    config.minGenerations ?? DEFAULT_MIN_GENERATIONS,
  );

  // Local accumulator — snapshots are stored in a mutable array but exposed
  // as a read-only view via getTrajectory().
  const trajectory: StrategyDivergenceSnapshot[] = [];

  /**
   * Append one generation's strategy-divergence snapshot to the trajectory.
   */
  function recordSnapshot(snapshot: StrategyDivergenceSnapshot): void {
    trajectory.push(snapshot);
  }

  /**
   * Return the read-only snapshot trajectory.
   */
  function getTrajectory(): readonly StrategyDivergenceSnapshot[] {
    return trajectory;
  }

  /**
   * Classify the accumulated trajectory and return a classifier result.
   *
   * Returns a zeros-default result when fewer than `minGenerations` snapshots
   * have been recorded.
   */
  function classify(): StrategyDivergenceClassifierResult {
    if (trajectory.length < minGenerations) {
      return zerosResult();
    }
    return classifyTrajectory(trajectory);
  }

  return {
    recordSnapshot,
    classify,
    getTrajectory,
  };
}

// ───────────────────────────────────────────────────────────────────────────
// Helpers (below the fold)
// ───────────────────────────────────────────────────────────────────────────

/**
 * Default zero-value classifier result returned when insufficient data
 * has been accumulated.
 */
function zerosResult(): StrategyDivergenceClassifierResult {
  return {
    isAlternating: false,
    dominantPeriod: 0,
    advantageAmplitude: 0,
    divergenceScore: 0,
  };
}

/**
 * Classify a team-fitness time series and compute divergence metrics.
 *
 * Algorithm:
 * 1. Compute per-generation advantage (teamAFitness - teamBFitness).
 * 2. Compute sign flips between consecutive advantage values.
 * 3. isAlternating = true when all consecutive pairs flip sign.
 * 4. dominantPeriod = 2 when alternating, 1 otherwise.
 * 5. advantageAmplitude = mean of absolute advantages.
 * 6. divergenceScore = advantageAmplitude / maxFitness, clamped to [0, 1].
 *
 * @param trajectory - Read-only array of strategy-divergence snapshots
 * @returns Classifier result with isAlternating, dominantPeriod,
 *   advantageAmplitude, and divergenceScore
 */
function classifyTrajectory(
  trajectory: readonly StrategyDivergenceSnapshot[],
): StrategyDivergenceClassifierResult {
  const advantages = computeAdvantages(trajectory);
  const isAlternating = detectAlternating(advantages);
  const dominantPeriod = isAlternating ? 2 : 1;
  const advantageAmplitude = computeMeanAbsoluteAdvantage(advantages);
  const maxFitness = computeMaxFitness(trajectory);
  const divergenceScore = computeDivergenceScore(
    advantageAmplitude,
    maxFitness,
  );

  return {
    isAlternating,
    dominantPeriod,
    advantageAmplitude,
    divergenceScore,
  };
}

/**
 * Compute per-generation advantage values (teamAFitness - teamBFitness).
 */
function computeAdvantages(
  trajectory: readonly StrategyDivergenceSnapshot[],
): number[] {
  return trajectory.map(
    (snapshot) => snapshot.teamAFitness - snapshot.teamBFitness,
  );
}

/**
 * Detect whether the advantage time series alternates sign on every
 * consecutive pair.
 *
 * A single-element or empty series is not alternating (no flips to detect).
 */
function detectAlternating(advantages: readonly number[]): boolean {
  if (advantages.length < 2) {
    return false;
  }

  let signFlips = 0;
  for (let i = 1; i < advantages.length; i++) {
    const previous = advantages[i - 1];
    const current = advantages[i];
    if (previous === 0 || current === 0) {
      return false;
    }
    if (Math.sign(previous) !== Math.sign(current)) {
      signFlips++;
    }
  }
  return signFlips === advantages.length - 1;
}

/**
 * Compute the mean of absolute advantage values.
 */
function computeMeanAbsoluteAdvantage(advantages: readonly number[]): number {
  if (advantages.length === 0) {
    return 0;
  }
  const sum = advantages.reduce((acc, value) => acc + Math.abs(value), 0);
  return sum / advantages.length;
}

/**
 * Compute the maximum fitness value across both teams and all generations.
 */
function computeMaxFitness(
  trajectory: readonly StrategyDivergenceSnapshot[],
): number {
  let max = 0;
  for (const snapshot of trajectory) {
    if (snapshot.teamAFitness > max) {
      max = snapshot.teamAFitness;
    }
    if (snapshot.teamBFitness > max) {
      max = snapshot.teamBFitness;
    }
  }
  return max;
}

/**
 * Compute the normalised divergence score, clamped to [0, 1].
 *
 * When maxFitness is zero (both teams scored zero), divergence is zero
 * because there is no meaningful separation to measure.
 */
function computeDivergenceScore(
  advantageAmplitude: number,
  maxFitness: number,
): number {
  if (maxFitness === 0) {
    return 0;
  }
  const raw = advantageAmplitude / maxFitness;
  return Math.min(1, Math.max(0, raw));
}
