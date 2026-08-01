/**
 * Headless enemy wave runner for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * The runner evaluates every variant in the enemy population against a fixed,
 * frozen seed pack in a single headless generation. It refreshes the rolling
 * enemy snapshot store through the generation barrier, simulates a lightweight
 * deterministic episode for each variant, scores the team-level result, and
 * selects a champion using deterministic lowest-id tie-breaking.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import { buildEnemyEvaluationBarrier, hashEnemySnapshot } from './barrier';
import { NEATENSTEIN_MAX_ACTIVE_ENEMIES } from './constants';
import { computeEnemyTeamFitness } from './fitness';
import { makeEnemySeedPack } from './seed-pack';
import { getEnemySnapshot } from './snapshot';
import { selectVariant } from './select';
import type {
  EnemyPopulation,
  EnemyTeamFitnessConfig,
  FitnessScore,
  Individual,
  MlpSnapshot,
  SeedPack,
  Snapshot,
} from './types';

/**
 * Optional configuration accepted by {@link runEnemyWaveRunner}.
 */
export interface EnemyWaveRunnerConfig {
  /** Generation label used by the evaluation barrier (defaults to `seed`). */
  generation?: number;
  /** Optional weights for the team-level enemy fitness composite. */
  fitness?: EnemyTeamFitnessConfig;
}

/**
 * Result emitted by {@link runEnemyWaveRunner} for one enemy wave.
 */
export interface EnemyWaveRunnerResult {
  /** Selected enemy champion with its score and frozen snapshot. */
  champion: {
    /** Stable variant id of the champion. */
    id: number;
    /** Scalar score used for selection. */
    score: FitnessScore;
    /** Frozen snapshot backing the champion. */
    snapshot: MlpSnapshot;
  };
  /** Per-variant scores in population order. */
  scores: { id: number; score: FitnessScore }[];
  /** Generation label used for the barrier. */
  generation: number;
  /** Fixed seed pack the wave was evaluated against. */
  seedPack: SeedPack;
}

/**
 * Run a headless enemy wave evaluation.
 *
 * All 32 enemy variants are evaluated against the same deterministic seed
 * pack. The rolling snapshot store is refreshed through
 * {@link buildEnemyEvaluationBarrier} so the evaluation reads frozen snapshots,
 * never live mutable population weights. Selection uses
 * {@link selectVariant}, which resolves ties by choosing the lowest variant id.
 *
 * @param population - Live enemy population to evaluate.
 * @param seed - Deterministic seed for the seed pack and episode RNGs.
 * @param config - Optional generation override and fitness weights.
 * @returns Champion, per-variant scores, generation, and the fixed seed pack.
 *
 * @example
 * ```ts
 * const population = createMlpEnemyPopulation({ seed: 1 });
 * const result = runEnemyWaveRunner(population, 123);
 * console.log(result.champion.id, result.scores.length); // 0..31, 32
 * ```
 */
export function runEnemyWaveRunner(
  population: EnemyPopulation,
  seed: number,
  config: EnemyWaveRunnerConfig = {},
): EnemyWaveRunnerResult {
  const generation = config.generation ?? seed;
  const seedPack = makeEnemySeedPack(seed);

  // Refresh the frozen snapshot store and establish the generation barrier.
  buildEnemyEvaluationBarrier(generation, population, seedPack);

  const evaluated: Individual<MlpSnapshot>[] = [];
  const scores: { id: number; score: FitnessScore }[] = [];

  for (let variantId = 0; variantId < population.size; variantId++) {
    const snapshot = getEnemySnapshot(variantId);
    const episodeSeed = seedPack.seeds[variantId % seedPack.seeds.length];
    const { damageDealt, enemiesSurvived } = simulateEnemyEpisode(
      snapshot,
      episodeSeed,
    );
    const score = computeEnemyTeamFitness(
      damageDealt,
      enemiesSurvived,
      config.fitness,
    );

    evaluated.push({ id: variantId, variant: snapshot, fitness: score });
    scores.push({ id: variantId, score });
  }

  const champion = selectVariant(evaluated);

  return {
    champion: {
      id: champion.id,
      score: champion.fitness as FitnessScore,
      snapshot: champion.variant,
    },
    scores,
    generation,
    seedPack,
  };
}

/**
 * Simulate one deterministic episode for a frozen enemy variant.
 *
 * The outcome is seeded by the variant's snapshot hash so different weight
 * vectors produce different but reproducible telemetry. The episode is a
 * stand-in: it returns collective damage dealt and the number of enemies that
 * survived, which the team-level fitness composite turns into a scalar.
 *
 * @param snapshot - Frozen enemy snapshot.
 * @param episodeSeed - Deterministic seed from the generation seed pack.
 * @returns Damage dealt to the main agent and surviving enemy count.
 */
function simulateEnemyEpisode(
  snapshot: Snapshot,
  episodeSeed: number,
): { damageDealt: number; enemiesSurvived: number } {
  const rng = seedrandom(`${episodeSeed}:enemy:${hashEnemySnapshot(snapshot)}`);

  const activeEnemies = Math.floor(rng() * NEATENSTEIN_MAX_ACTIVE_ENEMIES) + 1;
  const damageDealt = Math.floor(rng() * 200);
  const enemiesSurvived = Math.floor(rng() * (activeEnemies + 1));

  return { damageDealt, enemiesSurvived };
}
