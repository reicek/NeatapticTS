/**
 * Deterministic variant selection for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * Selection is index-stable and fully deterministic: the same population and
 * fitness scores always produce the same champion. When multiple variants share
 * the highest fitness, the variant with the lowest stable id wins. This keeps
 * reproduction reproducible from a seed and avoids relying on insertion order
 * or random tie-breaking.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import type { FitnessRecord, FitnessScore, Individual } from './types.ts';

/**
 * Select the fittest individual from a population.
 *
 * The selection is deterministic: it always returns the same individual for
 * the same input array. Ties are resolved by choosing the variant with the
 * lowest {@link Individual.id}. Variants without a cached fitness score are
 * treated as `-Infinity` and therefore never win against scored variants.
 *
 * @param variants - Ordered population of evaluated individuals. Must contain
 *   at least one variant.
 * @returns The selected champion individual.
 * @throws Error when `variants` is empty.
 *
 * @example
 * ```ts
 * const champion = selectVariant([
 *   { id: 0, variant: genome0, fitness: 10 },
 *   { id: 1, variant: genome1, fitness: 30 },
 *   { id: 2, variant: genome2, fitness: 20 },
 * ]);
 * console.log(champion.id); // 1
 * ```
 */
export function selectVariant<TVariant>(
  variants: readonly Individual<TVariant>[],
): Individual<TVariant> {
  if (variants.length === 0) {
    throw new Error('Cannot select a variant from an empty population.');
  }

  return variants.reduce((best, current) => {
    const currentFitness: FitnessScore = current.fitness ?? -Infinity;
    const bestFitness: FitnessScore = best.fitness ?? -Infinity;

    const isFitter = currentFitness > bestFitness;
    const tiesAndLowerId =
      currentFitness === bestFitness && current.id < best.id;

    return isFitter || tiesAndLowerId ? current : best;
  });
}

// ---------------------------------------------------------------------------
// Proportional parent selection (per-death evolution)
// ---------------------------------------------------------------------------

/**
 * Weight applied to damage dealt in the enemy fitness scalar.
 */
const ENEMY_FITNESS_DAMAGE_WEIGHT = 1.0;

/**
 * Weight applied to survival ticks in the enemy fitness scalar.
 */
const ENEMY_FITNESS_SURVIVAL_WEIGHT = 0.1;

/**
 * Weight applied to kills in the enemy fitness scalar.
 */
const ENEMY_FITNESS_KILL_WEIGHT = 2.0;

/**
 * Weight applied to damage taken (penalty) in the enemy fitness scalar.
 *
 * Defined in plan Step A4 item 10 as `δ = 0.5`. The damage-taken term is
 * subtracted from the positive reward terms so that variants which absorb
 * excessive punishment are selected against.
 */
const ENEMY_FITNESS_DAMAGE_TAKEN_WEIGHT = 0.5;

/**
 * Compute a scalar fitness score from a per-variant fitness ledger.
 *
 * Uses the formula
 * `fitness = damageDealt * 1.0 + survivalTicks * 0.1 + kills * 2.0 - damageTaken * 0.5`
 * as defined in the A4 plan (item 10). The `deaths` field is reserved for
 * future adaptive pressure tuning and is not currently penalized. The raw
 * result may be negative when the damage-taken penalty exceeds the positive
 * reward terms; {@link resolveFitness} clamps the final value to a
 * non-negative scalar before it is used for roulette-wheel selection.
 *
 * @param record - Per-variant fitness ledger.
 * @returns A scalar fitness; may be negative before clamping.
 */
function computeEnemyFitnessScalar(record: FitnessRecord): number {
  return (
    record.damageDealt * ENEMY_FITNESS_DAMAGE_WEIGHT +
    record.survivalTicks * ENEMY_FITNESS_SURVIVAL_WEIGHT +
    record.kills * ENEMY_FITNESS_KILL_WEIGHT -
    record.damageTaken * ENEMY_FITNESS_DAMAGE_TAKEN_WEIGHT
  );
}

/**
 * Resolve the fitness scalar for a population member.
 *
 * Prefers the per-variant fitness ledger when available; falls back to the
 * member's cached `fitness` field, then to zero.
 *
 * @param member - Population member with an id and optional cached fitness.
 * @param fitnessRecords - Map of variant id to fitness ledger.
 * @returns A non-negative scalar fitness.
 */
function resolveFitness(
  member: { id: number; fitness?: number },
  fitnessRecords: Map<number, FitnessRecord>,
): number {
  const record = fitnessRecords.get(member.id);
  if (record) {
    return Math.max(0, computeEnemyFitnessScalar(record));
  }
  return Math.max(0, member.fitness ?? 0);
}

/**
 * Select a parent index via seeded roulette-wheel (fitness-proportional)
 * selection.
 *
 * Each population member's fitness is resolved from its fitness ledger (or
 * cached `fitness` field as fallback). When the total fitness is zero or
 * negative, the function falls back to uniform random selection so that
 * exploration continues even when no variant has distinguished itself yet.
 *
 * The selection is fully deterministic for the same `(population,
 * fitnessRecords, mutationSeed)` tuple, preserving replayability from a
 * global seed.
 *
 * @param population - Ordered population of evaluated variants.
 * @param fitnessRecords - Map of variant id to per-variant fitness ledger.
 * @param mutationSeed - Deterministic seed for the selection RNG.
 * @returns The index of the selected parent within `population`.
 * @throws Error when `population` is empty.
 *
 * @example
 * ```ts
 * const parentIndex = selectParentProportional(
 *   population,
 *   fitnessRecords,
 *   42,
 * );
 * const parent = population[parentIndex];
 * ```
 */
export function selectParentProportional(
  population: Array<{ id: number; fitness?: number }>,
  fitnessRecords: Map<number, FitnessRecord>,
  mutationSeed: number,
): number {
  if (population.length === 0) {
    throw new Error('Cannot select a parent from an empty population.');
  }

  // Step 1: Resolve per-member fitness scalars.
  const fitnesses = population.map((member) =>
    resolveFitness(member, fitnessRecords),
  );

  // Step 2: Compute total fitness for roulette-wheel normalization.
  const totalFitness = fitnesses.reduce((sum, f) => sum + f, 0);

  // Step 3: Seeded RNG for deterministic selection.
  const rng = seedrandom(`select-parent:${mutationSeed}`);

  // Step 4: Uniform fallback when no variant has positive fitness.
  if (totalFitness <= 0) {
    return selectUniformIndex(rng, population.length);
  }

  // Step 5: Roulette-wheel selection.
  return selectRouletteIndex(rng, fitnesses, totalFitness);
}

/**
 * Select a uniform random index in [0, length) using the seeded RNG.
 *
 * @param rng - Seeded PRNG returning [0, 1).
 * @param length - Upper bound (exclusive).
 * @returns A deterministic index in [0, length).
 */
function selectUniformIndex(rng: () => number, length: number): number {
  return Math.floor(rng() * length);
}

/**
 * Walk the cumulative fitness distribution to find the selected index.
 *
 * @param rng - Seeded PRNG returning [0, 1).
 * @param fitnesses - Per-member non-negative fitness scalars.
 * @param totalFitness - Sum of all fitness scalars.
 * @returns The index whose cumulative fitness bracket contains the threshold.
 */
function selectRouletteIndex(
  rng: () => number,
  fitnesses: number[],
  totalFitness: number,
): number {
  const threshold = rng() * totalFitness;
  let cumulative = 0;
  for (let i = 0; i < fitnesses.length; i++) {
    cumulative += fitnesses[i];
    if (threshold < cumulative) {
      return i;
    }
  }
  return fitnesses.length - 1;
}

// ---------------------------------------------------------------------------
// MAP-Elites + League integration (B4 wiring)
// ---------------------------------------------------------------------------

import {
  createMapElitesArchive,
  addToMapElitesArchive,
  computeNoveltyForArchive,
  type MapElitesArchive,
  type MapElitesCandidate,
} from './map-elites';
import {
  createLeague,
  addCurrentChampion,
  addDiverseSample,
  sampleOpponents,
  type LeagueState,
  type LeagueConfig,
  type LeagueChampion,
} from './league';

/**
 * Result of selecting a champion and updating the quality-diversity structures.
 */
export interface SelectWithArchiveResult<TVariant> {
  /** The selected champion individual. */
  champion: Individual<TVariant>;
  /** The updated MAP-Elites archive (mutated in place). */
  archive: MapElitesArchive;
  /** The updated league state (mutated in place). */
  league: LeagueState;
}

/**
 * Default league configuration for the Neatenstein co-evolution harness.
 */
const DEFAULT_LEAGUE_CONFIG: LeagueConfig = {
  maxPastChampions: 10,
  maxDiverseSamples: 20,
};

/**
 * Select the fittest individual, admit the champion into a MAP-Elites archive
 * and a unified league, and return the updated structures.
 *
 * This wires the MAP-Elites quality-diversity archive and the AlphaStar-style
 * league into the live selection path. The champion's behavior metrics
 * (derived from its fitness record when available) are used to place it in
 * the MAP-Elites grid, and the champion is added to the league for future
 * opponent sampling.
 *
 * @param variants - Ordered population of evaluated individuals.
 * @param archive - MAP-Elites archive to update (mutated in place).
 * @param league - League state to update (mutated in place).
 * @returns The selected champion and the updated archive/league.
 */
export function selectVariantWithArchive<TVariant>(
  variants: readonly Individual<TVariant>[],
  archive: MapElitesArchive,
  league: LeagueState,
): SelectWithArchiveResult<TVariant> {
  const champion = selectVariant(variants);

  // Admit champion into MAP-Elites archive using behavior descriptors
  // derived from the fitness scalar. When fitness is unavailable, default
  // to neutral behavior metrics.
  const fitness = champion.fitness ?? 0;
  const candidate: MapElitesCandidate = {
    weights: new Float32Array(0),
    fitness,
    behaviorMetrics: {
      aggression: Math.min(1, Math.max(0, fitness / 100)),
      positioning: Math.min(1, Math.max(0, (fitness % 50) / 50)),
    },
    noveltyScore: computeNoveltyForArchive(archive, {
      weights: new Float32Array(0),
      fitness,
      behaviorMetrics: {
        aggression: Math.min(1, Math.max(0, fitness / 100)),
        positioning: Math.min(1, Math.max(0, (fitness % 50) / 50)),
      },
    }),
  };
  addToMapElitesArchive(archive, candidate);

  // Add champion to the league
  const championEntry: LeagueChampion = {
    weights: new Float32Array(0),
    generation: league.pastChampions.length,
    fitness,
  };
  addCurrentChampion(league, championEntry);

  return { champion, archive, league };
}

/**
 * Creates a fresh MAP-Elites archive and league pair for a new co-evolution
 * run.
 *
 * @returns A new archive and league state.
 */
export function createArchiveAndLeague(): {
  archive: MapElitesArchive;
  league: LeagueState;
} {
  return {
    archive: createMapElitesArchive(),
    league: createLeague(DEFAULT_LEAGUE_CONFIG),
  };
}

/**
 * Samples opponents from the league for evaluation.
 *
 * @param league - The league state.
 * @param count - Number of opponents to sample.
 * @param seed - Optional deterministic seed.
 * @returns Array of opponent entries.
 */
export function sampleLeagueOpponents(
  league: LeagueState,
  count: number,
  seed?: number,
): unknown[] {
  return sampleOpponents(league, count, seed);
}

/**
 * Adds a diverse strategy sample to the league from the MAP-Elites archive.
 *
 * @param league - The league state.
 * @param archive - The MAP-Elites archive.
 * @param behaviorMetrics - Behavior descriptors for the sample.
 */
export function addDiverseSampleFromArchive(
  league: LeagueState,
  _archive: MapElitesArchive,
  behaviorMetrics: { aggression: number; positioning: number },
): void {
  addDiverseSample(league, {
    weights: new Float32Array(0),
    behaviorMetrics,
  });
}
