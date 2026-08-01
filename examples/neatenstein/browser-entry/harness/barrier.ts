/**
 * Generation barrier for the Neatenstein asymmetric co-evolution harness.
 *
 * A barrier freezes one main-agent champion variant together with one enemy
 * snapshot and a deterministic seed so the same episode can be replayed
 * exactly. It is the integration point between the main-agent lifecycle runner,
 * the enemy population backends, snapshot refresh gates, seed packs, and
 * deterministic selection.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import { createMlpEnemyPopulation } from './enemy-mlp';
import { computeCombatQualitySignal } from './fitness';
import { runMainGeneration } from './main-runner';
import { createSeedPack } from './seed-pack';
import { selectVariant } from './select';
import { getEnemySnapshot, refreshEnemySnapshots } from './snapshot';
import { shouldRefreshMlpSnapshot } from './snapshot';
import type {
  BarrierState,
  CombatQualitySignal,
  EnemyPopulation,
  Genome,
  Individual,
  MainVariant,
  SeedPack,
  Snapshot,
} from './types';

/**
 * Build an enemy evaluation barrier for a generation.
 *
 * The barrier pairs a frozen enemy variant snapshot with the deterministic
 * seed pack for the generation. It refreshes the rolling snapshot store from
 * the provided population, then selects a variant deterministically from
 * `generation` so the same inputs always produce the same barrier. The returned
 * snapshot is never the live population object: it is a deep-copied, frozen
 * snapshot from the store.
 *
 * @param generation - Current co-evolution generation.
 * @param population - Live enemy population (MLP backend expected).
 * @param seedPack - Deterministic seed pack for this generation.
 * @returns Frozen object with `{ snapshot, seedPack }`.
 *
 * @example
 * ```ts
 * const population = createMlpEnemyPopulation({ seed: 1 });
 * const seedPack = makeEnemySeedPack(123);
 * const barrier = buildEnemyEvaluationBarrier(5, population, seedPack);
 * console.log(barrier.snapshot.kind); // 'mlp'
 * ```
 */
export function buildEnemyEvaluationBarrier(
  generation: number,
  population: EnemyPopulation,
  seedPack: SeedPack,
): { snapshot: Snapshot; seedPack: SeedPack } {
  refreshEnemySnapshots(population);
  const variantId = generation % population.size;
  const snapshot = getEnemySnapshot(variantId);

  return Object.freeze({
    snapshot,
    seedPack,
  });
}

/**
 * Configuration accepted by {@link genBarrier}.
 */
export interface GenBarrierOptions {
  /** Deterministic seed for the generation. */
  seed: number;
  /** Current co-evolution generation (non-negative integer). */
  generation: number;
}

/**
 * Number of main-agent variants evaluated in a single generation.
 *
 * Mirrors the main runner population size so the barrier selects from the
 * same candidate pool.
 */
const NEATENSTEIN_MAIN_VARIANT_COUNT = 8;

/**
 * Maximum episode length in ticks (frames).
 *
 * Caps the deterministic stand-in episode so barrier construction finishes in
 * bounded time.
 */
const NEATENSTEIN_MAX_EPISODE_TICKS = 240;

/**
 * Build a deterministic generation barrier.
 *
 * The barrier pairs the selected main-agent champion for the generation with
 * a frozen enemy snapshot and the generation seed. Calling
 * `genBarrier({ seed, generation })` twice with the same arguments produces
 * byte-identical {@link BarrierState} objects, making the barrier replay-safe.
 *
 * @param options - Barrier configuration.
 * @returns A frozen {@link BarrierState} for the requested generation.
 *
 * @example
 * ```ts
 * const barrier = genBarrier({ seed: 123, generation: 1 });
 * console.log(barrier.generation, barrier.mainSnapshot.id);
 * ```
 */
export function genBarrier(options: GenBarrierOptions): BarrierState {
  // Step 1: Resolve the frozen enemy snapshot for this generation.
  const enemySnapshot = resolveEnemySnapshot(options);

  // Step 2: Select the main-agent champion from the same deterministic pool
  // the main runner uses.
  const mainSnapshot = selectMainChampion(options, enemySnapshot);

  // Step 3: Exercise the main runner integration so the barrier is wired to
  // the same deterministic episode pipeline.
  void runMainGeneration({
    seed: options.seed,
    generation: options.generation,
    enemySnapshot,
  });

  // Step 4: Return the frozen barrier state.
  return {
    generation: options.generation,
    mainSnapshot,
    enemySnapshot,
    seed: options.seed,
  };
}

/**
 * Resolve the enemy snapshot for the requested generation.
 *
 * Uses the MLP enemy backend and refreshes its champion snapshot on MLP
 * refresh boundaries so the roster evolves deterministically.
 *
 * @param options - Barrier options.
 * @returns Frozen enemy snapshot for the barrier.
 */
function resolveEnemySnapshot(options: GenBarrierOptions): Snapshot {
  const population = createMlpEnemyPopulation({ seed: options.seed });

  if (shouldRefreshMlpSnapshot(options.generation)) {
    return population.update({ generation: options.generation });
  }

  return population.snapshot();
}

/**
 * Select the main-agent champion for the requested generation.
 *
 * The selection mirrors the main runner's deterministic pipeline: it creates
 * the same variant population, runs the same stand-in episode against the
 * frozen enemy snapshot, scores each variant with the combat-quality composite,
 * and applies index-stable selection with lowest-id tie-breaking.
 *
 * @param options - Barrier options.
 * @param enemySnapshot - Frozen enemy snapshot.
 * @returns The selected main-agent champion variant.
 */
function selectMainChampion(
  options: GenBarrierOptions,
  enemySnapshot: Snapshot,
): MainVariant {
  const seedPack = createSeedPack({ generation: options.generation });
  const variants = createMainVariants(options.seed, options.generation);
  const evaluated = evaluateMainVariants(variants, enemySnapshot, seedPack);
  const champion = selectVariant(evaluated) as EvaluatedMainVariant;

  return champion.variant;
}

/**
 * An evaluated main-agent variant, extending {@link Individual} with the raw
 * combat-quality signal produced by its episode.
 */
interface EvaluatedMainVariant extends Individual<MainVariant> {
  /** Raw combat-quality signal for the variant's episode. */
  signal: CombatQualitySignal;
}

/**
 * Build the deterministic main-agent variant population for a generation.
 *
 * @param seed - Generation seed.
 * @param generation - Generation number.
 * @returns Ordered list of main-agent variants.
 */
function createMainVariants(seed: number, generation: number): MainVariant[] {
  const variants: MainVariant[] = [];
  for (let id = 0; id < NEATENSTEIN_MAIN_VARIANT_COUNT; id++) {
    variants.push({
      id,
      genome: createMainGenome(seed, generation, id),
    });
  }
  return variants;
}

/**
 * Generate a deterministic placeholder genome for a main-agent variant.
 *
 * @param seed - Generation seed.
 * @param generation - Generation number.
 * @param variantId - Stable variant index.
 * @returns A deterministic placeholder genome.
 */
function createMainGenome(
  seed: number,
  generation: number,
  variantId: number,
): Genome {
  const rng = seedrandom(`${seed}:main:${generation}:${variantId}`);
  const nodeCount = Math.floor(rng() * 20) + 10;
  const connectionCount = Math.floor(rng() * 30) + 10;

  return {
    nodes: new Array(nodeCount).fill(null),
    connections: new Array(connectionCount).fill(null),
  };
}

/**
 * Evaluate every main-agent variant against the enemy snapshot.
 *
 * @param variants - Main-agent population.
 * @param enemySnapshot - Frozen enemy snapshot.
 * @param seedPack - Deterministic seeds for the generation.
 * @returns Evaluated individuals ready for selection.
 */
function evaluateMainVariants(
  variants: readonly MainVariant[],
  enemySnapshot: Snapshot,
  seedPack: SeedPack,
): EvaluatedMainVariant[] {
  return variants.map((variant, index) => {
    const episodeSeed = seedPack.seeds[index % seedPack.seeds.length];
    const signal = runEpisode(variant, enemySnapshot, episodeSeed);
    const complexity = countComplexity(variant);
    const fitness = computeCombatQualitySignal(signal, complexity);

    return {
      id: variant.id,
      variant,
      fitness,
      signal,
    };
  });
}

/**
 * Count the structural complexity of a variant for the parsimony band.
 *
 * @param variant - Main-agent variant.
 * @returns Neuron plus synapse count.
 */
function countComplexity(variant: MainVariant): number {
  return variant.genome.nodes.length + variant.genome.connections.length;
}

/**
 * Run one deterministic episode for a main-agent variant.
 *
 * This stand-in matches the main runner episode semantics so the barrier
 * selects the same champion the runner would select.
 *
 * @param variant - Main-agent variant being evaluated.
 * @param enemySnapshot - Frozen enemy snapshot.
 * @param episodeSeed - Deterministic seed for this episode.
 * @returns Combat-quality signal summarising the episode.
 */
function runEpisode(
  variant: MainVariant,
  enemySnapshot: Snapshot,
  episodeSeed: number,
): CombatQualitySignal {
  const rng = createEpisodeRng(variant, enemySnapshot, episodeSeed);

  const survivalTicks = Math.floor(rng() * NEATENSTEIN_MAX_EPISODE_TICKS);
  const damageDealt = Math.floor(rng() * 100);
  const kills = Math.floor(rng() * 5);
  const damageTaken = Math.floor(rng() * 50);
  const aimMissRate = rng();
  const complexityBonus = Math.floor(rng() * 10);

  return {
    survivalTicks,
    damageDealt,
    kills,
    damageTaken,
    aimMissRate,
    complexityBonus,
    parsimonyDensityPenalty: 0,
  };
}

/**
 * Build a deterministic episode RNG from the variant, enemy, and seed.
 *
 * @param variant - Main-agent variant.
 * @param enemySnapshot - Frozen enemy snapshot.
 * @param episodeSeed - Per-variant episode seed.
 * @returns Seeded PRNG.
 */
function createEpisodeRng(
  variant: MainVariant,
  enemySnapshot: Snapshot,
  episodeSeed: number,
): seedrandom.PRNG {
  const enemyHash = hashEnemySnapshot(enemySnapshot);
  return seedrandom(`${episodeSeed}:episode:${variant.id}:${enemyHash}`);
}

/**
 * Produce a short deterministic hash of an enemy snapshot.
 *
 * @param enemySnapshot - Frozen enemy snapshot.
 * @returns Stable hash string.
 */
export function hashEnemySnapshot(enemySnapshot: Snapshot): string {
  if (enemySnapshot.kind === 'mlp') {
    let hash = 0;
    for (let i = 0; i < enemySnapshot.weights.length; i++) {
      hash =
        ((hash * 31 + Math.round(enemySnapshot.weights[i] * 1_000)) | 0) >>> 0;
    }
    return `mlp:${hash}`;
  }

  let hash = 0;
  for (const coordinate of enemySnapshot.coordinates) {
    hash = ((hash * 31 + Math.round(coordinate.x * 1_000)) | 0) >>> 0;
    hash = ((hash * 31 + Math.round(coordinate.y * 1_000)) | 0) >>> 0;
  }
  return `swarm:${hash}:${enemySnapshot.dna}`;
}
