/**
 * Single main-agent lifecycle runner for the Neatenstein asymmetric
 * co-evolution harness.
 *
 * The runner advances one deterministic generation for the main NEAT agent. It
 * creates a frozen seed pack, materialises a small population of main-agent
 * variants, runs each variant through a deterministic episode against the
 * current enemy snapshot, scores the resulting {@link CombatQualitySignal}
 * with the combat-quality composite, and selects a champion using deterministic
 * selection. The returned quality signal belongs to the winning variant and
 * is fully replay-safe from the same `(seed, generation, enemySnapshot)` tuple.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import { createMlpEnemyPopulation } from './enemy-mlp';
import { computeCombatQualitySignal } from './fitness';
import { createSeedPack } from './seed-pack';
import { shouldRefreshMlpSnapshot } from './snapshot';
import { selectVariant } from './select';
import type {
  CombatQualitySignal,
  FitnessScore,
  Genome,
  Individual,
  MainVariant,
  SeedPack,
  Snapshot,
} from './types';

/**
 * Number of main-agent variants evaluated in a single generation.
 *
 * The main population is small because each variant must play a full episode;
 * selection pressure comes from deterministic replay rather than from a large
 * population.
 */
const NEATENSTEIN_MAIN_VARIANT_COUNT = 8;

/**
 * Maximum episode length in ticks (frames).
 *
 * Caps the deterministic simulation so a single generation finishes in
 * bounded time even when the agent survives indefinitely.
 */
const NEATENSTEIN_MAX_EPISODE_TICKS = 240;

/**
 * An evaluated main-agent variant, extending {@link Individual} with the raw
 * combat-quality signal produced by its episode.
 */
interface EvaluatedMainVariant extends Individual<MainVariant> {
  /** Raw combat-quality signal for the variant's episode. */
  signal: CombatQualitySignal;
}

/**
 * Configuration accepted by {@link runMainGeneration}.
 */
export interface RunMainGenerationOptions {
  /** Deterministic seed for the generation. */
  seed: number;
  /** Current co-evolution generation (non-negative integer). */
  generation: number;
  /** Frozen enemy snapshot to evaluate against. */
  enemySnapshot?: Snapshot;
}

/**
 * Run a single main-agent generation and return the champion's combat-quality
 * signal.
 *
 * The runner is fully deterministic: the same `seed`, `generation`, and
 * `enemySnapshot` always produce the same quality signal. It uses the harness
 * seed pack so every variant sees the same environmental randomness within a
 * generation, and it applies the combat-quality fitness composite so selection
 * ranks variants consistently.
 *
 * @param options - Generation configuration.
 * @returns The {@link CombatQualitySignal} of the selected champion variant.
 *
 * @example
 * ```ts
 * const quality = runMainGeneration({
 *   seed: 1,
 *   generation: 1,
 *   enemySnapshot: { kind: 'mlp', weights: new Float32Array(8) },
 * });
 * console.log(quality.survivalTicks, quality.damageDealt);
 * ```
 */
export function runMainGeneration(
  options: RunMainGenerationOptions,
): CombatQualitySignal {
  // Step 1: Resolve the enemy snapshot for this generation.
  const enemySnapshot = resolveEnemySnapshot(options);

  // Step 2: Build the deterministic seed pack for the generation.
  const seedPack = createSeedPack({ generation: options.generation });

  // Step 3: Materialise the main-agent variant population.
  const variants = createMainVariants(options.seed, options.generation);

  // Step 4: Evaluate every variant in a deterministic episode.
  const evaluated = evaluateVariants(variants, enemySnapshot, seedPack);

  // Step 5: Select the fittest variant.
  const champion = selectVariant(evaluated) as EvaluatedMainVariant;

  // Step 6: Return the champion's raw quality signal.
  return champion.signal;
}

/**
 * Resolve the enemy snapshot for the current generation.
 *
 * When the caller supplies a snapshot it is used directly. Otherwise a fresh
 * MLP enemy population is generated from the generation seed and advanced to
 * the current generation using the MLP refresh gate.
 *
 * @param options - Runner options.
 * @returns Enemy snapshot for the episode roster.
 */
function resolveEnemySnapshot(options: RunMainGenerationOptions): Snapshot {
  if (options.enemySnapshot) {
    return options.enemySnapshot;
  }

  const population = createMlpEnemyPopulation({ seed: options.seed });
  if (shouldRefreshMlpSnapshot(options.generation)) {
    return population.update({ generation: options.generation });
  }
  return population.snapshot();
}

/**
 * Build the deterministic main-agent variant population for a generation.
 *
 * Each variant receives a stable id and a deterministic genome sized from
 * the generation seed. The population is intentionally small so that every
 * variant can be evaluated end-to-end.
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
 * The genome is not wired to the library construction pipeline in this slice;
 * it provides a stable complexity count for the parsimony band and a stable
 * identity for the episode RNG. Future slices will replace this with real NEAT
 * genomes.
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
function evaluateVariants(
  variants: readonly MainVariant[],
  enemySnapshot: Snapshot,
  seedPack: SeedPack,
): EvaluatedMainVariant[] {
  return variants.map((variant, index) => {
    const episodeSeed = seedPack.seeds[index % seedPack.seeds.length];
    const signal = runEpisode(variant, enemySnapshot, episodeSeed);
    const complexity = countComplexity(variant);
    const fitness: FitnessScore = computeCombatQualitySignal(
      signal,
      complexity,
    );

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
 * This is a lightweight headless stand-in for the full game episode. The
 * outcome depends on the variant id, the enemy snapshot, and the per-variant
 * episode seed so that fitness differences reflect the variant, not random
 * environmental noise.
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
 * The hash is used to make episode outcomes depend on the enemy state without
 * needing the full game engine in this slice.
 *
 * @param enemySnapshot - Frozen enemy snapshot.
 * @returns Stable hash string.
 */
function hashEnemySnapshot(enemySnapshot: Snapshot): string {
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
