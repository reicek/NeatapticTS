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
import { createSwarmEnemyPopulation } from './enemy-swarm';
import { computeCombatQualitySignal } from './fitness';
import { createSeedPack } from './seed-pack';
import { selectVariant } from './select';
import type {
  NgeMainAgentEmbryo,
  NgeMainAgentLifecycleConfig,
} from '../../../../src/neat/nge-main-agent/neat.nge-main-agent.types';
import { buildMainAgentEmbryo } from '../../../../src/neat/nge-main-agent/neat.nge-main-agent.embryo';

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
  /**
   * Frozen enemy snapshot to evaluate against.
   *
   * Takes precedence over {@link enemy} when both are supplied.
   */
  enemySnapshot?: Snapshot;
  /**
   * Enemy backend selector.
   *
   * When `enemySnapshot` is omitted, the runner resolves a fresh frozen
   * snapshot from the requested enemy backend. Defaults to `'mlp'` to
   * preserve the original harness behavior.
   */
  enemy?: { kind: 'swarm' | 'mlp' };
}

/**
 * Result emitted by one main-agent generation.
 *
 * Extends the raw {@link CombatQualitySignal} with the champion genome produced
 * by the NGE main-agent pipeline and the enemy snapshot the generation was
 * evaluated against.
 */
export interface MainGenerationResult extends CombatQualitySignal {
  /** Champion main-agent genome built by the NGE pipeline. */
  championGenome: NgeMainAgentEmbryo;
  /** Frozen enemy snapshot the champion was evaluated against. */
  evaluatedEnemySnapshot: Snapshot;
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
): MainGenerationResult {
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

  // Step 6: Return the champion's raw quality signal plus NGE harness contract fields.
  return {
    ...champion.signal,
    championGenome: champion.variant.genome as unknown as NgeMainAgentEmbryo,
    evaluatedEnemySnapshot: enemySnapshot,
  };
}

/**
 * Resolve the enemy snapshot for the current generation.
 *
 * When the caller supplies a snapshot it is used directly. Otherwise a fresh
 * enemy population is generated from the generation seed and advanced to the
 * current generation. The `update` method on each backend internally gates
 * refresh cadence, so the returned snapshot is always the correct one for the
 * requested generation. Defaults to the MLP backend unless `enemy.kind` is
 * `'swarm'`.
 *
 * @param options - Runner options.
 * @returns Enemy snapshot for the episode roster.
 */
function resolveEnemySnapshot(options: RunMainGenerationOptions): Snapshot {
  if (options.enemySnapshot) {
    return options.enemySnapshot;
  }

  const kind = options.enemy?.kind ?? 'mlp';
  if (kind === 'swarm') {
    return createSwarmEnemyPopulation({ seed: options.seed }).update({
      generation: options.generation,
    });
  }

  return createMlpEnemyPopulation({ seed: options.seed }).update({
    generation: options.generation,
  });
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
 * Tier-capped topology budget for the Neatenstein main agent.
 *
 * Kept local to this module to avoid a circular dependency with the harness
 * entry point while still honoring the same caps.
 */
const NEATENSTEIN_MAIN_AGENT_TIER_BUDGET = {
  maxNodes: 64,
  maxEdges: 256,
} as const;

/**
 * Build a deterministic NGE main-agent embryo for a variant.
 *
 * The embryo is produced by the real NGE main-agent pipeline and carries the
 * allowlisted motif archetypes (AttentionHead, GatedRecurrentCell, EpisodicSlot)
 * within the configured tier budget. The same `(seed, generation, variantId)`
 * tuple always produces the same embryo.
 *
 * @param seed - Generation seed.
 * @param generation - Generation number.
 * @param variantId - Stable variant index.
 * @returns A real NGE main-agent embryo genome.
 */
function createMainGenome(
  seed: number,
  generation: number,
  variantId: number,
): Genome {
  const config: NgeMainAgentLifecycleConfig = {
    seed: seed + variantId,
    maxNodes: NEATENSTEIN_MAIN_AGENT_TIER_BUDGET.maxNodes,
    maxEdges: NEATENSTEIN_MAIN_AGENT_TIER_BUDGET.maxEdges,
  };

  const embryo = buildMainAgentEmbryo(config);

  // The harness {@link Genome} type is a shallow snapshot; the embryo is cast
  // here because the runner now drives selection from the NGE topology fields.
  return embryo as unknown as Genome;
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
 * The genome is always a real NGE main-agent embryo produced by
 * {@link buildMainAgentEmbryo}, so complexity is read directly from the
 * `nodeCount` and `edgeCount` fields.
 *
 * @param variant - Main-agent variant.
 * @returns Neuron plus synapse count.
 */
function countComplexity(variant: MainVariant): number {
  const genome = variant.genome as unknown as Pick<
    NgeMainAgentEmbryo,
    'nodeCount' | 'edgeCount'
  >;

  return genome.nodeCount + genome.edgeCount;
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
