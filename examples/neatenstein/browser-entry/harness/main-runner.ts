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

import { createMlpEnemyPopulation } from './enemy-mlp';
import { createSwarmEnemyPopulation } from './enemy-swarm';
import { computeCombatQualitySignal } from './fitness';
import { createSeedPack } from './seed-pack';
import { selectVariant } from './select';
import { hashSeed } from './hash-seed';
import type {
  NgeMainAgentEmbryo,
  NgeMainAgentLifecycleConfig,
} from '../../../../src/neat/nge-main-agent/neat.nge-main-agent.types';
import { buildMainAgentEmbryo } from '../../../../src/neat/nge-main-agent/neat.nge-main-agent.embryo';
import Network from '../../../../src/architecture/network/network';

import { createEpisode, endEpisode } from '../host/game/episode';
import { gameTick } from '../host/game/tick';
import type { GameTickInputSnapshot } from '../host/game/tick';
import {
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS,
  NEATENSTEIN_MAIN_VARIANT_COUNT,
} from './constants';
import { extractCombatQualitySignal } from './fitness';
import { buildNeatensteinMap, createCollisionMap } from '../renderer/map';
import { NEATENSTEIN_MAP_SIZE } from '../constants';
import { extractSensors } from '../../scripts/enemy-navigation';

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
  const seedPack = createSeedPack({
    generation: options.generation,
    seed: options.seed,
  });

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
 * Number of main-agent NEAT network inputs (sensor vector length).
 *
 * Mirrors the worker-side constant so the headless evaluation uses the same
 * sensor layout as the live auto-mode controller.
 */
const NEATENSTEIN_MAIN_NEAT_INPUTS = 12;

/**
 * Number of main-agent NEAT network outputs (action vector length).
 *
 * Mirrors the worker-side constant so the headless evaluation produces the
 * same tick-input mapping as the live auto-mode controller.
 */
const NEATENSTEIN_MAIN_NEAT_OUTPUTS = 5;

/**
 * Maximum look-delta per tick for headless evaluation NEAT controller output.
 *
 * @see AC-066
 */
const NEATENSTEIN_MAIN_NEAT_MAX_TURN_RATE = Math.PI / 4;

/**
 * Map a raw NEAT network output vector to a {@link GameTickInputSnapshot}.
 *
 * Mirrors the worker-side `networkOutputToTickInput` mapping so the headless
 * evaluation and the live auto-mode controller produce identical tick inputs
 * from the same network output.
 *
 * @param outputs - Raw activation output from the NEAT network.
 * @returns A game-tick input snapshot for {@link gameTick}.
 */
function networkOutputToTickInput(outputs: number[]): GameTickInputSnapshot {
  const out =
    outputs.length >= NEATENSTEIN_MAIN_NEAT_OUTPUTS
      ? outputs
      : [
          ...outputs,
          ...new Array<number>(
            NEATENSTEIN_MAIN_NEAT_OUTPUTS - outputs.length,
          ).fill(0),
        ];

  return {
    move: {
      x: Math.tanh(out[0]),
      y: Math.tanh(out[1]),
    },
    lookDelta: Math.tanh(out[2]) * NEATENSTEIN_MAIN_NEAT_MAX_TURN_RATE,
    fire: out[3] > 0,
    dash: out[4] > 0.5,
  };
}

/**
 * Run one deterministic headless episode for a main-agent variant.
 *
 * Creates a real game episode from the episode seed, constructs a deterministic
 * NEAT network for the variant, and drives the player through the full game
 * tick pipeline for {@link NEATENSTEIN_MAX_EPISODE_TICKS} ticks. The resulting
 * {@link CombatQualitySignal} is extracted from the final game state.
 *
 * @param variant - Main-agent variant being evaluated.
 * @param enemySnapshot - Frozen enemy snapshot (reserved for future enemy AI
 *   integration; not used in P5S1).
 * @param episodeSeed - Deterministic seed for this episode.
 * @returns Combat-quality signal summarising the episode.
 */
function runEpisode(
  variant: MainVariant,
  _enemySnapshot: Snapshot,
  episodeSeed: number,
): CombatQualitySignal {
  const episode = createEpisode({ seed: episodeSeed });
  let state = episode.state;
  const flatMap = buildNeatensteinMap(state.seed);
  const collisionMap = createCollisionMap(flatMap, NEATENSTEIN_MAP_SIZE);
  const network = new Network(
    NEATENSTEIN_MAIN_NEAT_INPUTS,
    NEATENSTEIN_MAIN_NEAT_OUTPUTS,
    { seed: hashSeed(episodeSeed, variant.id) },
  );

  for (let tick = 0; tick < NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS; tick += 1) {
    const sensors = extractSensors(state, flatMap, NEATENSTEIN_MAP_SIZE);
    const raw = network.activate(sensors);
    const out: number[] = Array.isArray(raw)
      ? raw.map((v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0))
      : new Array<number>(NEATENSTEIN_MAIN_NEAT_OUTPUTS).fill(0);
    const tickInput = networkOutputToTickInput(out);
    state = gameTick(
      state,
      tickInput,
      collisionMap,
      NEATENSTEIN_FIXED_TIMESTEP_MS,
    );
  }

  const finalState = endEpisode(state);
  const telemetry = finalState.telemetry ?? {
    damageDealt: 0,
    shotsFired: 0,
    shotsHit: 0,
    aimMissRate: 0,
  };

  return extractCombatQualitySignal(finalState, telemetry);
}
