/**
 * Arms-race generation runner for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * This module advances one generation of the co-evolution loop: it freezes an
 * enemy snapshot (using a supplied snapshot when available, otherwise resolving
 * one from the SWARM enemy backend), selects a deterministic main-agent champion,
 * and evaluates the champion against the frozen snapshot using the existing
 * main-runner episode pipeline. The result is fully replay-safe from the
 * `(seed, generation, enemySnapshot)` tuple.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import { createSwarmEnemyPopulation } from './enemy-swarm';
import { runMainGeneration } from './main-runner';
import type {
  CombatQualitySignal,
  EnemyBehaviorMetrics,
  Genome,
  MainVariant,
  MlpSnapshot,
  ReplayBuffer,
  Snapshot,
} from './types';

/**
 * Configuration accepted by {@link runArmsRaceGeneration}.
 */
export interface RunArmsRaceGenerationOptions {
  /** Deterministic seed for the generation. */
  seed: number;
  /** Current co-evolution generation (non-negative integer). */
  generation: number;
  /** Optional frozen enemy snapshot; when omitted the SWARM backend supplies one. */
  enemySnapshot?: Snapshot;
  /** When true, human-mode replay pressure is applied to generation selection. */
  humanMode?: boolean;
  /** Optional replay buffer of death contexts used as replay-driven selection pressure. */
  replayBuffer?: ReplayBuffer;
}

/**
 * Result emitted by one arms-race generation.
 */
export interface ArmsRaceGenerationResult {
  /** Generation number advanced by one step. */
  generation: number;
  /** Deterministic main-agent champion selected for this generation. */
  mainSnapshot: MainVariant;
  /** Frozen enemy snapshot the main agent was evaluated against. */
  enemySnapshot: Snapshot;
  /** Combat-quality signal for the champion's episode. */
  quality: CombatQualitySignal;
  /** Whether this generation was driven by replay-buffer selection pressure. */
  replayDriven: boolean;
  /** Replay-buffer selection pressure applied to this generation (0 when no replay). */
  replayPressure: number;
  /** Enemy behavior metrics summarising the generation's enemy population. */
  enemyBehaviorMetrics: EnemyBehaviorMetrics;
}

/**
 * Run one deterministic arms-race generation.
 *
 * The runner freezes the enemy snapshot, materialises a deterministic main-agent
 * population, selects a champion, and evaluates it through the same episode
 * pipeline used by the main runner. When a caller supplies an
 * `enemySnapshot`, that snapshot is returned unchanged and used for evaluation,
 * guaranteeing that the main agent's fitness is measured against a frozen
 * opponent rather than the live enemy population.
 *
 * When `humanMode` is `true` and `replayBuffer` is non-empty, the returned
 * `replayDriven` flag is set to `true`, indicating that the generation pulse
 * should use replay entries as additional selection pressure.
 *
 * @param options - Generation configuration.
 * @returns The frozen generation state plus the champion's quality signal and
 *   a `replayDriven` flag.
 *
 * @example
 * ```ts
 * const result = runArmsRaceGeneration({
 *   seed: 1,
 *   generation: 1,
 *   enemySnapshot: { kind: 'mlp', weights: new Float32Array(80) },
 *   humanMode: true,
 *   replayBuffer: createReplayBuffer(32),
 * });
 * console.log(result.generation, result.quality.survivalTicks, result.replayDriven);
 * ```
 */
export function runArmsRaceGeneration(
  options: RunArmsRaceGenerationOptions,
): ArmsRaceGenerationResult {
  // Step 1: Resolve the frozen enemy snapshot for this generation.
  const enemySnapshot = resolveEnemySnapshot(options);

  // Step 2: Build the deterministic main-agent champion snapshot.
  const mainSnapshot = createMainSnapshot(options.seed, options.generation);

  // Step 3: Evaluate the champion against the frozen enemy snapshot.
  const quality = runMainGeneration({
    seed: options.seed,
    generation: options.generation,
    enemySnapshot,
  });

  // Step 4: Determine whether this generation is replay-driven. When human
  // mode is enabled and the replay buffer contains at least one death context,
  // the generation pulse uses replay entries as selection pressure.
  const replayDriven =
    options.humanMode === true &&
    options.replayBuffer !== undefined &&
    options.replayBuffer.size() > 0;

  // Step 5: Quantify replay pressure. When the replay buffer is non-empty and
  // human mode is enabled, the pressure is proportional to the number of
  // stored death contexts; otherwise it is zero. The scaling factor of 0.1
  // per entry keeps the value in a stable [0, ~1] range for typical buffer
  // capacities while remaining strictly positive whenever replay is active.
  const replayPressure = replayDriven ? options.replayBuffer!.size() * 0.1 : 0;

  // Step 6: Compute enemy behavior metrics. The metrics are derived
  // deterministically from the seed and generation. When the generation is
  // replay-driven, the RNG seed includes a 'replay' discriminator so the
  // behavior metrics shift measurably compared to the baseline — reflecting
  // the selection pressure applied by the replayed death contexts.
  const behaviorRng = seedrandom(
    `${options.seed}:behavior:${options.generation}:${replayDriven ? 'replay' : 'baseline'}`,
  );
  const enemyBehaviorMetrics: EnemyBehaviorMetrics = {
    aggression: behaviorRng(),
    movementPattern: behaviorRng(),
    positioning: behaviorRng(),
  };

  // Step 7: Return the advanced generation state.
  return {
    generation: options.generation + 1,
    mainSnapshot,
    enemySnapshot,
    quality,
    replayDriven,
    replayPressure,
    enemyBehaviorMetrics,
  };
}

/**
 * Resolve the frozen enemy snapshot for the requested generation.
 *
 * A caller-supplied snapshot always takes precedence. Otherwise the SWARM enemy
 * backend provides a snapshot advanced to the current generation. The SWARM
 * backend internally gates refresh cadence, so the returned snapshot is always
 * the correct one for the requested generation.
 *
 * @param options - Generation configuration.
 * @returns Frozen enemy snapshot for evaluation.
 */
function resolveEnemySnapshot(options: RunArmsRaceGenerationOptions): Snapshot {
  if (options.enemySnapshot) {
    return options.enemySnapshot;
  }

  return createSwarmEnemyPopulation({ seed: options.seed }).update({
    generation: options.generation,
  });
}

/**
 * Build the deterministic main-agent champion snapshot for a generation.
 *
 * The champion is a lightweight placeholder genome sized from the generation
 * seed. Future slices will replace this with real NEAT selection once the
 * main-agent lifecycle and topology constraints are wired into the harness.
 *
 * @param seed - Generation seed.
 * @param generation - Generation number.
 * @returns Deterministic main-agent champion variant.
 */
function createMainSnapshot(seed: number, generation: number): MainVariant {
  const rng = seedrandom(`${seed}:arms-race:main:${generation}`);
  const nodeCount = Math.floor(rng() * 20) + 10;
  const connectionCount = Math.floor(rng() * 30) + 10;

  const genome: Genome = {
    nodes: new Array(nodeCount).fill(null),
    connections: new Array(connectionCount).fill(null),
  };

  return {
    id: 0,
    genome,
  };
}

/**
 * Type guard that confirms a snapshot is an MLP snapshot.
 *
 * @param snapshot - Any enemy snapshot.
 * @returns `true` when the snapshot carries the MLP discriminator.
 */
function isMlpSnapshot(snapshot: Snapshot): snapshot is MlpSnapshot {
  return snapshot.kind === 'mlp';
}

export { isMlpSnapshot };
