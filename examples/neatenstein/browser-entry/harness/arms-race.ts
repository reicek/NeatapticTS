/**
 * Arms-race generation runner for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * This module advances one generation of the co-evolution loop: it freezes an
 * enemy snapshot (using a supplied snapshot when available, otherwise resolving
 * one from the MLP enemy backend), selects a deterministic main-agent champion,
 * and evaluates the champion against the frozen snapshot using the existing
 * main-runner episode pipeline. The result is fully replay-safe from the
 * `(seed, generation, enemySnapshot)` tuple.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import { createMlpEnemyPopulation } from './enemy-mlp';
import { runMainGeneration } from './main-runner';
import { shouldRefreshMlpSnapshot } from './snapshot';
import type {
  CombatQualitySignal,
  Genome,
  MainVariant,
  MlpSnapshot,
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
  /** Optional frozen enemy snapshot; when omitted the MLP backend supplies one. */
  enemySnapshot?: Snapshot;
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
 * @param options - Generation configuration.
 * @returns The frozen generation state plus the champion's quality signal.
 *
 * @example
 * ```ts
 * const result = runArmsRaceGeneration({
 *   seed: 1,
 *   generation: 1,
 *   enemySnapshot: { kind: 'mlp', weights: new Float32Array(80) },
 * });
 * console.log(result.generation, result.quality.survivalTicks);
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

  // Step 4: Return the advanced generation state.
  return {
    generation: options.generation + 1,
    mainSnapshot,
    enemySnapshot,
    quality,
  };
}

/**
 * Resolve the frozen enemy snapshot for the requested generation.
 *
 * A caller-supplied snapshot always takes precedence. Otherwise the MLP enemy
 * backend provides a snapshot that is refreshed on MLP refresh boundaries.
 *
 * @param options - Generation configuration.
 * @returns Frozen enemy snapshot for evaluation.
 */
function resolveEnemySnapshot(options: RunArmsRaceGenerationOptions): Snapshot {
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
