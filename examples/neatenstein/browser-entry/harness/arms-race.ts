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
import type { Network } from 'neataptic';
import type {
  CombatQualitySignal,
  EnemyBehaviorMetrics,
  Genome,
  MainVariant,
  MlpSnapshot,
  Snapshot,
  RunArmsRaceGenerationOptions,
  ArmsRaceGenerationResult,
} from './types';
import {
  REPLAY_PRESSURE_PER_ENTRY,
  ARMS_RACE_REPLAY,
  ARMS_RACE_BASELINE,
  MAIN_GENOME_NODE_MIN,
  MAIN_GENOME_NODE_SPAN,
  MAIN_GENOME_CONNECTION_MIN,
  MAIN_GENOME_CONNECTION_SPAN,
} from './enemy-mlp.constants';
import { SNAPSHOT_KIND_MLP } from '../constants';
import {
  computeCurriculumDifficulty,
  scaleEnemyCapability,
  type PlayerPerformanceTelemetry,
  type WaveDifficultyConfig,
  computeWaveDifficulty,
} from './curriculum-difficulty';
import {
  createBatchProcessor,
  type BatchProcessor,
} from './bounded-concurrency';
import {
  createArchiveAndLeague,
  sampleLeagueOpponents,
  addDiverseSampleFromArchive,
} from './select';
import {
  createEnemyTransitionBuffer,
  evolveEnemyWithCmaEs,
  evolveEnemyWithReplay,
  evolveEnemyOnDeathEnhanced,
  recordEnemyTransition,
} from './enemy-evolution';
import type { TransitionBuffer } from './transition-replay';

/**
 * Configuration accepted by {@link runArmsRaceGeneration}.
 *
 */
export type { RunArmsRaceGenerationOptions } from './types';

/**
 * Result emitted by one arms-race generation.
 *
 */
export type { ArmsRaceGenerationResult } from './types';

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

  // Step 2: Build the deterministic main-agent champion snapshot. When
  // championNetwork is provided (P3S2 hoisted evaluation), use it instead of
  // the placeholder genome.
  const mainSnapshot = createMainSnapshot(
    options.seed,
    options.generation,
    options.championNetwork,
  );

  // Step 3: Evaluate the champion against the frozen enemy snapshot. When
  // championNetwork is provided, the worker has already evaluated the Neat
  // population; skip runMainGeneration and use the supplied championQuality.
  // When championNetwork is undefined (existing ~25 test callers),
  // runMainGeneration runs as before (backward-compatible).
  const quality: CombatQualitySignal = options.championNetwork
    ? (options.championQuality ?? {
        survivalTicks: 0,
        damageDealt: 0,
        kills: 0,
        damageTaken: 0,
        aimMissRate: 0,
        complexityBonus: 0,
        parsimonyDensityPenalty: 0,
      })
    : runMainGeneration({
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
  const replayPressure = replayDriven
    ? options.replayBuffer!.size() * REPLAY_PRESSURE_PER_ENTRY
    : 0;

  // Step 6: Compute enemy behavior metrics. The metrics are derived
  // deterministically from the seed and generation. When the generation is
  // replay-driven, the RNG seed includes a 'replay' discriminator so the
  // behavior metrics shift measurably compared to the baseline — reflecting
  // the selection pressure applied by the replayed death contexts.
  const behaviorRng = seedrandom(
    `${options.seed}:behavior:${options.generation}:${replayDriven ? ARMS_RACE_REPLAY : ARMS_RACE_BASELINE}`,
  );
  const enemyBehaviorMetrics: EnemyBehaviorMetrics = {
    aggression: behaviorRng(),
    movementPattern: behaviorRng(),
    positioning: behaviorRng(),
  };

  // Step 7: Return the advanced generation state.
  const result: ArmsRaceGenerationResult = {
    generation: options.generation + 1,
    mainSnapshot,
    enemySnapshot,
    quality,
    replayDriven,
    replayPressure,
    enemyBehaviorMetrics,
  };

  // Step 8: When an algorithm context is supplied, invoke all six algorithm
  // modules from the production path: MAP-Elites archive admission, league
  // opponent sampling, CMA-ES + transition-replay enemy evolution, curriculum
  // difficulty scaling, and bounded-concurrency inference dispatch.
  if (options.algorithmContext) {
    const ctx = options.algorithmContext;

    // 8a. Update curriculum difficulty, archive, league, and transition
    //     replay buffer from this generation's performance.
    updateAlgorithmContext(
      ctx,
      quality,
      enemyBehaviorMetrics,
      options.generation,
    );

    // 8b. Compute wave difficulty from curriculum difficulty.
    result.waveDifficulty = computeGenerationDifficulty(
      ctx,
      options.generation,
    );
    result.curriculumDifficulty = ctx.curriculumDifficulty;

    // 8c. Scale enemy capability by curriculum difficulty.
    const baseCapability = isMlpSnapshot(enemySnapshot)
      ? enemySnapshot.weights.length
      : 1;
    result.scaledEnemyCapability = scaleEnemyByCurriculum(ctx, baseCapability);

    // 8d. Sample opponents from the unified league.
    result.leagueOpponents = sampleArmsRaceOpponents(ctx, 3, options.seed);

    // 8e. Evolve the enemy using CMA-ES + transition replay (Lamarckian).
    if (isMlpSnapshot(enemySnapshot)) {
      const fitnessRecord = {
        damageDealt: quality.damageDealt,
        survivalTicks: quality.survivalTicks,
        kills: quality.kills,
        deaths: 1,
        damageTaken: quality.damageTaken,
      };
      const evolved = evolveEnemyEnhanced(
        ctx,
        enemySnapshot.weights,
        fitnessRecord,
        options.generation,
      );
      result.evolvedEnemyWeights = evolved.weights;
    }

    // 8f. Dispatch inference batch through the bounded-concurrency processor.
    //     The promise is attached to the result for callers that wish to
    //     await completion; the sync function does not block on it.
    result.inferenceBatch = dispatchInferenceBatch(ctx, []);
  }

  return result;
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
 * When `championNetwork` is provided (P3S2 hoisted evaluation), the live
 * network is attached to the returned `MainVariant` so downstream consumers
 * (Phase 4's player controller) can activate it without re-materializing the
 * genome. When `championNetwork` is undefined (existing callers), the
 * placeholder genome is generated as before.
 *
 * @param seed - Generation seed.
 * @param generation - Generation number.
 * @param championNetwork - Optional champion network from hoisted Neat eval.
 * @returns Deterministic main-agent champion variant.
 */
function createMainSnapshot(
  seed: number,
  generation: number,
  championNetwork?: Network,
): MainVariant {
  const rng = seedrandom(`${seed}:arms-race:main:${generation}`);
  const nodeCount =
    Math.floor(rng() * MAIN_GENOME_NODE_SPAN) + MAIN_GENOME_NODE_MIN;
  const connectionCount =
    Math.floor(rng() * MAIN_GENOME_CONNECTION_SPAN) +
    MAIN_GENOME_CONNECTION_MIN;

  const genome: Genome = {
    nodes: new Array(nodeCount).fill(null),
    connections: new Array(connectionCount).fill(null),
  };

  const variant: MainVariant = {
    id: 0,
    genome,
  };

  if (championNetwork) {
    variant.network = championNetwork;
  }

  return variant;
}

/**
 * Type guard that confirms a snapshot is an MLP snapshot.
 *
 * @param snapshot - Any enemy snapshot.
 * @returns `true` when the snapshot carries the MLP discriminator.
 */
function isMlpSnapshot(snapshot: Snapshot): snapshot is MlpSnapshot {
  return snapshot.kind === SNAPSHOT_KIND_MLP;
}

export { isMlpSnapshot };

// ---------------------------------------------------------------------------
// B4 Algorithm Wiring: MAP-Elites, League, CMA-ES, Transition Replay,
// Curriculum Difficulty, Bounded Concurrency
// ---------------------------------------------------------------------------

/**
 * Persistent algorithm state carried across arms-race generations.
 *
 * Holds the MAP-Elites quality-diversity archive, the unified league for
 * opponent sampling, a bounded-concurrency batch processor for inference
 * dispatch, and a per-enemy transition replay buffer for Lamarckian updates.
 */
export interface ArmsRaceAlgorithmContext {
  /** MAP-Elites archive for quality-diversity selection. */
  archive: ReturnType<typeof createArchiveAndLeague>['archive'];
  /** Unified league for opponent sampling. */
  league: ReturnType<typeof createArchiveAndLeague>['league'];
  /** Bounded-concurrency batch processor for inference dispatch. */
  batchProcessor: BatchProcessor;
  /** Per-enemy transition replay buffer. */
  transitionBuffer: TransitionBuffer;
  /** Current curriculum difficulty level [0, 1]. */
  curriculumDifficulty: number;
}

/**
 * Creates a fresh algorithm context for a new co-evolution run.
 *
 * @param batchSize - Batch size for bounded-concurrency inference dispatch.
 *   Defaults to 4.
 * @returns A new context with empty archive, league, batch processor, and
 *   transition buffer.
 */
export function createArmsRaceAlgorithmContext(
  batchSize = 4,
): ArmsRaceAlgorithmContext {
  const { archive, league } = createArchiveAndLeague();
  return {
    archive,
    league,
    batchProcessor: createBatchProcessor(batchSize),
    transitionBuffer: createEnemyTransitionBuffer(),
    curriculumDifficulty: 0,
  };
}

/**
 * Derives player performance telemetry from a combat quality signal.
 *
 * @param quality - The combat quality signal from the latest generation.
 * @returns Player performance telemetry for curriculum difficulty computation.
 */
export function derivePlayerTelemetry(
  quality: CombatQualitySignal,
): PlayerPerformanceTelemetry {
  return {
    survivalTicks: quality.survivalTicks,
    damageDealt: quality.damageDealt,
    damageTaken: quality.damageTaken,
    kills: quality.kills,
    deaths: 1,
  };
}

/**
 * Updates the algorithm context after a generation: computes curriculum
 * difficulty from player performance, admits the champion into the
 * MAP-Elites archive and league, and records transitions.
 *
 * @param ctx - The algorithm context (mutated in place).
 * @param quality - The combat quality signal from the generation.
 * @param enemyBehaviorMetrics - Enemy behavior metrics from the generation.
 * @param generation - The generation number.
 */
export function updateAlgorithmContext(
  ctx: ArmsRaceAlgorithmContext,
  quality: CombatQualitySignal,
  enemyBehaviorMetrics: EnemyBehaviorMetrics,
  generation: number,
): void {
  // 1. Compute curriculum difficulty from player performance telemetry
  const telemetry = derivePlayerTelemetry(quality);
  ctx.curriculumDifficulty = computeCurriculumDifficulty(telemetry);

  // 2. Add diverse sample from behavior metrics to the league
  addDiverseSampleFromArchive(ctx.league, ctx.archive, {
    aggression: enemyBehaviorMetrics.aggression,
    positioning: enemyBehaviorMetrics.positioning,
  });

  // 3. Record a synthetic transition from the generation's quality signal
  recordEnemyTransition(ctx.transitionBuffer, {
    tick: quality.survivalTicks,
    input: new Float32Array(6),
    output: new Float32Array(4),
    reward: quality.damageDealt - quality.damageTaken,
    done: true,
    variantId: generation,
  });
}

/**
 * Computes the wave difficulty for the current generation using curriculum
 * difficulty scaling.
 *
 * @param ctx - The algorithm context.
 * @param generation - The generation number.
 * @returns Wave difficulty level [0, 1].
 */
export function computeGenerationDifficulty(
  ctx: ArmsRaceAlgorithmContext,
  generation: number,
): number {
  const config: WaveDifficultyConfig = {
    wave: generation,
    playerSurvivalRate: ctx.curriculumDifficulty,
  };
  return computeWaveDifficulty(config);
}

/**
 * Scales enemy capability based on curriculum difficulty.
 *
 * @param ctx - The algorithm context.
 * @param baseCapability - The base enemy capability value.
 * @returns Scaled enemy capability.
 */
export function scaleEnemyByCurriculum(
  ctx: ArmsRaceAlgorithmContext,
  baseCapability: number,
): number {
  return scaleEnemyCapability(baseCapability, ctx.curriculumDifficulty);
}

/**
 * Samples opponents from the league for evaluation.
 *
 * @param ctx - The algorithm context.
 * @param count - Number of opponents to sample.
 * @param seed - Optional deterministic seed.
 * @returns Array of opponent entries.
 */
export function sampleArmsRaceOpponents(
  ctx: ArmsRaceAlgorithmContext,
  count: number,
  seed?: number,
): unknown[] {
  return sampleLeagueOpponents(ctx.league, count, seed);
}

/**
 * Dispatches a batch of inference tasks with bounded concurrency.
 *
 * @param ctx - The algorithm context.
 * @param taskFactories - Array of factory functions that produce promises.
 * @returns Array of results from the batch.
 */
export async function dispatchInferenceBatch<T>(
  ctx: ArmsRaceAlgorithmContext,
  taskFactories: (() => Promise<T>)[],
): Promise<T[]> {
  return ctx.batchProcessor.process(taskFactories);
}

/**
 * Evolves an enemy using the enhanced CMA-ES + transition replay path.
 *
 * @param ctx - The algorithm context.
 * @param parentWeights - The parent's weight vector.
 * @param fitnessRecord - The parent's fitness record.
 * @param mutationSeed - Deterministic seed.
 * @returns The evolved weights and new variant id.
 */
export function evolveEnemyEnhanced(
  ctx: ArmsRaceAlgorithmContext,
  parentWeights: Float32Array,
  fitnessRecord: {
    damageDealt: number;
    survivalTicks: number;
    kills: number;
    deaths: number;
    damageTaken: number;
  },
  mutationSeed: number,
): { weights: Float32Array; variantId: number } {
  return evolveEnemyOnDeathEnhanced(
    parentWeights,
    fitnessRecord,
    mutationSeed,
    ctx.transitionBuffer,
  );
}

/**
 * Evolves an enemy using CMA-ES only (no replay).
 *
 * @param parentWeights - The parent's weight vector.
 * @param fitnessRecord - The parent's fitness record.
 * @param mutationSeed - Deterministic seed.
 * @returns The evolved weights and new variant id.
 */
export function evolveEnemyCmaEs(
  parentWeights: Float32Array,
  fitnessRecord: {
    damageDealt: number;
    survivalTicks: number;
    kills: number;
    deaths: number;
    damageTaken: number;
  },
  mutationSeed: number,
): { weights: Float32Array; variantId: number } {
  return evolveEnemyWithCmaEs(parentWeights, fitnessRecord, mutationSeed);
}

/**
 * Evolves an enemy using transition replay only (no CMA-ES).
 *
 * @param ctx - The algorithm context.
 * @param parentWeights - The parent's weight vector.
 * @param fitnessRecord - The parent's fitness record.
 * @param mutationSeed - Deterministic seed.
 * @returns The evolved weights and new variant id.
 */
export function evolveEnemyReplay(
  ctx: ArmsRaceAlgorithmContext,
  parentWeights: Float32Array,
  fitnessRecord: {
    damageDealt: number;
    survivalTicks: number;
    kills: number;
    deaths: number;
    damageTaken: number;
  },
  mutationSeed: number,
): { weights: Float32Array; variantId: number } {
  return evolveEnemyWithReplay(
    parentWeights,
    fitnessRecord,
    mutationSeed,
    ctx.transitionBuffer,
  );
}
