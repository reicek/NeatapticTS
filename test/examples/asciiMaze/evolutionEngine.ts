// Handles the main NEAT evolution loop for maze solving
// Exports: EvolutionEngine class with static methods

import { createEngineState, EngineState } from './evolutionEngine/engineState';
import {
  clearDeterministicMode,
  getProfilingAccumulators,
  isProfilingDetailsEnabled,
  resolveRngParameters,
  setDeterministicMode,
} from './evolutionEngine/rngAndTiming';
import { ensureScratchCapacity } from './evolutionEngine/scratchPools';
import {
  applyCompassWarmStart,
  centerOutputBiases,
} from './evolutionEngine/populationPruning';
import {
  buildLamarckianTrainingSet,
  pretrainPopulationWarmStart,
  warmStartPopulationIfNeeded,
} from './evolutionEngine/trainingWarmStart';
import {
  normalizeRunOptions,
  prepareEnvironmentForRun,
  createAndSeedNeat,
} from './evolutionEngine/optionsAndSetup';
import {
  runEvolutionLoop,
  prepareLoopHelpers,
  emitProfileSummary,
} from './evolutionEngine/evolutionLoop';
import { printNetworkStructure } from './evolutionEngine/networkInspection';
import { INetwork, IRunMazeEvolutionOptions } from './interfaces';
import type Network from '../../../src/architecture/network';

/**
 * Runtime type for network node with dynamic properties.
 * Nodes may have type, connections, and other runtime-added fields.
 */
interface RuntimeNetworkNode {
  type?: string;
  connections?: {
    out?: RuntimeNetworkConnection[];
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

/**
 * Runtime type for network connection with dynamic properties.
 * Connections track from/to nodes and enabled state.
 */
interface RuntimeNetworkConnection {
  from?: RuntimeNetworkNode;
  to?: RuntimeNetworkNode;
  enabled?: boolean;
  [key: string]: unknown;
}

/**
 * Runtime type for evolution result with dynamic exit reason.
 * Results may include exitReason from simulation outcomes.
 */
interface RuntimeEvolutionResult {
  exitReason?: string;
  [key: string]: unknown;
}

/**
 * Runtime type for EvolutionEngine class with dynamic properties.
 * Engine may have _speciesHistory for telemetry tracking.
 */
interface RuntimeEvolutionEngine {
  _speciesHistory?: unknown[];
  [key: string]: unknown;
}

/**
 * EvolutionEngine: Thin façade for NEAT-based maze solving.
 *
 * This class provides a stable public API that delegates to specialized modules:
 * - Engine state management → `engineState.ts`
 * - RNG and profiling → `rngAndTiming.ts`
 * - Scratch buffer pools → `scratchPools.ts`
 * - Population dynamics → `populationDynamics.ts`
 * - Telemetry metrics → `telemetryMetrics.ts`
 * - Training/warm-start → `trainingWarmStart.ts`
 * - Options and setup → `optionsAndSetup.ts`
 * - Evolution loop → `evolutionLoop.ts`
 * - Network inspection → `networkInspection.ts`
 *
 * Public API (frozen entry points):
 * - `runMazeEvolution(options)`: Main entry point for maze solving
 * - `printNetworkStructure(network)`: Debug utility for network topology
 * - `setDeterministic(seed?)`: Enable deterministic mode
 * - `clearDeterministic()`: Disable deterministic mode
 *
 * Internal constants (private configuration values) are defined as static fields
 * and passed explicitly to module functions for transparent, testable orchestration.
 */
export class EvolutionEngine {
  /** Shared engine state instance backing all façade helpers. */
  static #STATE: EngineState = createEngineState();

  /** Reusable empty vector constant to avoid ephemeral allocations from `|| []` fallbacks. */
  static #EMPTY_VEC: unknown[] = [];

  /** Number of action outputs (N,E,S,W) */
  static #ACTION_DIM = 4;
  /** Adaptive logits ring capacity (power-of-two). */
  static #LOGITS_RING_CAP = 512;
  /** Max allowed ring capacity (safety bound). */
  static #LOGITS_RING_CAP_MAX = 8192;
  /** Indicates SharedArrayBuffer-backed ring is active. */
  static #LOGITS_RING_SHARED = false;
  /** Write cursor for non-shared ring. */
  static #SCRATCH_LOGITS_RING_W = 0;

  /**
   * Enable deterministic mode and optionally reseed the internal RNG via the shared state helpers.
   *
   * @param seed Optional numeric seed used to reseed the deterministic RNG sequence.
   */
  static setDeterministic(seed?: number): void {
    setDeterministicMode(EvolutionEngine.#STATE, seed);
  }

  /**
   * Disable deterministic mode and return to non-deterministic random number generation.
   */
  static clearDeterministic(): void {
    clearDeterministicMode(EvolutionEngine.#STATE);
  }
  /** Default tail history size used by telemetry */
  static #RECENT_WINDOW = 40;
  /** Default supervised training error threshold for local training */
  static #DEFAULT_TRAIN_ERROR = 0.01;
  /** Default supervised training learning rate for local training */
  static #DEFAULT_TRAIN_RATE = 0.001;
  /** Default supervised training momentum */
  static #DEFAULT_TRAIN_MOMENTUM = 0.2;
  /** Default small batch size used during Lamarckian training */
  static #DEFAULT_TRAIN_BATCH_SMALL = 2;
  /** Default batch size used when training the fittest network for evaluation */
  static #DEFAULT_TRAIN_BATCH_LARGE = 20;
  /** Iterations used when training the fittest network for evaluation */
  static #FITTEST_TRAIN_ITERATIONS = 1000;
  /** Saturation fraction threshold triggering hidden-output pruning */
  static #SATURATION_PRUNE_THRESHOLD = 0.5;
  /** Default probability used for small randomized jitter (25%) */
  static #DEFAULT_JITTER_PROB = 0.25;
  /** Small std threshold to consider 'small' std */
  static #DEFAULT_STD_SMALL = 0.25;
  /** Multiplier applied when std is small */
  static #DEFAULT_STD_ADJUST_MULT = 0.7;
  /** High target probability for the chosen action during supervised warm start */
  static #TRAIN_OUT_PROB_HIGH = 0.92;
  /** Low target probability for non-chosen actions during supervised warm start */
  static #TRAIN_OUT_PROB_LOW = 0.02;
  /** Progress intensity: medium (single open path typical) */
  static #PROGRESS_MEDIUM = 0.7;
  /** Progress intensity: strong forward signal */
  static #PROGRESS_STRONG = 0.9;
  /** Progress intensity: typical junction neutrality */
  static #PROGRESS_JUNCTION = 0.6;
  /** Progress intensity: four-way moderate signal */
  static #PROGRESS_FOURWAY = 0.55;
  /** Progress intensity: regressing / weak progress */
  static #PROGRESS_REGRESS = 0.4;
  /** Progress intensity: mild regression / noise */
  static #PROGRESS_MILD_REGRESS = 0.45;
  /** Minimal progress positive blip used in a corner-case sample */
  static #PROGRESS_MIN_SIGNAL = 0.001;
  /** Augmentation: base openness jitter value */
  static #AUGMENT_JITTER_BASE = 0.95;
  /** Augmentation: openness jitter range added to base */
  static #AUGMENT_JITTER_RANGE = 0.05;
  /** Augmentation: probability to jitter progress channel */
  static #AUGMENT_PROGRESS_JITTER_PROB = 0.35;
  /** Augmentation: progress delta full range */
  static #AUGMENT_PROGRESS_DELTA_RANGE = 0.1;
  /** Augmentation: progress delta half range (range/2) */
  static #AUGMENT_PROGRESS_DELTA_HALF = 0.05;
  /** Max iterations used during population pretrain */
  static #PRETRAIN_MAX_ITER = 60;
  /** Base iterations added in pretrain (8 + floor(setLen/2)) */
  static #PRETRAIN_BASE_ITER = 8;
  /** Default learning rate used during pretraining population warm-start */
  static #DEFAULT_PRETRAIN_RATE = 0.002;
  /** Default momentum used during pretraining population warm-start */
  static #DEFAULT_PRETRAIN_MOMENTUM = 0.1;

  /**
   * Populate the engine's pooled node-index scratch buffer with indices of nodes matching `type`.
   * @internal - Small helper used by various engine methods; retained for internal use.
   */
  static #getNodeIndicesByType(nodes: RuntimeNetworkNode[] | undefined, type: string): number {
    if (!Array.isArray(nodes) || nodes.length === 0) return 0;
    let writeCount = 0;
    let scratch = EvolutionEngine.#STATE.scratch.nodeIndexBuffer;
    for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
      const nodeRef = nodes[nodeIndex];
      if (!nodeRef || nodeRef.type !== type) continue;
      if (writeCount >= scratch.length) {
        const nextCapacity = 1 << Math.ceil(Math.log2(writeCount + 1));
        const grown = new Int32Array(nextCapacity);
        grown.set(scratch);
        EvolutionEngine.#STATE.scratch.nodeIndexBuffer = grown;
        scratch = grown;
      }
      scratch[writeCount++] = nodeIndex;
    }
    return writeCount;
  }

  /**
   * Collect enabled outgoing connections from a hidden node terminating at output nodes.
   * @internal - Small helper used by network analysis methods; retained for internal use.
   */
  static #collectHiddenToOutputConns(
    hiddenNode: RuntimeNetworkNode,
    nodesRef: RuntimeNetworkNode[],
    outputCount: number,
  ): RuntimeNetworkConnection[] {
    if (
      !hiddenNode?.connections ||
      !Array.isArray(nodesRef) ||
      outputCount <= 0
    )
      return [];
    const maxScratch = EvolutionEngine.#STATE.scratch.nodeIndexBuffer.length;
    const effectiveOutputCount = Math.min(
      outputCount | 0,
      maxScratch,
      nodesRef.length,
    );
    if (effectiveOutputCount <= 0) return [];
    const hiddenOutBuffer =
      EvolutionEngine.#STATE.scratch.hiddenToOutputConnections;
    hiddenOutBuffer.length = 0;
    const outgoing = hiddenNode.connections.out ?? EvolutionEngine.#EMPTY_VEC;
    for (let outIndex = 0; outIndex < outgoing.length; outIndex++) {
      const candidate = outgoing[outIndex] as unknown as RuntimeNetworkConnection;
      if (!candidate || candidate.enabled === false) continue;
      for (
        let outputIndex = 0;
        outputIndex < effectiveOutputCount;
        outputIndex++
      ) {
        const nodeIdx =
          EvolutionEngine.#STATE.scratch.nodeIndexBuffer[outputIndex];
        const targetNode = nodesRef[nodeIdx];
        if (candidate.to === targetNode) {
          hiddenOutBuffer.push(candidate);
          break;
        }
      }
    }
    return hiddenOutBuffer;
  }

  /**
   * Run NEAT-based neuro-evolution to train an agent to solve an ASCII maze.
   *
   * This is the main entry point for the evolution process. It orchestrates:
   * 1. Option normalization and environment preparation
   * 2. NEAT instance creation and population seeding
   * 3. Optional Lamarckian warm-start (supervised pretraining)
   * 4. Full evolution loop with adaptive dynamics and telemetry
   * 5. Result packaging with best network and exit reason
   *
   * The engine delegates all heavy lifting to specialized modules, maintaining
   * a thin orchestration layer that passes explicit parameters (no hidden state).
   *
   * @param options - Configuration object specifying maze, evolution parameters,
   *                  telemetry toggles, and stop conditions.
   * @returns Promise resolving to an object containing:
   *  - `bestNetwork`: The highest-scoring evolved network (genome)
   *  - `bestResult`: Simulation result object (path, score, telemetry)
   *  - `neat`: The final NEAT instance (for inspection/continuation)
   *  - `exitReason`: String indicating why evolution stopped (e.g., 'solved', 'maxGenerations')
   *
   * @example
   * const result = await EvolutionEngine.runMazeEvolution({
   *   maze: myMazeString,
   *   maxGenerations: 100,
   *   popSize: 500,
   *   deterministicSeed: 42,
   * });
   * console.log(`Best score: ${result.bestResult.score}`);
   * EvolutionEngine.printNetworkStructure(result.bestNetwork);
   */
  static async runMazeEvolution(options: IRunMazeEvolutionOptions) {
    // 1) Normalise and validate options (descriptive names, defaulting).
    const opts = normalizeRunOptions(
      options,
      (seed: number) => EvolutionEngine.setDeterministic(seed),
      (enabled: boolean) => {
        EvolutionEngine.#STATE.toggles.reducedTelemetry = enabled;
      },
      (enabled: boolean) => {
        EvolutionEngine.#STATE.toggles.telemetryMinimal = enabled;
      },
      (disabled: boolean) => {
        EvolutionEngine.#STATE.toggles.disableBaldwinPhase = disabled;
      },
    );

    // 2) Prepare maze, encoded maps and fitness context. This reuses pooled buffers where possible.
    const {
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      inputSize,
      outputSize,
      fitnessContext,
    } = prepareEnvironmentForRun(opts, EvolutionEngine.#STATE.scratch);

    // 3) Create and seed NEAT instance via a descriptive helper.
    const { neat, scratchPopClone, scratchSample } = createAndSeedNeat(
      opts,
      inputSize,
      outputSize,
      fitnessContext,
      EvolutionEngine.#STATE.scratch.populationCloneBuffer,
      EvolutionEngine.#STATE.scratch.samplePool,
    );
    EvolutionEngine.#STATE.scratch.populationCloneBuffer = scratchPopClone;
    EvolutionEngine.#STATE.scratch.samplePool = scratchSample;

    // 4) Ensure internal scratch/pooling capacity is sufficient for the configured population & network sizes.
    ensureScratchCapacity(EvolutionEngine.#STATE, {
      populationSize: opts.popSize,
      inputSize,
      outputSize,
    });

    // 5) Lamarckian warm-start (pretrain generation 0) when training cases exist.
    const lamarckianTrainingSet = buildLamarckianTrainingSet(
      EvolutionEngine.#STATE,
      {
        TRAIN_OUT_PROB_HIGH: EvolutionEngine.#TRAIN_OUT_PROB_HIGH,
        TRAIN_OUT_PROB_LOW: EvolutionEngine.#TRAIN_OUT_PROB_LOW,
        PROGRESS_MEDIUM: EvolutionEngine.#PROGRESS_MEDIUM,
        PROGRESS_STRONG: EvolutionEngine.#PROGRESS_STRONG,
        PROGRESS_JUNCTION: EvolutionEngine.#PROGRESS_JUNCTION,
        PROGRESS_FOURWAY: EvolutionEngine.#PROGRESS_FOURWAY,
        PROGRESS_REGRESS: EvolutionEngine.#PROGRESS_REGRESS,
        PROGRESS_MIN_SIGNAL: EvolutionEngine.#PROGRESS_MIN_SIGNAL,
        PROGRESS_MILD_REGRESS: EvolutionEngine.#PROGRESS_MILD_REGRESS,
        DEFAULT_JITTER_PROB: EvolutionEngine.#DEFAULT_JITTER_PROB,
        AUGMENT_JITTER_BASE: EvolutionEngine.#AUGMENT_JITTER_BASE,
        AUGMENT_JITTER_RANGE: EvolutionEngine.#AUGMENT_JITTER_RANGE,
        AUGMENT_PROGRESS_JITTER_PROB:
          EvolutionEngine.#AUGMENT_PROGRESS_JITTER_PROB,
        AUGMENT_PROGRESS_DELTA_RANGE:
          EvolutionEngine.#AUGMENT_PROGRESS_DELTA_RANGE,
        AUGMENT_PROGRESS_DELTA_HALF:
          EvolutionEngine.#AUGMENT_PROGRESS_DELTA_HALF,
        RNG_PARAMETERS: resolveRngParameters(),
      },
    );
    warmStartPopulationIfNeeded(
      neat,
      lamarckianTrainingSet,
      EvolutionEngine.#STATE,
      (neatInstance, trainingSet) => {
        pretrainPopulationWarmStart(
          neatInstance,
          trainingSet,
          {
            PRETRAIN_MAX_ITER: EvolutionEngine.#PRETRAIN_MAX_ITER,
            PRETRAIN_BASE_ITER: EvolutionEngine.#PRETRAIN_BASE_ITER,
            DEFAULT_TRAIN_ERROR: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
            DEFAULT_PRETRAIN_RATE: EvolutionEngine.#DEFAULT_PRETRAIN_RATE,
            DEFAULT_PRETRAIN_MOMENTUM:
              EvolutionEngine.#DEFAULT_PRETRAIN_MOMENTUM,
            DEFAULT_TRAIN_BATCH_SMALL:
              EvolutionEngine.#DEFAULT_TRAIN_BATCH_SMALL,
          },
          applyCompassWarmStart,
          centerOutputBiases,
        );
      },
    );

    // 6) Prepare loop helpers and run the full evolution loop inside a private helper.
    const loopHelpers = prepareLoopHelpers(
      opts,
      EvolutionEngine.#STATE.scratch,
    );

    // Lightweight profiling (opt-in): set env ASCII_MAZE_PROFILE=1 to enable
    const doProfile = !!(
      typeof process !== 'undefined' &&
      typeof process.env !== 'undefined' &&
      process.env.ASCII_MAZE_PROFILE === '1'
    );

    const runResult = await runEvolutionLoop(
      EvolutionEngine.#STATE,
      neat,
      opts,
      lamarckianTrainingSet,
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      loopHelpers,
      doProfile,
      EvolutionEngine.#STATE.scratch.logitsRing,
      EvolutionEngine.#LOGITS_RING_CAP,
      EvolutionEngine.#LOGITS_RING_CAP_MAX,
      EvolutionEngine.#ACTION_DIM,
      EvolutionEngine.#LOGITS_RING_SHARED,
      EvolutionEngine.#STATE.scratch.sharedLogits,
      EvolutionEngine.#STATE.scratch.sharedLogitsWriteIndex,
      EvolutionEngine.#SCRATCH_LOGITS_RING_W,
      EvolutionEngine.#EMPTY_VEC as unknown as Network[],
      EvolutionEngine.#STATE.scratch.nodeIndexBuffer,
      EvolutionEngine.#STATE.scratch.snapshotReusableObject,
      EvolutionEngine.#STATE.scratch.snapshotTopEntries,
      EvolutionEngine.#getNodeIndicesByType,
      EvolutionEngine.#collectHiddenToOutputConns,
      {
        DEFAULT_TRAIN_ERROR: EvolutionEngine.#DEFAULT_TRAIN_ERROR,
        DEFAULT_TRAIN_RATE: EvolutionEngine.#DEFAULT_TRAIN_RATE,
        DEFAULT_TRAIN_MOMENTUM: EvolutionEngine.#DEFAULT_TRAIN_MOMENTUM,
        DEFAULT_TRAIN_BATCH_SMALL: EvolutionEngine.#DEFAULT_TRAIN_BATCH_SMALL,
        DEFAULT_TRAIN_BATCH_LARGE: EvolutionEngine.#DEFAULT_TRAIN_BATCH_LARGE,
        DEFAULT_STD_SMALL: EvolutionEngine.#DEFAULT_STD_SMALL,
        DEFAULT_STD_ADJUST_MULT: EvolutionEngine.#DEFAULT_STD_ADJUST_MULT,
        FITTEST_TRAIN_ITERATIONS: EvolutionEngine.#FITTEST_TRAIN_ITERATIONS,
        TELEMETRY_MINIMAL: EvolutionEngine.#STATE.toggles.telemetryMinimal,
        SATURATION_PRUNE_THRESHOLD: EvolutionEngine.#SATURATION_PRUNE_THRESHOLD,
        RECENT_WINDOW: EvolutionEngine.#RECENT_WINDOW,
        REDUCED_TELEMETRY: EvolutionEngine.#STATE.toggles.reducedTelemetry,
        DISABLE_BALDWIN: EvolutionEngine.#STATE.toggles.disableBaldwinPhase,
      },
      (EvolutionEngine as unknown as RuntimeEvolutionEngine)._speciesHistory as unknown as number[] ?? EvolutionEngine.#EMPTY_VEC as unknown as number[],
    );

    // Update ring state from loop result
    EvolutionEngine.#LOGITS_RING_CAP = runResult.updatedRingState.logitsRingCap;
    EvolutionEngine.#LOGITS_RING_SHARED =
      runResult.updatedRingState.logitsRingShared;
    EvolutionEngine.#SCRATCH_LOGITS_RING_W =
      runResult.updatedRingState.scratchLogitsRingW;

    // Unpack results from the loop helper
    const {
      bestNetwork,
      bestResult,
      completedGenerations,
      totalEvolveMs,
      totalLamarckMs,
      totalSimMs,
    } = runResult;

    // Emit profiling summary when enabled (use loopHelpers.safeWrite to avoid duplicating writer resolution)
    if (doProfile && completedGenerations > 0) {
      emitProfileSummary(
        EvolutionEngine.#STATE,
        loopHelpers.safeWrite,
        completedGenerations,
        totalEvolveMs,
        totalLamarckMs,
        totalSimMs,
        isProfilingDetailsEnabled,
        getProfilingAccumulators,
      );
    }

    // Final return: best network, its simulation result, the NEAT instance and exit reason
    return {
      bestNetwork,
      bestResult,
      neat,
      exitReason: (bestResult as unknown as RuntimeEvolutionResult).exitReason ?? 'incomplete',
    };
  }

  /**
   * Print a concise, human-readable summary of a network's topology.
   *
   * Delegates to the `networkInspection` module for detailed analysis.
   * Logs node counts by type, activation functions, connection counts,
   * and whether recurrent/gated connections are present.
   *
   * Best-effort utility: swallows errors and logs partial data when inspection fails.
   * Never throws from this debug helper.
   *
   * @param network - The network (genome) to inspect.
   *                  Expected shape: `{ nodes: any[], connections: any[] }`
   *
   * @example
   * const { bestNetwork } = await EvolutionEngine.runMazeEvolution(options);
   * EvolutionEngine.printNetworkStructure(bestNetwork);
   * // Output:
   * // Network Structure:
   * // Nodes: 25
   * //   Input nodes: 5
   * //   Hidden nodes: 16
   * //   Output nodes: 4
   * // Activation functions: ['LOGISTIC', 'TANH', 'RELU']
   * // Connections: 120
   * // Has recurrent/gated connections: false
   */
  static printNetworkStructure(network: INetwork): void {
    printNetworkStructure(EvolutionEngine.#STATE, network);
  }
}
