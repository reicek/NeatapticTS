// Handles the main NEAT evolution loop for maze solving
// Exports: EvolutionEngine class with static methods

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
import {
  EVOLUTION_ENGINE_ACTION_DIMENSION,
  EVOLUTION_ENGINE_EMPTY_VECTOR,
  EVOLUTION_ENGINE_LOOP_CONSTANTS,
  EVOLUTION_ENGINE_PRETRAIN_CONSTANTS,
  EVOLUTION_ENGINE_WARM_START_CONSTANTS,
} from './evolutionEngine/evolutionEngine.constants';
import {
  applyEvolutionEngineRingState,
  configureEvolutionEngineToggles,
  getEvolutionEngineFacadeRuntimeState,
  getEvolutionEngineMaxLogitsRingCapacity,
  getEvolutionEngineSharedState,
} from './evolutionEngine/evolutionEngine.services';
import { resolveMazeEvolutionPhaseOutcome as resolveMazeEvolutionPhaseOutcomeImpl } from './evolutionEngine/curriculumPhase';
import { printNetworkStructure } from './evolutionEngine/networkInspection';
import {
  collectEvolutionEngineHiddenToOutputConnections,
  collectEvolutionEngineNodeIndicesByType,
} from './evolutionEngine/evolutionEngine.utils';
import type { INetwork } from './interfaces';
import type {
  EvolutionOptions,
  EvolutionLoopRuntimeContext,
  EvolutionLoopSupportContext,
  EvolutionLoopTelemetryContext,
  IRunMazeEvolutionOptions,
  MazeEvolutionCurriculumPhaseOutcome,
  MazeEvolutionRunResult,
  SpeciesHistoryHost,
} from './evolutionEngine/evolutionEngine.types';
import type Network from '../../src/architecture/network';

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
 * - `resolveMazeEvolutionPhaseOutcome(result, previousBest, minProgressToPass)`: Shared curriculum-phase interpretation helper
 * - `printNetworkStructure(network)`: Debug utility for network topology
 * - `setDeterministic(seed?)`: Enable deterministic mode
 * - `clearDeterministic()`: Disable deterministic mode
 *
 * Internal constants (private configuration values) are defined as static fields
 * and passed explicitly to module functions for transparent, testable orchestration.
 */
export class EvolutionEngine {
  /** Reusable empty vector constant to avoid ephemeral allocations from `|| []` fallbacks. */
  static #EMPTY_VEC: unknown[] = EVOLUTION_ENGINE_EMPTY_VECTOR;

  /** Number of action outputs (N,E,S,W) */
  static #ACTION_DIM = EVOLUTION_ENGINE_ACTION_DIMENSION;

  /**
   * Enable deterministic mode and optionally reseed the internal RNG via the shared state helpers.
   *
   * @param seed Optional numeric seed used to reseed the deterministic RNG sequence.
   */
  static setDeterministic(seed?: number): void {
    setDeterministicMode(getEvolutionEngineSharedState(), seed);
  }

  /**
   * Disable deterministic mode and return to non-deterministic random number generation.
   */
  static clearDeterministic(): void {
    clearDeterministicMode(getEvolutionEngineSharedState());
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
  static async runMazeEvolution(
    options: IRunMazeEvolutionOptions,
  ): Promise<MazeEvolutionRunResult> {
    const sharedEngineState = getEvolutionEngineSharedState();
    const facadeRuntimeState = getEvolutionEngineFacadeRuntimeState();

    // 1) Normalise and validate options (descriptive names, defaulting).
    const opts = normalizeRunOptions(
      options,
      (seed: number) => EvolutionEngine.setDeterministic(seed),
      (enabled: boolean) => {
        configureEvolutionEngineToggles(
          enabled,
          sharedEngineState.toggles.telemetryMinimal,
          sharedEngineState.toggles.disableBaldwinPhase,
        );
      },
      (enabled: boolean) => {
        configureEvolutionEngineToggles(
          sharedEngineState.toggles.reducedTelemetry,
          enabled,
          sharedEngineState.toggles.disableBaldwinPhase,
        );
      },
      (disabled: boolean) => {
        configureEvolutionEngineToggles(
          sharedEngineState.toggles.reducedTelemetry,
          sharedEngineState.toggles.telemetryMinimal,
          disabled,
        );
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
    } = prepareEnvironmentForRun(opts, sharedEngineState.scratch);

    // 3) Create and seed NEAT instance via a descriptive helper.
    const { neat, scratchPopClone, scratchSample } = createAndSeedNeat(
      opts,
      inputSize,
      outputSize,
      fitnessContext,
      sharedEngineState.scratch.populationCloneBuffer as Network[],
      sharedEngineState.scratch.samplePool,
    );
    if (!neat) {
      throw new Error('ASCII Maze failed to create a NEAT instance.');
    }
    sharedEngineState.scratch.populationCloneBuffer = scratchPopClone;
    sharedEngineState.scratch.samplePool = scratchSample;

    // 4) Ensure internal scratch/pooling capacity is sufficient for the configured population & network sizes.
    ensureScratchCapacity(sharedEngineState, {
      populationSize: opts.popSize,
      inputSize,
      outputSize,
    });

    // 5) Lamarckian warm-start (pretrain generation 0) when training cases exist.
    const lamarckianTrainingSet = buildLamarckianTrainingSet(
      sharedEngineState,
      {
        ...EVOLUTION_ENGINE_WARM_START_CONSTANTS,
        RNG_PARAMETERS: resolveRngParameters(),
      },
    );
    warmStartPopulationIfNeeded(
      neat,
      lamarckianTrainingSet,
      sharedEngineState,
      (neatInstance, trainingSet) => {
        pretrainPopulationWarmStart(
          neatInstance,
          trainingSet,
          EVOLUTION_ENGINE_PRETRAIN_CONSTANTS,
          sharedEngineState,
          (network) =>
            applyCompassWarmStart({ state: sharedEngineState, network }),
          (network) =>
            centerOutputBiases({ state: sharedEngineState, network }),
        );
      },
    );

    // 6) Prepare loop helpers and run the full evolution loop inside a private helper.
    const loopHelpers = prepareLoopHelpers(
      opts as unknown as EvolutionOptions,
      sharedEngineState.scratch,
    );

    // Lightweight profiling (opt-in): set env ASCII_MAZE_PROFILE=1 to enable
    const doProfile = !!(
      typeof process !== 'undefined' &&
      typeof process.env !== 'undefined' &&
      process.env.ASCII_MAZE_PROFILE === '1'
    );

    const runResult = await runEvolutionLoop(
      sharedEngineState,
      neat,
      opts as unknown as EvolutionOptions,
      lamarckianTrainingSet,
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      loopHelpers,
      doProfile,
      {
        scratchLogitsRing: sharedEngineState.scratch.logitsRing,
        logitsRingCapMax: getEvolutionEngineMaxLogitsRingCapacity(),
        actionDim: EvolutionEngine.#ACTION_DIM,
        scratchLogitsShared: sharedEngineState.scratch.sharedLogits,
        scratchLogitsSharedW: sharedEngineState.scratch.sharedLogitsWriteIndex,
      } satisfies EvolutionLoopRuntimeContext,
      {
        logitsRingCap: facadeRuntimeState.logitsRingCap,
        logitsRingShared: facadeRuntimeState.logitsRingShared,
        scratchLogitsRingW: facadeRuntimeState.scratchLogitsRingW,
      },
      {
        telemetryMinimal: sharedEngineState.toggles.telemetryMinimal,
        saturationPruneThreshold:
          EVOLUTION_ENGINE_LOOP_CONSTANTS.SATURATION_PRUNE_THRESHOLD,
        recentWindow: EVOLUTION_ENGINE_LOOP_CONSTANTS.RECENT_WINDOW,
        reducedTelemetry: sharedEngineState.toggles.reducedTelemetry,
      } satisfies EvolutionLoopTelemetryContext,
      {
        emptyVec: EvolutionEngine.#EMPTY_VEC as unknown as Network[],
        scratchNodeIdx: sharedEngineState.scratch.nodeIndexBuffer,
        scratchSnapshotObj: sharedEngineState.scratch.snapshotReusableObject,
        scratchSnapshotTop: sharedEngineState.scratch.snapshotTopEntries,
        speciesHistoryRef:
          ((EvolutionEngine as unknown as SpeciesHistoryHost)
            ._speciesHistory as unknown as number[]) ??
          (EvolutionEngine.#EMPTY_VEC as unknown as number[]),
        loopHelpers: {
          getNodeIndicesByType: (nodes, type) => {
            return collectEvolutionEngineNodeIndicesByType(
              sharedEngineState,
              nodes,
              type,
            );
          },
          collectHiddenToOutputConns: (hiddenNode, nodes, outputCount) => {
            return collectEvolutionEngineHiddenToOutputConnections(
              sharedEngineState,
              hiddenNode,
              nodes,
              outputCount,
            );
          },
        },
      } satisfies EvolutionLoopSupportContext,
      {
        ...EVOLUTION_ENGINE_LOOP_CONSTANTS,
        TELEMETRY_MINIMAL: sharedEngineState.toggles.telemetryMinimal,
        REDUCED_TELEMETRY: sharedEngineState.toggles.reducedTelemetry,
        DISABLE_BALDWIN: sharedEngineState.toggles.disableBaldwinPhase,
      },
    );

    // Update ring state from loop result
    applyEvolutionEngineRingState(runResult.updatedRingState);

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
        sharedEngineState,
        loopHelpers.safeWrite,
        completedGenerations,
        totalEvolveMs,
        totalLamarckMs,
        totalSimMs,
        isProfilingDetailsEnabled,
        getProfilingAccumulators,
      );
    }

    // Final return: best network, its simulation result, the NEAT instance, exit reason, and seeding profile id
    return {
      bestNetwork,
      bestResult,
      neat,
      exitReason: bestResult?.exitReason ?? 'incomplete',
      architectureProfileId: opts.architectureProfileId,
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
   *                  Expected shape: `{ nodes: Array<unknown>, connections: Array<unknown> }`
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
    printNetworkStructure(getEvolutionEngineSharedState(), network);
  }
}

/**
 * Stable curriculum-phase compatibility surface exposed from the engine facade.
 *
 * @remarks
 * The implementation lives in `evolutionEngine/curriculumPhase.ts`, but
 * callers that already import from `./evolutionEngine` should keep using this
 * façade export so the dedicated engine folder retains ownership without
 * forcing import churn across browser-entry, tests, or downstream examples.
 *
 * @param evolutionResult - Stable engine result returned by `runMazeEvolution()`.
 * @param previousBestNetwork - Previously carried curriculum winner, if one exists.
 * @param minProgressToPass - Progress threshold required before the curriculum advances.
 * @returns Shared curriculum outcome describing solve status and next carry-over winner.
 *
 * @example
 * ```ts
 * const phaseOutcome = resolveMazeEvolutionPhaseOutcome(result, previousBest, 95);
 * if (phaseOutcome.solved) {
 *   previousBest = phaseOutcome.nextBestNetwork;
 * }
 * ```
 */
export const resolveMazeEvolutionPhaseOutcome = (
  evolutionResult: MazeEvolutionRunResult,
  previousBestNetwork: INetwork | undefined,
  minProgressToPass: number,
): MazeEvolutionCurriculumPhaseOutcome => {
  return resolveMazeEvolutionPhaseOutcomeImpl(
    evolutionResult,
    previousBestNetwork,
    minProgressToPass,
  );
};
