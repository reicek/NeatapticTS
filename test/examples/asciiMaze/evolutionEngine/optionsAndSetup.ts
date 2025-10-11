/**
 * Options and Setup Module
 *
 * Purpose:
 * -------
 * Provides utilities for normalizing run options, preparing maze environment,
 * and orchestrating NEAT driver creation with population seeding.
 *
 * This module encapsulates:
 *  - Run options validation and normalization with sensible defaults
 *  - Environment preparation (maze encoding, distance maps, I/O sizing)
 *  - NEAT driver creation and population seeding orchestration
 *
 * ES2023 Policy:
 * -------------
 * - Uses nullish coalescing `??` for default values (never `||`)
 * - Descriptive variable names (no short identifiers)
 * - Optional chaining `?.` for safe property access
 * - Spread operator for object composition
 *
 * @module evolutionEngine/optionsAndSetup
 */

import { Network } from '../../../../src/neataptic';
import { MazeUtils } from '../mazeUtils';
import { FitnessEvaluator } from '../fitness';
import {
  IFitnessEvaluationContext,
  IRunMazeEvolutionOptions,
} from '../interfaces';
import { createNeat, seedInitialPopulation } from './neatConfiguration';

/**
 * Normalize and validate run options with sensible defaults.
 *
 * This function accepts the user-provided run options and returns a normalized
 * options object with all defaults applied and configuration groups extracted.
 *
 * Configuration Philosophy:
 *  - Prefer explicit defaults over implicit framework defaults
 *  - Use nullish coalescing `??` for clarity (avoid falsy semantics)
 *  - Extract nested configuration groups for easier parameter passing
 *  - Apply side effects cautiously (determinism mode, telemetry flags)
 *
 * Default Values:
 *  - Population size: 500
 *  - Max stagnant generations: 500
 *  - Min progress to pass: 95%
 *  - Max generations: Infinity
 *  - Lamarckian iterations: 10
 *  - Plateau generations: 40
 *  - Plateau improvement threshold: 1e-6
 *  - Simplify duration: 30 seconds
 *  - Simplify prune fraction: 0.05 (5%)
 *  - Simplify strategy: 'weakWeight'
 *  - Persist every: 25 generations
 *  - Persist directory: './ascii_maze_snapshots'
 *  - Persist top K: 3
 *  - Dynamic population enabled: true
 *  - Dynamic population expand interval: 25
 *  - Dynamic population expand factor: 0.15
 *  - Dynamic population plateau slack: 0.6
 *  - Memory compaction interval: 50
 *
 * Side Effects:
 *  - May call setDeterministic() when deterministic mode or randomSeed is provided
 *  - Sets global telemetry flags (REDUCED_TELEMETRY, TELEMETRY_MINIMAL, DISABLE_BALDWIN)
 *
 * Parameters:
 * @param options - Raw run options from the user
 * @param setDeterministic - Callback to set deterministic mode (seed: number) => void
 * @param setReducedTelemetry - Callback to set reduced telemetry flag (enabled: boolean) => void
 * @param setMinimalTelemetry - Callback to set minimal telemetry flag (enabled: boolean) => void
 * @param setDisableBaldwin - Callback to disable Baldwinian refinement (disabled: boolean) => void
 *
 * @returns Normalized options object with all defaults applied and configuration groups extracted
 *
 * @example
 * // Normalize with custom population size and deterministic mode
 * const opts = normalizeRunOptions(
 *   { evolutionAlgorithmConfig: { popSize: 200, deterministic: true } },
 *   (seed) => EvolutionEngine.setDeterministic(seed),
 *   (enabled) => EvolutionEngine.setReducedTelemetry(enabled),
 *   (enabled) => EvolutionEngine.setMinimalTelemetry(enabled),
 *   (disabled) => EvolutionEngine.setDisableBaldwin(disabled)
 * );
 */
export const normalizeRunOptions = (
  options: IRunMazeEvolutionOptions,
  setDeterministic: (seed: number) => void,
  setReducedTelemetry: (enabled: boolean) => void,
  setMinimalTelemetry: (enabled: boolean) => void,
  setDisableBaldwin: (disabled: boolean) => void
  // Type assertion: Return type contains mixed configuration from user options
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): any => {
  // Step 1: normalise option groups using nullish coalescing for robustness.
  const mazeConfig = options?.mazeConfig;
  const agentSimConfig = options?.agentSimConfig ?? {};
  const evolutionAlgorithmConfig = options?.evolutionAlgorithmConfig ?? {};
  const reportingConfig = options?.reportingConfig ?? {};

  // Step 2: pull algorithm-level settings with clear defaults and descriptive locals.
  const {
    allowRecurrent = true,
    popSize = 500,
    maxStagnantGenerations = 500,
    minProgressToPass = 95,
    maxGenerations = Infinity,
    randomSeed,
    initialPopulation,
    initialBestNetwork,
    lamarckianIterations = 10,
    lamarckianSampleSize,
    plateauGenerations = 40,
    plateauImprovementThreshold = 1e-6,
    simplifyDuration = 30,
    simplifyPruneFraction = 0.05,
    simplifyStrategy = 'weakWeight',
    persistEvery = 25,
    persistDir = './ascii_maze_snapshots',
    persistTopK = 3,
    dynamicPopEnabled = true,
    dynamicPopMax: dynamicPopulationMaxCfg,
    dynamicPopExpandInterval = 25,
    dynamicPopExpandFactor = 0.15,
    dynamicPopPlateauSlack = 0.6,
    stopOnlyOnSolve = false,
    autoPauseOnSolve = true,
    deterministic = false,
    memoryCompactionInterval = 50,
    telemetryReduceStats = false,
    telemetryMinimal = false,
    disableBaldwinianRefinement = false,
  } = evolutionAlgorithmConfig;

  // Step 3: determinism and engine global flags (best-effort side-effects).
  if (deterministic || typeof randomSeed === 'number') {
    setDeterministic(typeof randomSeed === 'number' ? randomSeed : 0x12345678);
  }
  setReducedTelemetry(!!telemetryReduceStats);
  setMinimalTelemetry(!!telemetryMinimal);
  setDisableBaldwin(!!disableBaldwinianRefinement);

  // Step 4: derive conservative dynamic-population max when not explicitly provided.
  const dynamicPopMax =
    typeof dynamicPopulationMaxCfg === 'number'
      ? dynamicPopulationMaxCfg
      : Math.max(popSize, 120);

  // Step 5: compose the final normalised options object (shape expected by callers).
  return {
    mazeConfig,
    agentSimConfig,
    evolutionAlgorithmConfig,
    reportingConfig,
    fitnessEvaluator: options?.fitnessEvaluator,
    popSize,
    allowRecurrent,
    maxStagnantGenerations,
    minProgressToPass,
    maxGenerations,
    randomSeed,
    initialPopulation,
    initialBestNetwork,
    lamarckianIterations,
    lamarckianSampleSize,
    plateauGenerations,
    plateauImprovementThreshold,
    simplifyDuration,
    simplifyPruneFraction,
    simplifyStrategy,
    persistEvery,
    persistDir,
    persistTopK,
    dynamicPopEnabled,
    dynamicPopMax,
    dynamicPopExpandInterval,
    dynamicPopExpandFactor,
    dynamicPopPlateauSlack,
    stopOnlyOnSolve,
    autoPauseOnSolve,
    deterministic,
    memoryCompactionInterval,
    telemetryReduceStats,
    telemetryMinimal,
    disableBaldwinianRefinement,
    neatOptions: {
      popSize,
      allowRecurrent,
      adaptiveMutation: { enabled: true, strategy: 'twoTier' },
      multiObjective: {
        enabled: true,
        complexityMetric: 'nodes',
        autoEntropy: true,
      },
      telemetry: {
        enabled: true,
        performance: true,
        complexity: true,
        hypervolume: true,
      },
      lineageTracking: true,
      novelty: { enabled: true, blendFactor: 0.15 },
      targetSpecies: 10,
      adaptiveTargetSpecies: {
        enabled: true,
        entropyRange: [0.3, 0.8],
        speciesRange: [6, 14],
        smooth: 0.5,
      },
    },
    maze: mazeConfig?.maze,
    // Type assertion: Mixed configuration object with diverse property types
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any;
};

/**
 * Prepare maze encoding, start/exit positions, distance map, and fitness context for the run.
 *
 * This function accepts normalized options and returns a bundle of precomputed
 * artifacts consumed by the evolution loop: encoded maze, start/exit coords,
 * distance map, fixed I/O sizes, and a compact fitness context.
 *
 * Design Rationale:
 *  - Precompute expensive artifacts once (encoding, distance map) for reuse
 *  - Use descriptive, allocation-light locals for clarity
 *  - Best-effort scratch pool warm-up to reduce first-use allocation spikes
 *  - All pool allocation failures are swallowed (non-fatal optimization)
 *
 * Fixed I/O Sizes:
 *  - Input size: 6 [compassScalar, openN, openE, openS, openW, progressDelta]
 *  - Output size: 4 [moveN, moveE, moveS, moveW]
 *
 * Steps (high-level):
 *  1) Resolve maze source from `opts` (top-level `maze` or nested `mazeConfig.maze`)
 *  2) Encode the maze into a simulator-friendly representation
 *  3) Locate start ('S') and exit ('E') coordinates
 *  4) Build a distance map from the exit to speed simulations
 *  5) Assemble fixed I/O sizes and a `fitnessContext` object and return everything
 *
 * Parameters:
 * @param opts - Normalized run options (produced by `normalizeRunOptions`)
 * @param scratchBundle - Engine scratch state for optional pool warm-up
 *
 * @returns Object containing:
 *  - encodedMaze: Simulator-friendly maze representation
 *  - startPosition: {x, y} coordinates of start ('S')
 *  - exitPosition: {x, y} coordinates of exit ('E')
 *  - distanceMap: Precomputed distance from each cell to exit
 *  - inputSize: Fixed network input count (6)
 *  - outputSize: Fixed network output count (4)
 *  - fitnessContext: Compact bundle for fitness evaluator
 *
 * @example
 * // Prepare environment for a maze evolution run
 * const env = prepareEnvironmentForRun(normalizedOpts, engineState.scratch);
 * const neat = createAndSeedNeat(normalizedOpts, env.inputSize, env.outputSize, env.fitnessContext);
 */
export const prepareEnvironmentForRun = (
  // Type assertion: Accepts normalized options with mixed types from normalizeRunOptions
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  opts: any,
  // Type assertion: Scratch bundle contains dynamic pooled structures
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  scratchBundle: any
  // Type assertion: Returns diverse environment properties for evolution setup
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): any => {
  // Step 1: Resolve the maze input in a null-safe way.
  const mazeSource = opts?.maze ?? opts?.mazeConfig?.maze;

  // Step 2: Encode the maze (delegated to MazeUtils). Keep the variable name explicit.
  const encodedMaze = MazeUtils.encodeMaze(mazeSource);

  // Step 3: Locate start and exit coordinates used by simulators.
  const startPosition = MazeUtils.findPosition(mazeSource, 'S');
  const exitPosition = MazeUtils.findPosition(mazeSource, 'E');

  // Step 4: Build or obtain a distance-map from the exit to speed up agent simulation.
  const distanceMap = MazeUtils.buildDistanceMap(encodedMaze, exitPosition);

  // Step 5: Fixed network I/O sizes for this domain - named constants for readability.
  const inputSize = 6; // [compassScalar, openN, openE, openS, openW, progressDelta]
  const outputSize = 4; // [moveN, moveE, moveS, moveW]

  // Assemble the compact fitness evaluation context consumed by the evaluator.
  const fitnessContext: IFitnessEvaluationContext = {
    encodedMaze,
    startPosition,
    exitPosition,
    agentSimConfig: opts?.agentSimConfig ?? {},
    distanceMap,
  };

  // Best-effort: warm a couple of small engine-level pools to reduce first-use allocations.
  try {
    if (!Array.isArray(scratchBundle.samplePool)) {
      scratchBundle.samplePool = new Array(32);
    }
    if (!(scratchBundle.profilingScratch instanceof Float64Array)) {
      scratchBundle.profilingScratch = new Float64Array(4);
    }
  } catch {
    // Swallow pool allocation errors - they are non-fatal and only an optimisation.
  }

  return {
    encodedMaze,
    startPosition,
    exitPosition,
    distanceMap,
    inputSize,
    outputSize,
    fitnessContext,
  };
};

/**
 * Create and seed a NEAT driver with normalized configuration and optional initial population.
 *
 * This function orchestrates the complete NEAT setup workflow:
 *  1) Build a fitness callback bound to the fitness context
 *  2) Instantiate the NEAT driver with normalized options
 *  3) Seed the driver's population from optional initial networks
 *  4) Warm up pooled scratch buffers to reduce first-use allocation spikes
 *
 * Design Rationale:
 *  - Single orchestration point for NEAT creation + seeding
 *  - Delegates heavy lifting to createNeat and seedInitialPopulation
 *  - Best-effort buffer warm-up (failures swallowed)
 *  - Returns updated scratch buffers for caller to persist
 *
 * Buffer Management:
 *  - Accepts and returns scratchPopClone buffer (grown if needed)
 *  - Accepts and returns scratchSample buffer (grown if needed)
 *  - Caller should persist returned buffers for reuse across runs
 *
 * Parameters:
 * @param opts - Normalized run options (produced by normalizeRunOptions)
 * @param inputSize - Network input count
 * @param outputSize - Network output count
 * @param fitnessContext - Compact fitness evaluation context
 * @param scratchPopClone - Pooled clone buffer (will be grown if needed)
 * @param scratchSample - Pooled sample buffer (will be grown if needed)
 *
 * @returns Object containing:
 *  - neat: Configured and seeded NEAT driver instance
 *  - scratchPopClone: Updated clone buffer (may be new array if grown)
 *  - scratchSample: Updated sample buffer (may be new array if grown)
 *
 * @example
 * // Create and seed NEAT with optional initial population
 * const { neat, scratchPopClone, scratchSample } = createAndSeedNeat(
 *   normalizedOpts,
 *   6,
 *   4,
 *   fitnessContext,
 *   scratchPopCloneBuffer,
 *   scratchSampleBuffer
 * );
 */
export const createAndSeedNeat = (
  // Type assertion: Accepts normalized options with mixed types from normalizeRunOptions
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  opts: any,
  inputSize: number,
  outputSize: number,
  fitnessContext: IFitnessEvaluationContext,
  // Type assertion: Pooled clone buffer for network population
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  scratchPopClone: any[],
  // Type assertion: Pooled sample buffer for selection operations
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  scratchSample: any[]
  // Type assertion: Returns NEAT driver and updated scratch buffers
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): any => {
  try {
    // Step 1: Build a descriptive, bound fitness callback.
    const fitnessCallback = (network: Network) =>
      (opts.fitnessEvaluator ?? FitnessEvaluator.defaultFitnessEvaluator)(
        network,
        fitnessContext
      );

    // Step 2: Instantiate the NEAT driver with the derived options.
    const neatDriver = createNeat(
      inputSize,
      outputSize,
      fitnessCallback,
      opts.neatOptions
    );

    // Step 3: Seed the newly created driver using provided initial population / best network.
    scratchPopClone = seedInitialPopulation(
      neatDriver,
      opts.initialPopulation ?? undefined,
      opts.initialBestNetwork ?? undefined,
      Number.isFinite(opts.popSize) ? Math.max(0, Math.floor(opts.popSize)) : 0,
      scratchPopClone
    );

    // Step 4 (optional best-effort): warm-up a pooled sample buffer to avoid
    // first-use reallocation spikes. This is intentionally non-essential and
    // wrapped in try/catch so failures don't affect the driver.
    try {
      // If the engine exposes a pooled sample array, ensure it has at least
      // a small capacity matching the configured population to reduce immediate growth.
      const targetPopulation = Number.isFinite(opts.popSize)
        ? Math.max(0, Math.floor(opts.popSize))
        : 0;
      if (targetPopulation > 0) {
        const pooledSample = scratchSample;
        if (
          !Array.isArray(pooledSample) ||
          pooledSample.length < targetPopulation
        ) {
          scratchSample = new Array(targetPopulation);
        }
      }
    } catch {
      // Best-effort: swallow sample buffer warm-up errors.
    }

    return { neat: neatDriver, scratchPopClone, scratchSample };
  } catch {
    // Top-level safety net: return null driver on catastrophic failure.
    // Caller should check for null and handle gracefully.
    return { neat: null, scratchPopClone, scratchSample };
  }
};
