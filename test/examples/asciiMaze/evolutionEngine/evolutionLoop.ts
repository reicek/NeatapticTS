/**
 * Evolution Loop Module
 *
 * Purpose:
 * -------
 * Provides utilities for running the main NEAT evolution loop, including
 * generation orchestration, cancellation checking, and loop helper preparation.
 *
 * This module encapsulates:
 *  - Cancellation detection (AbortSignal and legacy cancellation API)
 *  - Loop helper preparation (frame flushing, persistence, logging)
 *  - Generation execution and orchestration
 *  - Stop condition evaluation
 *
 * ES2023 Policy:
 * -------------
 * - Uses nullish coalescing `??` and optional chaining `?.`
 * - Descriptive variable names (no short identifiers)
 * - Async/await for generation loops
 * - Best-effort error handling (swallow non-fatal errors)
 *
 * @module evolutionEngine/evolutionLoop
 */

/** Extended Window interface for maze-specific global state */
interface MazeWindow extends Window {
  asciiMazePaused?: boolean;
}

/** Mutable result object with exitReason field */
interface MutableMazeResult extends IMazeRunResult {
  exitReason?: string;
}

/** Generation outcome with profiling timings */
interface GenerationOutcome {
  fittest: NetworkInstance;
  tEvolve?: number;
  tLamarck?: number;
}

/** Simulation result with profiling and ring state */
interface SimulationOutcome {
  generationResult: IMazeRunResult;
  simTime: number;
  updatedRingState: {
    logitsRingCap: number;
    logitsRingShared: boolean;
    scratchLogitsRingW: number;
  };
}

/** Evolution loop result */
interface EvolutionLoopResult {
  bestNetwork: NetworkInstance | null;
  bestResult: IMazeRunResult | undefined;
  neat: NeatInstance;
  completedGenerations: number;
  totalEvolveMs: number;
  totalLamarckMs: number;
  totalSimMs: number;
  updatedRingState: {
    logitsRingCap: number;
    logitsRingShared: boolean;
    scratchLogitsRingW: number;
  };
}

/** Network with dynamic runtime properties */
interface RuntimeNetworkInstance extends NetworkInstance {
  _lastStepOutputs?: Float32Array[];
  _saturationFraction?: number;
  _actionEntropy?: number;
}

/** Simulation result with step outputs */
interface SimResultWithOutputs extends IMazeRunResult {
  stepOutputs?: number[][];
}

import { MazeMovement } from '../mazeMovement';
import {
  makeFlushToFrame,
  initPersistence,
  makeSafeWriter,
} from './setupHelpers';
import { readHighResolutionTime } from './rngAndTiming';
import {
  ensureOutputIdentity,
  handleSpeciesHistory,
  maybeExpandPopulation,
  pruneSaturatedHiddenOutputs,
  antiCollapseRecovery,
  updatePlateauState,
  handleSimplifyState,
  compactPopulation,
  getSortedIndicesByScore,
} from './populationDynamics';
import {
  applyLamarckianTraining,
  adjustOutputBiasesAfterTraining,
} from './trainingWarmStart';
import { ensureLogitsRingCapacity, maybeShrinkScratch } from './scratchPools';
import { sampleSegmentIntoScratch } from './sampling';
import {
  logGenerationTelemetry,
  collectTelemetryTail,
} from './telemetryMetrics';
import {
  isProfilingDetailsEnabled,
  profilingStartTimestamp,
  accumulateProfilingDuration,
} from './rngAndTiming';
import type { EngineState } from './engineState';
import type {
  EvolutionOptions,
  IMazeRunResult,
  INetwork,
  NetworkInstance,
  NeatInstance,
  IDashboardManager,
  LoopHelpers,
  ScratchBundle,
  SnapshotEntry,
  ProfilingAccumulators,
  TrainingConstants,
  NetworkNode,
  NetworkConnection,
  SimulationResult,
} from '../interfaces';

/**
 * Inspect cooperative cancellation sources and annotate the provided result when cancelled.
 *
 * This function checks for cancellation from two sources in priority order:
 *  1) Legacy cancellation object with `isCancelled()` method
 *  2) Standard AbortSignal with `aborted` property
 *
 * Design Rationale:
 *  - Allocation-free (uses only short-lived local references)
 *  - Best-effort error handling (swallow exceptions to avoid disrupting loop)
 *  - Sets `exitReason` on result object for caller inspection
 *  - Safe to call on hot paths (no scratch buffers or pooling needed)
 *
 * Cancellation Priority:
 *  1. Check `options.cancellation.isCancelled()` (legacy API)
 *  2. Check `options.signal.aborted` (standard AbortSignal)
 *  3. Return undefined if no cancellation detected
 *
 * Parameters:
 * @param options - Optional run configuration which may contain `cancellation` and/or `signal`
 * @param bestResult - Optional mutable result object that will be annotated with `exitReason`
 *
 * @returns A reason string ('cancelled' | 'aborted') when cancellation is detected, otherwise `undefined`
 *
 * @example
 * // Check cancellation before starting next generation
 * const cancelReason = checkCancellation(opts, runResult);
 * if (cancelReason) return cancelReason;
 *
 * @example
 * // Check cancellation without result annotation
 * if (checkCancellation(opts)) {
 *   console.log('User requested cancellation');
 *   break;
 * }
 */
export const checkCancellation = (
  options: EvolutionOptions,
  bestResult?: IMazeRunResult,
): string | undefined => {
  try {
    // Step 1: Check legacy cancellation object first (if present).
    const legacyCancellation = options?.cancellation;
    if (
      legacyCancellation &&
      typeof legacyCancellation.isCancelled === 'function' &&
      legacyCancellation.isCancelled()
    ) {
      if (bestResult) bestResult.exitReason = 'cancelled';
      return 'cancelled';
    }

    // Step 2: Check standard AbortSignal (many hosts expose `options.signal`).
    const abortSignal = options?.signal;
    if (abortSignal?.aborted) {
      if (bestResult) bestResult.exitReason = 'aborted';
      return 'aborted';
    }
  } catch {
    // Best-effort: swallow any unexpected errors to avoid breaking the caller.
  }
  // No cancellation detected.
  return undefined;
};

/**
 * Build lightweight helpers used inside the evolution loop.
 *
 * This function assembles the helper utilities needed by the main evolution loop:
 *  - Frame flushing for cooperative yielding
 *  - Persistence handles for snapshot saving (Node.js only)
 *  - Safe logging writer with fallback chain
 *  - Scratch buffer warm-up (best-effort)
 *
 * Design Rationale:
 *  - All initialization is best-effort (failures swallowed)
 *  - Warms up common scratch buffers to reduce first-use allocation spikes
 *  - Returns simple POJO with utilities (no class coupling)
 *  - Side effects isolated to scratch bundle parameter
 *
 * Scratch Buffer Warm-Up:
 *  - samplePool: Array for population sampling
 *  - profilingScratch: Float64Array(4) for timing accumulation
 *  - exps: Float64Array(64) for exponential computations
 *
 * Parameters:
 * @param opts - Normalized run options (contains persistDir and dashboardManager)
 * @param scratchBundle - Engine scratch state for optional buffer warm-up
 *
 * @returns Object containing:
 *  - flushToFrame: Async function for cooperative yielding
 *  - fs: Node.js fs module (null in browsers)
 *  - path: Node.js path module (null in browsers)
 *  - safeWrite: Resilient logging function with fallback chain
 *
 * @example
 * // Prepare loop helpers with scratch buffer warm-up
 * const { flushToFrame, fs, path, safeWrite } = prepareLoopHelpers(opts, engineState.scratch);
 * safeWrite('Starting evolution...\n');
 * await flushToFrame(); // Yield to host
 */
export const prepareLoopHelpers = (
  opts: EvolutionOptions,
  scratchBundle: ScratchBundle,
): LoopHelpers => {
  // Step 1: Create the lightweight host-yield helper first.
  const flushToFrame = makeFlushToFrame();

  // Step 2: Initialise persistence handles (best-effort; may be null in browsers).
  const { fs, path } = initPersistence(opts?.persistDir);

  // Step 3: Build a resilient writer that falls back to console when necessary.
  const safeWrite = makeSafeWriter(opts?.reportingConfig?.dashboardManager);

  // Step 4 (non-critical): Warm-up common pooled scratch buffers to avoid
  // surprising allocations during the first few generations. All failures are
  // swallowed so the engine remains robust.
  try {
    if (!Array.isArray(scratchBundle.samplePool)) {
      scratchBundle.samplePool = [];
    }

    if (!(scratchBundle.profilingScratch instanceof Float64Array)) {
      scratchBundle.profilingScratch = new Float64Array(4);
    }

    if (!(scratchBundle.exps instanceof Float64Array)) {
      scratchBundle.exps = new Float64Array(64);
    }
  } catch {
    // Intentionally ignore pool initialisation failures; not essential.
  }

  // Return the same simple shape the rest of the engine expects.
  return { flushToFrame, fs, path, safeWrite };
};

/**
 * Inspect common termination conditions and perform minimal, best-effort side-effects.
 *
 * This function checks three canonical stop reasons in priority order:
 *  1) **Solved**: Best result achieves minimum progress threshold
 *  2) **Stagnation**: No improvement for maxStagnantGenerations
 *  3) **MaxGenerations**: Absolute generation cap reached
 *
 * Design Rationale:
 *  - Allocation-light (uses only local references)
 *  - Best-effort error handling (all side effects swallowed)
 *  - Safe to call frequently on hot paths
 *  - Updates dashboard and yields to host when stopping
 *  - Emits optional 'asciiMazeSolved' event on solve (browser only)
 *
 * Side Effects (Best-Effort):
 *  - Updates dashboard manager when stopping
 *  - Awaits flushToFrame to yield to host
 *  - Sets `window.asciiMazePaused` flag on solve (if autoPauseOnSolve)
 *  - Dispatches 'asciiMazeSolved' CustomEvent on solve
 *  - Annotates `bestResult.exitReason` with canonical reason string
 *
 * Parameters:
 * @param bestResult - Mutable run summary object (may be mutated with `exitReason`)
 * @param bestNetwork - Network object associated with the best result (for dashboard)
 * @param maze - Maze descriptor passed to dashboard updates/events
 * @param completedGenerations - Current generation index (integer)
 * @param neat - NEAT driver instance (passed to dashboard update)
 * @param dashboardManager - Optional manager exposing `update(maze, result, network, gen, neat)`
 * @param flushToFrame - Async function to yield to host renderer (e.g. requestAnimationFrame)
 * @param minProgressToPass - Numeric threshold to consider a run 'solved'
 * @param autoPauseOnSolve - When truthy set cooperative pause flag and emit event on solve
 * @param stopOnlyOnSolve - When true ignore stagnation/maxGenerations as stop reasons
 * @param stagnantGenerations - Current count of stagnant generations observed
 * @param maxStagnantGenerations - Max allowed stagnant generations before stopping
 * @param maxGenerations - Absolute generation cap after which the run stops
 *
 * @returns A canonical reason string ('solved'|'stagnation'|'maxGenerations') when stopping, otherwise `undefined`
 *
 * @example
 * // Check stop conditions after each generation
 * const reason = await checkStopConditions(
 *   bestResult, bestNet, maze, gen, neat, dashboard, flush,
 *   95, true, false, stagnant, 500, 10000
 * );
 * if (reason) {
 *   console.log('Stopping due to', reason);
 *   break;
 * }
 */
export const checkStopConditions = async (
  bestResult: IMazeRunResult | undefined,
  bestNetwork: NetworkInstance | null,
  maze: string[],
  completedGenerations: number,
  neat: NeatInstance,
  dashboardManager: IDashboardManager | undefined,
  flushToFrame: () => Promise<void>,
  minProgressToPass: number,
  autoPauseOnSolve: boolean,
  stopOnlyOnSolve: boolean,
  stagnantGenerations: number,
  maxStagnantGenerations: number,
  maxGenerations: number,
): Promise<string | undefined> => {
  // Local convenience aliases for small, hot checks.
  const hasBest = Boolean(bestResult);
  const shouldConsiderStops = !stopOnlyOnSolve;

  // --- 1) Solved check ---
  if (bestResult?.success && bestResult.progress >= (minProgressToPass ?? 0)) {
    // Attempt dashboard update and yield; swallow any errors.
    try {
      dashboardManager?.update?.(
        maze,
        bestResult,
        bestNetwork as INetwork | null,
        completedGenerations,
        neat,
      );
    } catch {
      // Swallow dashboard update errors
    }

    try {
      await flushToFrame?.();
    } catch {
      // Swallow frame flush errors
    }

    // Optionally set a cooperative pause and emit a small event for UIs to react to.
    if (autoPauseOnSolve) {
      try {
        if (typeof window !== 'undefined') {
          (window as MazeWindow).asciiMazePaused = true;
          try {
            window.dispatchEvent(
              new CustomEvent('asciiMazeSolved', {
                detail: {
                  maze,
                  generations: completedGenerations,
                  progress: bestResult?.progress,
                },
              }),
            );
          } catch {
            // Swallow event dispatch errors
          }
        }
      } catch {
        // Swallow window access errors
      }
    }

    if (hasBest) (bestResult as MutableMazeResult).exitReason = 'solved';
    return 'solved';
  }

  // --- 2) Stagnation check ---
  if (
    shouldConsiderStops &&
    isFinite(maxStagnantGenerations) &&
    stagnantGenerations >= maxStagnantGenerations
  ) {
    try {
      dashboardManager?.update?.(
        maze,
        bestResult,
        bestNetwork as INetwork | null,
        completedGenerations,
        neat,
      );
    } catch {
      // Swallow dashboard update errors
    }
    try {
      await flushToFrame?.();
    } catch {
      // Swallow frame flush errors
    }
    if (hasBest) (bestResult as MutableMazeResult).exitReason = 'stagnation';
    return 'stagnation';
  }

  // --- 3) Max generations check ---
  if (
    shouldConsiderStops &&
    isFinite(maxGenerations) &&
    completedGenerations >= maxGenerations
  ) {
    if (hasBest)
      (bestResult as MutableMazeResult).exitReason = 'maxGenerations';
    return 'maxGenerations';
  }

  // No stop condition matched.
  return undefined;
};

/**
 * Persist a population snapshot to disk at the configured interval.
 *
 * This function writes a JSON snapshot of the current generation state when the
 * generation cadence aligns with the configured persistence interval. It captures
 * top-K genomes, telemetry tail, and metadata for later analysis or resume.
 *
 * Design Rationale:
 *  - Best-effort semantics: swallows all errors to avoid disrupting evolution loop
 *  - Reuses pooled scratch objects (SCRATCH_SNAPSHOT_OBJ, SCRATCH_SNAPSHOT_TOP) to minimize allocations
 *  - Optional profiling when enabled (measures snapshot serialization time)
 *  - Defensive validation to ensure FS/path modules are available
 *
 * Scheduling Logic:
 *  - Only persists when `completedGenerations % persistEvery === 0`
 *  - Requires valid fs.writeFileSync and pathModule.join functions
 *  - Requires non-empty population
 *
 * Snapshot Structure:
 *  - generation: completed generation index
 *  - bestFitness: best fitness score this generation
 *  - simplifyMode: whether simplification was active
 *  - plateauCounter: current plateau detection counter
 *  - timestamp: Date.now() when snapshot was created
 *  - telemetryTail: last N telemetry entries
 *  - top: top-K genomes with minimal metadata (idx, score, nodes, connections, json)
 *
 * Parameters:
 * @param engineState - Shared engine state with scratch buffers and profiling
 * @param fs - Node.js fs module or compatible FS API
 * @param pathModule - Node.js path module or compatible path API
 * @param persistDir - Directory path where snapshots should be written
 * @param persistTopK - Number of top genomes to include in snapshot
 * @param completedGenerations - Current generation index
 * @param persistEvery - Generation interval for persistence (e.g., 25 = every 25th generation)
 * @param neat - NEAT instance with population
 * @param bestFitness - Best fitness score this generation
 * @param simplifyMode - Whether simplification mode is active
 * @param plateauCounter - Current plateau counter value
 * @param scratchSnapshotObj - Pooled snapshot object (reused across calls)
 * @param scratchSnapshotTop - Pooled top-K buffer (reused across calls)
 * @param collectTelemetryTailFn - Function to collect telemetry tail from state
 * @param getSortedIndicesByScoreFn - Function to get sorted genome indices
 * @param isProfilingDetailsEnabledFn - Function to check profiling state
 * @param profilingStartTimestampFn - Function to get profiling start time
 * @param accumulateProfilingDurationFn - Function to accumulate profiling duration
 *
 * @example
 * // Persist snapshot every 25 generations
 * persistSnapshotIfNeeded(
 *   state, fs, path, './snapshots', 10, 50, 25, neat, 0.95, false, 3,
 *   scratchObj, scratchTop, collectTail, getSorted, isProfilingEnabled, profileStart, profileAccum
 * );
 */
export const persistSnapshotIfNeeded = (
  engineState: EngineState,
  fs: { writeFileSync?: (path: string, data: string) => void } | null,
  pathModule: { join?: (...paths: string[]) => string } | null,
  persistDir: string | undefined,
  persistTopK: number,
  completedGenerations: number,
  persistEvery: number,
  neat: NeatInstance,
  bestFitness: number,
  simplifyMode: boolean,
  plateauCounter: number,
  scratchSnapshotObj: Record<string, unknown>,
  scratchSnapshotTop: SnapshotEntry[],
  collectTelemetryTailFn: (
    state: EngineState,
    neat: NeatInstance,
    count: number,
  ) => unknown,
  getSortedIndicesByScoreFn: (
    state: EngineState,
    population: NetworkInstance[],
  ) => number[],
  isProfilingDetailsEnabledFn: (state: EngineState) => boolean,
  profilingStartTimestampFn: (state: EngineState) => number,
  accumulateProfilingDurationFn: (
    state: EngineState,
    label: string,
    duration: number,
  ) => void,
) => {
  // Step 1: Defensive validation & scheduling cadence.
  if (
    !fs ||
    typeof fs.writeFileSync !== 'function' ||
    !pathModule ||
    typeof pathModule.join !== 'function' ||
    !persistDir ||
    !Number.isFinite(persistEvery) ||
    persistEvery <= 0
  )
    return;

  if (
    !Number.isFinite(completedGenerations) ||
    completedGenerations % persistEvery !== 0
  )
    return;

  if (!neat || !Array.isArray(neat.population) || neat.population.length === 0)
    return;

  try {
    // Step 2: Optional profiling start (high-precision timer when enabled).
    const snapshotProfilingEnabled = isProfilingDetailsEnabledFn(engineState);
    const profileStart = snapshotProfilingEnabled
      ? profilingStartTimestampFn(engineState)
      : 0;

    // Step 3: Populate pooled snapshot metadata (mutate shared scratch snapshot object).
    const snapshot = scratchSnapshotObj;
    snapshot.generation = completedGenerations;
    snapshot.bestFitness = bestFitness;
    snapshot.simplifyMode = Boolean(simplifyMode);
    snapshot.plateauCounter = Number.isFinite(plateauCounter)
      ? plateauCounter
      : 0;
    snapshot.timestamp = Date.now();
    snapshot.telemetryTail = collectTelemetryTailFn(engineState, neat, 5);

    // Step 4: Prepare the top-K minimal metadata list by reusing pooled buffer.
    const populationRef: NetworkInstance[] =
      (neat.population as NetworkInstance[]) ?? [];
    const sortedIndices =
      getSortedIndicesByScoreFn(engineState, populationRef) ?? [];
    const normalizedTopK = Math.max(
      0,
      Math.floor(Number.isFinite(persistTopK) ? persistTopK : 0),
    );
    const topLimit = Math.min(normalizedTopK, sortedIndices.length);

    const topBuffer = scratchSnapshotTop;
    if (topBuffer.length < topLimit) topBuffer.length = topLimit; // grow in place if needed

    for (let rank = 0; rank < topLimit; rank++) {
      // Reuse or create an entry object in the pooled topBuffer.
      const entry = topBuffer[rank] ?? (topBuffer[rank] = {});
      const genome = populationRef[sortedIndices[rank]];

      // Minimal invariant fields for later analysis/UI. Use optional chaining and defaults.
      entry.idx = sortedIndices[rank];
      entry.score = genome?.score;
      entry.nodes = genome?.nodes?.length ?? 0;
      entry.connections = genome?.connections?.length ?? 0;
      entry.json =
        typeof genome?.toJSON === 'function'
          ? JSON.stringify(genome.toJSON())
          : undefined;
    }

    topBuffer.length = topLimit;
    snapshot.top = topBuffer;

    // Step 5: Serialize compact JSON and write to disk using provided path/FS.
    const snapshotFilePath = pathModule.join(
      persistDir,
      `snapshot_gen${completedGenerations}.json`,
    );
    fs.writeFileSync(snapshotFilePath, JSON.stringify(snapshot));

    // Step 6: Profile delta accumulation (best-effort).
    if (snapshotProfilingEnabled) {
      accumulateProfilingDurationFn(
        engineState,
        'snapshot',
        profilingStartTimestampFn(engineState) - profileStart || 0,
      );
    }
  } catch {
    // Best-effort: swallow persistence errors silently to avoid disrupting the evolution loop.
  }
};

/**
 * Safely update a UI dashboard with the latest run state and optionally yield to the
 * host/frame via an awaited flush function.
 *
 * Behaviour (best-effort):
 *  1) If `dashboardManager.update` exists and is callable, call it with the stable
 *     argument order (maze, result, network, completedGenerations, neat). Any exception
 *     raised by the dashboard is swallowed to avoid interrupting the evolution loop.
 *  2) If `flushToFrame` is supplied as an async function, await it to yield control to
 *     the event loop or renderer (for example `() => new Promise(r => requestAnimationFrame(r))`).
 *  3) The helper avoids heap allocations and relies on existing pooled scratch buffers in
 *     the engine for heavy telemetry elsewhere; this method intentionally performs only
 *     short-lived control flow and minimal work.
 *
 * Design Rationale:
 *  - Allocation-free (no ephemeral objects or arrays created)
 *  - Best-effort error handling (swallow all dashboard/flush errors)
 *  - Stable argument order for dashboard implementations
 *  - Async support for cooperative yielding to host scheduler
 *
 * Parameters:
 * @param maze - Maze instance or descriptor used by dashboard rendering.
 * @param result - Per-run result object (path, progress, telemetry, etc.).
 * @param network - Network or genome object that should be visualised.
 * @param completedGenerations - Integer index of the completed generation.
 * @param neat - NEAT manager instance (context passed to the dashboard update).
 * @param dashboardManager - Optional manager exposing `update(maze, result, network, gen, neat)`.
 * @param flushToFrame - Optional async function used to yield to the host/frame scheduler; may be omitted.
 *
 * @example
 * // Yield to the browser's next repaint after dashboard update:
 * await updateDashboardAndMaybeFlush(
 *   maze, genResult, fittestNetwork, gen, neatInstance, dashboard, () => new Promise(r => requestAnimationFrame(r))
 * );
 */
export const updateDashboardAndMaybeFlush = async (
  maze: string[],
  result: IMazeRunResult | undefined,
  network: NetworkInstance | null,
  completedGenerations: number,
  neat: NeatInstance,
  dashboardManager: IDashboardManager | undefined,
  flushToFrame?: () => Promise<void>,
) => {
  // Step 0: Defensive local aliases with descriptive names to improve readability in hot paths.
  const manager = dashboardManager;
  const yieldFrame = flushToFrame;

  // Step 1: Call dashboard update if provided. This is intentionally best-effort and must not
  // throw — any error from the dashboard implementation is swallowed to keep the engine stable.
  if (manager?.update && typeof manager.update === 'function') {
    try {
      // Use the stable argument order so dashboard implementations are consistent.
      manager.update(
        maze,
        result,
        network as INetwork | null,
        completedGenerations,
        neat,
      );
    } catch {
      // Swallow dashboard errors — telemetry/UI must not break evolution.
    }
  }

  // Step 2: Optionally yield to the host/frame scheduler. Guard typeof to avoid an accidental
  // Promise rejection when a caller passes undefined/non-function.
  if (typeof yieldFrame === 'function') {
    try {
      await yieldFrame();
    } catch {
      // Swallow flush errors; a failed frame yield is non-fatal for the evolution loop.
    }
  }
};

/**
 * Periodic dashboard update used when the engine wants to refresh a non-primary
 * dashboard view (for example background or periodic reporting). This helper is
 * intentionally small, allocation-light and best-effort: dashboard errors are
 * swallowed so the evolution loop cannot be interrupted by UI issues.
 *
 * Behavioural contract:
 *  1) If `dashboardManager.update` is present and callable the method invokes it with
 *     the stable argument order: (maze, bestResult, bestNetwork, completedGenerations, neat).
 *  2) If `flushToFrame` is supplied the helper awaits it after the update to yield to
 *     the host renderer (eg. requestAnimationFrame). Any exceptions raised by the
 *     flush are swallowed.
 *  3) The helper avoids creating ephemeral arrays/objects and therefore does not use
 *     typed-array scratch buffers here — there is no hot numerical work to pool. Other
 *     engine helpers already reuse class-level scratch buffers where appropriate.
 *
 * Design Rationale:
 *  - Fast-guard early when update cannot be performed
 *  - Best-effort error handling (swallow all exceptions)
 *  - Preserves dashboard `this` binding with `.call()`
 *  - Minimal allocations (no scratch buffers needed)
 *
 * Steps / inline intent:
 *  1. Fast-guard when an update cannot be performed (missing manager, update method,
 *     or missing content to visualise).
 *  2. Call the dashboard update in a try/catch to preserve best-effort semantics.
 *  3. Optionally await the provided `flushToFrame` function to yield to the host.
 *
 * Parameters:
 * @param maze - Maze descriptor passed to the dashboard renderer.
 * @param bestResult - Best-run result object used for display (may be falsy when not present).
 * @param bestNetwork - Network or genome object to visualise (may be falsy when not present).
 * @param completedGenerations - Completed generation index (number).
 * @param neat - NEAT manager instance (passed through to dashboard update).
 * @param dashboardManager - Optional manager exposing `update(maze, result, network, gen, neat)`.
 * @param flushToFrame - Optional async function used to yield to the host/frame scheduler
 *                      (for example: `() => new Promise(r => requestAnimationFrame(r))`).
 * @example
 * // Safe periodic update and yield to next frame
 * await updateDashboardPeriodic(
 *   maze, result, network, gen, neatInstance, dashboard, () => new Promise(r => requestAnimationFrame(r))
 * );
 */
export const updateDashboardPeriodic = async (
  maze: string[],
  bestResult: IMazeRunResult | undefined,
  bestNetwork: NetworkInstance | null,
  completedGenerations: number,
  neat: NeatInstance,
  dashboardManager: IDashboardManager | undefined,
  flushToFrame?: () => Promise<void>,
) => {
  // Step 0: create descriptive local aliases to clarify intent and keep hot-path refs short.
  const dashboard = dashboardManager;
  const updateFunction = dashboard?.update;
  const frameFlush = flushToFrame;

  // Step 1: Fast-guard — nothing to do when update isn't callable or we lack meaningful data.
  if (typeof updateFunction !== 'function' || !bestNetwork || !bestResult)
    return;

  // Step 2: Invoke dashboard update in a best-effort manner. Swallow any errors so the
  // evolution loop cannot be disrupted by UI failures.
  try {
    // Use `.call` to preserve potential dashboard `this` binding semantics.
    updateFunction.call(
      dashboard,
      maze,
      bestResult,
      bestNetwork as unknown as INetwork,
      completedGenerations,
      neat,
    );
  } catch {
    // Intentionally ignore update errors — dashboard should not crash the engine.
  }

  // Step 3: Optionally yield to the host renderer/scheduler.
  if (typeof frameFlush === 'function') {
    try {
      await frameFlush();
    } catch {
      // Ignore flush errors — non-critical for engine progress.
    }
  }
};

/**
 * Emit a formatted profiling summary showing average per-generation timings.
 *
 * This function prints a compact profiling summary with average millisecond timings
 * for the main evolution phases (evolve, Lamarckian training, simulation). If detailed
 * profiling is enabled, it also prints averages for telemetry, simplify, snapshot, and
 * prune operations.
 *
 * Design Rationale:
 *  - Allocation-free (reuses pooled Float64Array for intermediate calculations)
 *  - Best-effort error handling (swallow all exceptions)
 *  - Defensive numeric validation with divide-by-zero guards
 *  - Conditional detailed profiling output
 *
 * Calculation Steps:
 *  1. Validate and normalize generation count (guard divide-by-zero)
 *  2. Store totals in pooled scratch buffer (4-slot Float64Array)
 *  3. Compute per-generation averages by dividing totals by generation count
 *  4. Format numbers with 2 decimal places and print compact summary
 *  5. If detailed profiling enabled, print averaged detail line
 *
 * Parameters:
 * @param engineState - Shared engine state with scratch buffers and profiling accumulators
 * @param safeWrite - Safe logging function (best-effort, never throws)
 * @param completedGenerations - Number of completed generations (must be > 0)
 * @param totalEvolveMs - Total milliseconds spent in NEAT evolve() calls
 * @param totalLamarckMs - Total milliseconds spent in Lamarckian training
 * @param totalSimMs - Total milliseconds spent in simulation
 * @param isProfilingDetailsEnabledFn - Function to check if detailed profiling is enabled
 * @param getProfilingAccumulatorsFn - Function to get detailed profiling accumulators
 *
 * @example
 * // Print averages after a run that completed 100 generations
 * emitProfileSummary(
 *   state, console.log, 100, 12000, 3000, 4500,
 *   isProfilingDetailsEnabled, getProfilingAccumulators
 * );
 */
export const emitProfileSummary = (
  engineState: EngineState,
  safeWrite: (msg: string) => void,
  completedGenerations: number,
  totalEvolveMs: number,
  totalLamarckMs: number,
  totalSimMs: number,
  isProfilingDetailsEnabledFn: (state: EngineState) => boolean,
  getProfilingAccumulatorsFn: (state: EngineState) => ProfilingAccumulators,
) => {
  try {
    // Step 1: Normalise inputs and guard against divide-by-zero.
    const generations =
      Number.isFinite(completedGenerations) && completedGenerations > 0
        ? Math.floor(completedGenerations)
        : 0;
    if (generations === 0) return; // nothing to report

    // Step 2: Obtain a small pooled Float64Array (4 slots) to avoid per-call allocations.
    // Layout: [0]=totalEvolveMs, [1]=totalLamarckMs, [2]=totalSimMs, [3]=totalPerGen
    const scratchBundle = engineState.scratch;
    const profilingBuffer =
      scratchBundle.profilingScratch ??
      (scratchBundle.profilingScratch = new Float64Array(4));
    profilingBuffer[0] = Number.isFinite(totalEvolveMs) ? totalEvolveMs : 0;
    profilingBuffer[1] = Number.isFinite(totalLamarckMs) ? totalLamarckMs : 0;
    profilingBuffer[2] = Number.isFinite(totalSimMs) ? totalSimMs : 0;

    // Step 3: Compute per-generation averages using the pooled buffer.
    profilingBuffer[0] = profilingBuffer[0] / generations; // avg evolve
    profilingBuffer[1] = profilingBuffer[1] / generations; // avg lamarck
    profilingBuffer[2] = profilingBuffer[2] / generations; // avg sim
    profilingBuffer[3] =
      profilingBuffer[0] + profilingBuffer[1] + profilingBuffer[2]; // avg total per gen

    // Step 4: Format numbers with two decimals and print the compact summary.
    const avgEvolveStr = profilingBuffer[0].toFixed(2);
    const avgLamarckStr = profilingBuffer[1].toFixed(2);
    const avgSimStr = profilingBuffer[2].toFixed(2);
    const avgTotalPerGenStr = profilingBuffer[3].toFixed(2);

    safeWrite(
      `\n[PROFILE] Generations=${generations} avg(ms): evolve=${avgEvolveStr} lamarck=${avgLamarckStr} sim=${avgSimStr} totalPerGen=${avgTotalPerGenStr}\n`,
    );

    // Step 5: If the engine accumulates detailed profiling, print averaged detail line.
    if (isProfilingDetailsEnabledFn(engineState)) {
      const detailAccum = getProfilingAccumulatorsFn(engineState);
      const denom = generations || 1;
      // Defensive numeric extraction and formatting (do not allocate intermediate arrays)
      const avgTelemetry = Number.isFinite(detailAccum?.telemetry)
        ? ((detailAccum.telemetry ?? 0) / denom).toFixed(2)
        : '0.00';
      const avgSimplify = Number.isFinite(detailAccum?.simplify)
        ? ((detailAccum.simplify ?? 0) / denom).toFixed(2)
        : '0.00';
      const avgSnapshot = Number.isFinite(detailAccum?.snapshot)
        ? ((detailAccum.snapshot ?? 0) / denom).toFixed(2)
        : '0.00';
      const avgPrune = Number.isFinite(detailAccum?.prune)
        ? ((detailAccum.prune ?? 0) / denom).toFixed(2)
        : '0.00';
      safeWrite(
        `[PROFILE_DETAIL] avgTelemetry=${avgTelemetry} avgSimplify=${avgSimplify} avgSnapshot=${avgSnapshot} avgPrune=${avgPrune}\n`,
      );
    }
  } catch {
    // Best-effort: never let profiling output disrupt the run.
  }
};

/**
 * Run one generation: evolve, ensure output identity, update species history, maybe expand population,
 * and run Lamarckian training if configured.
 *
 * Behaviour & contract:
 *  - Performs a single NEAT generation step in a best-effort, non-throwing manner.
 *  - Measures profiling durations when `doProfile` is truthy. Profiling is optional and
 *    kept allocation-free (uses local numeric temporaries only).
 *  - Invokes the following steps in order (each step is wrapped in a try/catch so
 *    the evolution loop remains resilient to per-stage failures):
 *      1) `neat.evolve()` to produce the fittest network for this generation.
 *      2) `ensureOutputIdentity` to normalise output activations for consumers.
 *      3) `handleSpeciesHistory` to update species statistics and history.
 *      4) `maybeExpandPopulation` to grow the population when configured and warranted.
 *      5) Optional Lamarckian warm-start training via `applyLamarckianTraining`.
 *  - The method is allocation-light and reuses engine helpers / pooled buffers where
 *    appropriate. It never throws; internal errors are swallowed and optionally logged
 *    via the provided `safeWrite` function.
 *
 * @param engineState - Shared engine state with scratch buffers and RNG
 * @param neat - NEAT driver instance used for evolving the generation
 * @param doProfile - When truthy measure timing for the evolve step (ms) using engine clock
 * @param lamarckianIterations - Number of supervised training iterations to run per genome (0 to skip)
 * @param lamarckianTrainingSet - Array of supervised training cases used for warm-start (may be empty)
 * @param lamarckianSampleSize - Optional per-network sample size used by the warm-start routine
 * @param safeWrite - Safe logging function; used only for best-effort diagnostic messages
 * @param completedGenerations - Current generation index (used by expansion heuristics)
 * @param dynamicPopEnabled - Whether dynamic population expansion is enabled
 * @param dynamicPopMax - Upper bound on population size for expansion
 * @param plateauGenerations - Window size used by plateau detection
 * @param plateauCounter - Current plateau counter used by expansion heuristics
 * @param dynamicPopExpandInterval - Generation interval to attempt expansion
 * @param dynamicPopExpandFactor - Fractional growth factor used to compute new members
 * @param dynamicPopPlateauSlack - Minimum plateau ratio required to trigger expansion
 * @param speciesHistoryRef - Mutable array holding species history (maintained externally)
 * @param emptyVec - Empty array fallback to avoid ephemeral allocations
 * @param scratchNodeIdx - Pooled node index buffer (passed through to helpers)
 * @param getNodeIndicesByType - Helper function to collect node indices by type
 * @param constants - Object containing DEFAULT_TRAIN_ERROR, DEFAULT_TRAIN_RATE, DEFAULT_TRAIN_MOMENTUM, DEFAULT_TRAIN_BATCH_SMALL, DEFAULT_STD_SMALL, DEFAULT_STD_ADJUST_MULT
 *
 * @returns An object shaped { fittest, tEvolve, tLamarck } where:
 *  - `fittest` is the network returned by `neat.evolve()` (may be null on error),
 *  - `tEvolve` is the measured evolve duration in milliseconds when `doProfile` is true (0 otherwise),
 *  - `tLamarck` is the total time spent in Lamarckian training (0 when skipped)
 *
 * @example
 * // Run a single generation with profiling and optional Lamarckian warm-start
 * const { fittest, tEvolve, tLamarck } = await runGeneration(
 *   engineState,
 *   neatInstance,
 *   true,   // doProfile
 *   5,      // lamarckianIterations
 *   trainingSet,
 *   16,     // lamarckianSampleSize
 *   console.log,
 *   genIndex,
 *   true,
 *   500,
 *   10,
 *   plateauCounter,
 *   5,
 *   0.1,
 *   0.75,
 *   speciesHistory,
 *   [],
 *   nodeIndexBuffer,
 *   getNodeIndicesByTypeFn,
 *   { DEFAULT_TRAIN_ERROR: 0.01, ... }
 * );
 */
export const runGeneration = async (
  engineState: EngineState,
  neat: NeatInstance,
  doProfile: boolean,
  lamarckianIterations: number,
  lamarckianTrainingSet: { input: number[]; output: number[] }[],
  lamarckianSampleSize: number | undefined,
  safeWrite: (msg: string) => void,
  completedGenerations: number,
  dynamicPopEnabled: boolean,
  dynamicPopMax: number,
  plateauGenerations: number,
  plateauCounter: number,
  dynamicPopExpandInterval: number,
  dynamicPopExpandFactor: number,
  dynamicPopPlateauSlack: number,
  speciesHistoryRef: number[],
  emptyVec: NetworkInstance[],
  scratchNodeIdx: Int32Array,
  getNodeIndicesByType: (nodes: NetworkNode[], type: string) => number,
  constants: TrainingConstants,
) => {
  // Step 0: Local descriptive aliases and profiling setup.
  const profileEnabled = Boolean(doProfile);
  const clockNow = () => readHighResolutionTime();
  const startTime = profileEnabled ? clockNow() : 0;

  // Results we will populate. Keep names descriptive for readability in hot paths.
  let fittestNetwork: NetworkInstance | null = null;
  let evolveDuration = 0;
  let lamarckDuration = 0;

  // Step 1: Run the evolutionary step and measure time when profiling is enabled.
  try {
    // `neat` is expected to provide an async `evolve()` method that returns the fittest genome.
    fittestNetwork = await neat?.evolve();
    if (profileEnabled) evolveDuration = clockNow() - startTime;
  } catch {
    // Best-effort: log a short diagnostic and continue. Do not rethrow.
    try {
      safeWrite?.(`runGeneration: evolve() threw error`);
    } catch {
      // Swallow write errors
    }
    // leave fittestNetwork null and continue with remaining housekeeping.
  }

  // Step 2: Ensure outputs are using identity activation where required (non-throwing).
  try {
    ensureOutputIdentity(neat);
  } catch {
    try {
      safeWrite?.(`runGeneration: ensureOutputIdentity failed`);
    } catch {
      // Swallow write errors
    }
  }

  // Step 3: Update species history (best-effort; internal errors are swallowed).
  try {
    const effectiveHistoryRef = speciesHistoryRef ?? emptyVec;
    handleSpeciesHistory(engineState, neat, effectiveHistoryRef);
  } catch {
    try {
      safeWrite?.(`runGeneration: handleSpeciesHistory failed`);
    } catch {
      // Swallow write errors
    }
  }

  // Step 4: Possibly expand the population when configured and plateau conditions are met.
  try {
    maybeExpandPopulation(
      engineState,
      neat,
      Boolean(dynamicPopEnabled),
      completedGenerations,
      dynamicPopMax,
      plateauGenerations,
      plateauCounter,
      dynamicPopExpandInterval,
      dynamicPopExpandFactor,
      dynamicPopPlateauSlack,
      safeWrite,
    );
  } catch {
    try {
      safeWrite?.(`runGeneration: maybeExpandPopulation failed`);
    } catch {
      // Swallow write errors
    }
  }

  // Step 5: Optional Lamarckian warm-start training. This step may be expensive;
  // we keep it synchronous as the called helper currently returns a numeric time.
  try {
    const shouldRunLamarckian =
      Number.isFinite(lamarckianIterations) &&
      lamarckianIterations > 0 &&
      Array.isArray(lamarckianTrainingSet) &&
      lamarckianTrainingSet.length > 0;

    if (shouldRunLamarckian) {
      // The helper returns the measured time (ms) spent in training when profiling is enabled.
      lamarckDuration = applyLamarckianTraining(
        neat,
        lamarckianTrainingSet,
        lamarckianIterations,
        lamarckianSampleSize,
        safeWrite,
        doProfile,
        completedGenerations,
        engineState,
        {
          DEFAULT_TRAIN_ERROR: constants.DEFAULT_TRAIN_ERROR,
          DEFAULT_TRAIN_RATE: constants.DEFAULT_TRAIN_RATE,
          DEFAULT_TRAIN_MOMENTUM: constants.DEFAULT_TRAIN_MOMENTUM,
          DEFAULT_TRAIN_BATCH_SMALL: constants.DEFAULT_TRAIN_BATCH_SMALL,
        },
        (network) => {
          adjustOutputBiasesAfterTraining(
            network,
            engineState,
            {
              DEFAULT_STD_SMALL: constants.DEFAULT_STD_SMALL,
              DEFAULT_STD_ADJUST_MULT: constants.DEFAULT_STD_ADJUST_MULT,
            },
            scratchNodeIdx,
            getNodeIndicesByType,
          );
        },
      );
    }
  } catch {
    try {
      safeWrite?.(`runGeneration: applyLamarckianTraining failed`);
    } catch {
      // Swallow write errors
    }
  }

  // Final: return the canonical result shape. Keep original property names for callers.
  return {
    fittest: fittestNetwork,
    tEvolve: evolveDuration,
    tLamarck: lamarckDuration,
  } as GenerationOutcome;
};

/**
 * Simulate the supplied `fittest` genome/network and perform allocation-light postprocessing.
 *
 * Behaviour & contract:
 *  - Runs the simulation via `MazeMovement.simulateAgent` and attaches compact telemetry
 *    (saturation fraction, action entropy) directly onto the `fittest` object (in-place).
 *  - When per-step logits are returned the helper attempts to copy them into the engine's pooled
 *    ring buffers to avoid per-run allocations. Two copy modes are supported:
 *      1) Shared SAB-backed flat Float32Array with an atomic Int32 write index (cross-worker safe).
 *      2) Local in-process per-row Float32Array ring (`scratchLogitsRing`).
 *  - Best-effort: all mutation and buffer-copy steps are guarded; failures are swallowed so the
 *    evolution loop is not interrupted. Use `safeWrite` for optional diagnostic messages.
 *
 * Steps (high level):
 *  1) Run the simulator and capture wall-time when `doProfile` is truthy.
 *  2) Attach compact telemetry fields to `fittest` and ensure legacy `_lastStepOutputs` exists.
 *  3) If per-step logits are available, ensure ring capacity and copy them into the selected ring.
 *  4) Optionally prune saturated hidden->output connections and emit telemetry via logGenerationTelemetry.
 *  5) Return the raw simulation result and elapsed simulation time (ms when profiling enabled).
 *
 * Notes on pooling / reentrancy:
 *  - The local ring is not re-entrant; callers must avoid concurrent writes.
 *  - When shared mode is true we prefer the SAB-backed path which uses Atomics and is safe
 *    for cross-thread producers.
 *
 * @param engineState - Shared engine state with scratch buffers and ring configuration
 * @param fittest - Genome/network considered the generation's best; may be mutated with metadata
 * @param encodedMaze - Maze descriptor used by the simulator
 * @param startPosition - Start co-ordinates passed as-is to the simulator
 * @param exitPosition - Exit co-ordinates passed as-is to the simulator
 * @param distanceMap - Optional precomputed distance map consumed by the simulator
 * @param maxSteps - Optional maximum simulation steps; may be undefined to allow default
 * @param doProfile - When truthy measure and return the simulation time in milliseconds
 * @param safeWrite - Optional logger used for non-fatal diagnostic messages
 * @param logEvery - Emit telemetry every `logEvery` generations (0 disables periodic telemetry)
 * @param completedGenerations - Current generation index used for conditional telemetry
 * @param neat - NEAT driver instance passed to telemetry hooks
 * @param scratchLogitsRing - Pooled logits ring buffer reference
 * @param logitsRingCap - Current ring capacity (power of two)
 * @param logitsRingCapMax - Maximum allowed ring capacity
 * @param actionDim - Number of action dimensions (typically 4 for NESW)
 * @param logitsRingShared - Whether shared SAB mode is enabled
 * @param scratchLogitsShared - Shared flat Float32Array (when shared mode enabled)
 * @param scratchLogitsSharedW - Shared atomic write index (when shared mode enabled)
 * @param scratchLogitsRingW - Local ring write cursor (when not shared)
 * @param telemetryMinimal - Whether minimal telemetry mode is active
 * @param saturationPruneThreshold - Threshold above which to prune saturated outputs
 * @param recentWindow - Size of telemetry tail window
 * @param reducedTelemetry - Whether reduced telemetry mode is active
 * @param getNodeIndicesByType - Helper to collect node indices by type
 * @param collectHiddenToOutputConns - Helper to collect hidden-to-output connections
 *
 * @returns An object { generationResult, simTime, updatedRingState } where simTime is ms when profiling is enabled
 *
 * @example
 * const { generationResult, simTime, updatedRingState } = simulateAndPostprocess(
 *   state, bestGenome, maze, start, exit, distMap, 1000, true, console.log, 10, genIdx, neat, ...
 * );
 */
export const simulateAndPostprocess = (
  engineState: EngineState,
  fittest: NetworkInstance,
  encodedMaze: number[][],
  startPosition: readonly [number, number],
  exitPosition: readonly [number, number],
  distanceMap: number[][],
  maxSteps: number | undefined,
  doProfile: boolean,
  safeWrite: (msg: string) => void,
  logEvery: number,
  completedGenerations: number,
  neat: NeatInstance,
  scratchLogitsRing: Float32Array[],
  logitsRingCap: number,
  logitsRingCapMax: number,
  actionDim: number,
  logitsRingShared: boolean,
  scratchLogitsShared: Float32Array | undefined,
  scratchLogitsSharedW: Int32Array | undefined,
  scratchLogitsRingW: number,
  telemetryMinimal: boolean,
  saturationPruneThreshold: number,
  recentWindow: number,
  reducedTelemetry: boolean,
  getNodeIndicesByType: (nodes: NetworkNode[], type: string) => number,
  collectHiddenToOutputConns: (
    hiddenNode: NetworkNode,
    nodesRef: NetworkNode[],
    outputCount: number,
  ) => NetworkConnection[],
): SimulationResult => {
  // Step 1: Run simulator and optionally capture elapsed time.
  const startTime = doProfile ? readHighResolutionTime() : 0;
  const simResult = MazeMovement.simulateAgent(
    fittest as unknown as INetwork,
    encodedMaze,
    startPosition,
    exitPosition,
    distanceMap,
    maxSteps,
  );

  // Best-effort: attach legacy buffer refs and compact telemetry onto the genome.
  const runtimeFittest = fittest as RuntimeNetworkInstance;
  try {
    if (!runtimeFittest._lastStepOutputs) {
      runtimeFittest._lastStepOutputs = scratchLogitsRing;
    }
  } catch {
    // Swallow legacy buffer attachment errors
  }

  try {
    runtimeFittest._saturationFraction = simResult?.saturationFraction ?? 0;
    runtimeFittest._actionEntropy = simResult?.actionEntropy ?? 0;
  } catch {
    // Swallow telemetry assignment errors
  }

  // Mutable ring state (will be updated and returned)
  let updatedLogitsRingCap = logitsRingCap;
  let updatedLogitsRingShared = logitsRingShared;
  let updatedScratchLogitsRingW = scratchLogitsRingW;

  // Step 3: If the simulator returned per-step logits, copy them into the pooled ring buffers.
  try {
    const simResultWithOutputs = simResult as SimResultWithOutputs;
    const perStepLogits = simResultWithOutputs?.stepOutputs;
    if (Array.isArray(perStepLogits) && perStepLogits.length > 0) {
      // Ensure the ring can hold the incoming sequence to avoid overflow resize churn.
      const logitsRingResult = ensureLogitsRingCapacity({
        state: engineState,
        desiredRecentSteps: perStepLogits.length,
        currentCapacity: updatedLogitsRingCap,
        minimumCapacity: 128,
        maximumCapacity: logitsRingCapMax,
        actionDimension: actionDim,
        sharedModeEnabled: updatedLogitsRingShared,
      });
      updatedLogitsRingCap = logitsRingResult.capacity;
      updatedLogitsRingShared = logitsRingResult.sharedModeEnabled;

      const useSharedSAB =
        updatedLogitsRingShared && scratchLogitsShared && scratchLogitsSharedW;

      if (useSharedSAB) {
        // Shared flat Float32Array layout: [ idx(Int32), floats... ] with atomic index at view[0].
        const sharedBuffer = scratchLogitsShared as Float32Array;
        const atomicIndexView = scratchLogitsSharedW as Int32Array;
        const capacityMask = updatedLogitsRingCap - 1;

        for (let stepIndex = 0; stepIndex < perStepLogits.length; stepIndex++) {
          const logitsVector = perStepLogits[stepIndex];
          if (!Array.isArray(logitsVector)) continue;

          // Reserve a slot atomically and compute its base offset in the flat buffer.
          const currentWriteIndex =
            Atomics.load(atomicIndexView, 0) & capacityMask;
          const baseOffset = currentWriteIndex * actionDim;
          const copyLength = Math.min(actionDim, logitsVector.length);
          for (let dimIndex = 0; dimIndex < copyLength; dimIndex++) {
            sharedBuffer[baseOffset + dimIndex] = logitsVector[dimIndex] ?? 0;
          }

          // Advance the atomic write pointer (wrap safely using 31-bit mask to avoid negative values).
          Atomics.store(
            atomicIndexView,
            0,
            (Atomics.load(atomicIndexView, 0) + 1) & 0x7fffffff,
          );
        }
      } else {
        // Fallback: local per-row ring of Float32Array rows stored in scratchLogitsRing.
        const ringCapacityMask = updatedLogitsRingCap - 1;

        for (let stepIndex = 0; stepIndex < perStepLogits.length; stepIndex++) {
          const logitsVector = perStepLogits[stepIndex];
          if (!Array.isArray(logitsVector)) continue;

          const writePos = updatedScratchLogitsRingW & ringCapacityMask;
          const targetRow = scratchLogitsRing[writePos];
          const copyLength = Math.min(actionDim, logitsVector.length);

          // Copy into the pooled Float32Array row (no allocation).
          for (let dimIndex = 0; dimIndex < copyLength; dimIndex++) {
            targetRow[dimIndex] = logitsVector[dimIndex] ?? 0;
          }

          // Advance the non-shared ring write cursor.
          updatedScratchLogitsRingW =
            (updatedScratchLogitsRingW + 1) & 0x7fffffff;
        }
      }
    }
  } catch {
    // Swallow logits postprocessing errors
  }

  // Step 4: Optionally prune saturated outputs and emit telemetry (best-effort).
  try {
    if (
      simResult?.saturationFraction &&
      simResult.saturationFraction > saturationPruneThreshold
    ) {
      pruneSaturatedHiddenOutputs(
        engineState,
        fittest,
        getNodeIndicesByType,
        collectHiddenToOutputConns,
      );
    }
  } catch {
    // Swallow saturation pruning errors
  }

  try {
    if (
      !telemetryMinimal &&
      logEvery > 0 &&
      completedGenerations % logEvery === 0
    ) {
      logGenerationTelemetry(
        engineState,
        neat,
        fittest,
        simResult,
        completedGenerations,
        safeWrite,
        actionDim,
        recentWindow,
        reducedTelemetry,
        telemetryMinimal,
        () => {
          antiCollapseRecovery(
            engineState,
            neat,
            completedGenerations,
            safeWrite,
            sampleSegmentIntoScratch,
          );
        },
        isProfilingDetailsEnabled,
        profilingStartTimestamp,
        accumulateProfilingDuration,
      );
    }
  } catch {
    // Swallow telemetry dispatch errors
  }

  const elapsed = doProfile ? readHighResolutionTime() - startTime : 0;
  return {
    generationResult: simResult,
    simTime: elapsed,
    updatedRingState: {
      logitsRingCap: updatedLogitsRingCap,
      logitsRingShared: updatedLogitsRingShared,
      scratchLogitsRingW: updatedScratchLogitsRingW,
    },
  } as SimulationOutcome;
};

/**
 * Internal evolution loop that executes generations until a stop condition or cancellation.
 *
 * Behaviour & contract:
 *  - Runs generations in a resilient, best-effort manner; internal errors are swallowed
 *    so a single failure cannot abort the whole run.
 *  - When `doProfile` is truthy the loop accumulates timing into a pooled Float64Array
 *    to avoid per-iteration allocations. The pooled buffer is reused across calls.
 *  - The helper performs side-effects (dashboard updates, persistence) in a non-fatal
 *    fashion and yields to the host when requested via `helpers.flushToFrame`.
 *
 * @param engineState - Shared engine state with scratch buffers and configuration
 * @param neat - NEAT driver instance used to perform evolution and mutation operations
 * @param opts - Normalised run options (produced by normalizeRunOptions)
 * @param lamarckianTrainingSet - Optional supervised training cases used for Lamarckian warm-start
 * @param encodedMaze - Encoded maze representation consumed by simulators
 * @param startPosition - Start coordinates for the simulated agent
 * @param exitPosition - Exit coordinates for the simulated agent
 * @param distanceMap - Optional precomputed distance map to speed simulation
 * @param helpers - Helper utilities: { flushToFrame, fs, path, safeWrite }
 * @param doProfile - When truthy collect and return millisecond timings in the result
 * @param scratchLogitsRing - Pooled logits ring buffer
 * @param logitsRingCap - Current ring capacity
 * @param logitsRingCapMax - Maximum ring capacity
 * @param actionDim - Number of action dimensions
 * @param logitsRingShared - Whether shared mode is enabled
 * @param scratchLogitsShared - Shared flat buffer (when shared mode)
 * @param scratchLogitsSharedW - Shared atomic write index
 * @param scratchLogitsRingW - Local ring write cursor
 * @param emptyVec - Empty array fallback
 * @param scratchNodeIdx - Pooled node index buffer
 * @param scratchSnapshotObj - Reusable snapshot object
 * @param scratchSnapshotTop - Reusable top-K snapshot buffer
 * @param getNodeIndicesByType - Helper to collect node indices by type
 * @param collectHiddenToOutputConns - Helper to collect connections
 * @param constants - Object containing all engine constants (DEFAULT_TRAIN_ERROR, etc.)
 *
 * @returns Promise resolving to an object:
 *  { bestNetwork, bestResult, neat, completedGenerations, totalEvolveMs, totalLamarckMs, totalSimMs, updatedRingState }
 *
 * @example
 * const runSummary = await runEvolutionLoop(
 *   state, neat, opts, trainingSet, maze, start, exit, distMap, helpers, true, ...
 * );
 */
export const runEvolutionLoop = async (
  engineState: EngineState,
  neat: NeatInstance,
  opts: EvolutionOptions,
  lamarckianTrainingSet: { input: number[]; output: number[] }[],
  encodedMaze: number[][],
  startPosition: readonly [number, number],
  exitPosition: readonly [number, number],
  distanceMap: number[][],
  helpers: LoopHelpers,
  doProfile: boolean,
  scratchLogitsRing: Float32Array[],
  logitsRingCap: number,
  logitsRingCapMax: number,
  actionDim: number,
  logitsRingShared: boolean,
  scratchLogitsShared: Float32Array | undefined,
  scratchLogitsSharedW: Int32Array | undefined,
  scratchLogitsRingW: number,
  emptyVec: NetworkInstance[],
  scratchNodeIdx: Int32Array,
  scratchSnapshotObj: Record<string, unknown>,
  scratchSnapshotTop: SnapshotEntry[],
  getNodeIndicesByType: (nodes: NetworkNode[], type: string) => number,
  collectHiddenToOutputConns: (
    hiddenNode: NetworkNode,
    nodesRef: NetworkNode[],
    outputCount: number,
  ) => NetworkConnection[],
  constants: TrainingConstants & {
    DEFAULT_TRAIN_BATCH_LARGE: number;
    FITTEST_TRAIN_ITERATIONS: number;
    TELEMETRY_MINIMAL: boolean;
    SATURATION_PRUNE_THRESHOLD: number;
    RECENT_WINDOW: number;
    REDUCED_TELEMETRY: boolean;
    DISABLE_BALDWIN: boolean;
  },
  speciesHistoryRef: number[],
) => {
  const { flushToFrame, fs, path, safeWrite } = helpers;

  // State: descriptive local names improve readability for future maintainers.
  let bestNetworkSoFar: NetworkInstance | null =
    (opts.initialBestNetwork as NetworkInstance | null) ?? null;
  let bestFitnessSoFar = -Infinity;
  let bestRunResult: IMazeRunResult | undefined = undefined;
  let stagnantGenerationsCount = 0;
  let completedGenerations = 0;
  let plateauCounter = 0;
  let simplifyMode = false;
  let simplifyRemaining = 0;
  let lastBestFitnessForPlateau = -Infinity;
  let lastCompactionGeneration = 0;

  // Mutable ring state
  let updatedLogitsRingCap = logitsRingCap;
  let updatedLogitsRingShared = logitsRingShared;
  let updatedScratchLogitsRingW = scratchLogitsRingW;

  // Profiling accumulators are stored in a pooled Float64Array to avoid
  // per-run object creation. Layout: [0]=evolveMs, [1]=lamarckMs, [2]=simMs, [3]=reserved
  const scratchBundle = engineState.scratch;
  const profileScratch: Float64Array =
    scratchBundle.profilingScratch ??
    (scratchBundle.profilingScratch = new Float64Array(4));
  profileScratch[0] = 0; // total evolve ms
  profileScratch[1] = 0; // total lamarck ms
  profileScratch[2] = 0; // total sim ms

  // Main evolution loop: resilient and best-effort. Uses descriptive names
  // and keeps allocations to a minimum.
  while (true) {
    // Step 1: cooperative cancellation check (non-allocating, safe)
    const cancelReason = checkCancellation(opts, bestRunResult);
    if (cancelReason) break;

    // Step 2: perform one generation and collect per-stage timings when enabled
    const generationOutcome = await runGeneration(
      engineState,
      neat,
      doProfile,
      opts.lamarckianIterations ?? 0,
      lamarckianTrainingSet,
      opts.lamarckianSampleSize,
      safeWrite,
      completedGenerations,
      opts.dynamicPopEnabled ?? false,
      opts.dynamicPopMax ?? 0,
      opts.plateauGenerations ?? 0,
      plateauCounter,
      opts.dynamicPopExpandInterval ?? 0,
      opts.dynamicPopExpandFactor ?? 0,
      opts.dynamicPopPlateauSlack ?? 0,
      speciesHistoryRef,
      emptyVec,
      scratchNodeIdx,
      getNodeIndicesByType,
      {
        DEFAULT_TRAIN_ERROR: constants.DEFAULT_TRAIN_ERROR,
        DEFAULT_TRAIN_RATE: constants.DEFAULT_TRAIN_RATE,
        DEFAULT_TRAIN_MOMENTUM: constants.DEFAULT_TRAIN_MOMENTUM,
        DEFAULT_TRAIN_BATCH_SMALL: constants.DEFAULT_TRAIN_BATCH_SMALL,
        DEFAULT_STD_SMALL: constants.DEFAULT_STD_SMALL,
        DEFAULT_STD_ADJUST_MULT: constants.DEFAULT_STD_ADJUST_MULT,
      },
    );

    const fallbackFittest =
      generationOutcome.fittest ??
      (typeof (neat as unknown as { getFittest?: () => NetworkInstance | null })
        .getFittest === 'function'
        ? (neat as unknown as { getFittest: () => NetworkInstance | null })
            .getFittest()
        : null) ??
      ((neat as unknown as { population?: NetworkInstance[] }).population?.[0] ??
        null);
    const fittest = fallbackFittest;
    if (!fittest) {
      safeWrite('[evolutionLoop] No fittest genome available for generation; stopping loop safely.');
      break;
    }
    if (doProfile) {
      // Use pooled scratch to accumulate totals (avoid creating new numbers/objects)
      profileScratch[0] += Number(generationOutcome.tEvolve ?? 0);
      profileScratch[1] += Number(generationOutcome.tLamarck ?? 0);
    }

    // Step 3: optional Lamarckian refinement (best-effort)
    if (!constants.DISABLE_BALDWIN) {
      try {
        fittest.train(lamarckianTrainingSet, {
          iterations: constants.FITTEST_TRAIN_ITERATIONS,
          error: constants.DEFAULT_TRAIN_ERROR,
          rate: constants.DEFAULT_TRAIN_RATE,
          momentum: constants.DEFAULT_TRAIN_MOMENTUM,
          batchSize: constants.DEFAULT_TRAIN_BATCH_LARGE,
          allowRecurrent: true,
        });
      } catch {
        // ignore training errors - non-fatal
      }
    }

    // Step 4: update per-generation counters and plateau/simplify state
    const fitnessScore = fittest.score ?? 0;
    completedGenerations += 1;

    ({ plateauCounter, lastBestFitnessForPlateau } = updatePlateauState(
      fitnessScore,
      lastBestFitnessForPlateau,
      plateauCounter,
      opts.plateauImprovementThreshold ?? 1e-6,
    ));

    ({ simplifyMode, simplifyRemaining, plateauCounter } = handleSimplifyState(
      engineState,
      neat,
      plateauCounter,
      opts.plateauGenerations ?? 0,
      opts.simplifyDuration ?? 0,
      simplifyMode,
      simplifyRemaining,
      opts.simplifyStrategy ?? 'weakWeight',
      opts.simplifyPruneFraction ?? 0,
    ));

    // Step 5: simulate the fittest genome and optionally capture sim time
    const simulationResult = simulateAndPostprocess(
      engineState,
      fittest,
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      opts.agentSimConfig?.maxSteps,
      doProfile,
      safeWrite,
      opts.reportingConfig?.logEvery ?? 10,
      completedGenerations,
      neat,
      scratchLogitsRing,
      updatedLogitsRingCap,
      logitsRingCapMax,
      actionDim,
      updatedLogitsRingShared,
      scratchLogitsShared,
      scratchLogitsSharedW,
      updatedScratchLogitsRingW,
      constants.TELEMETRY_MINIMAL,
      constants.SATURATION_PRUNE_THRESHOLD,
      constants.RECENT_WINDOW,
      constants.REDUCED_TELEMETRY,
      getNodeIndicesByType,
      collectHiddenToOutputConns,
    );
    const generationResult = simulationResult.generationResult;
    if (doProfile) profileScratch[2] += Number(simulationResult.simTime ?? 0);

    // Update ring state from simulation result
    updatedLogitsRingCap = simulationResult.updatedRingState.logitsRingCap;
    updatedLogitsRingShared =
      simulationResult.updatedRingState.logitsRingShared;
    updatedScratchLogitsRingW =
      simulationResult.updatedRingState.scratchLogitsRingW;

    // Step 6: update best-so-far and dashboard periodically
    if (fitnessScore > bestFitnessSoFar) {
      bestFitnessSoFar = fitnessScore;
      bestNetworkSoFar = fittest;
      bestRunResult = generationResult;
      stagnantGenerationsCount = 0;
      try {
        await updateDashboardAndMaybeFlush(
          opts.mazeConfig.maze,
          generationResult,
          fittest,
          completedGenerations,
          neat,
          opts.reportingConfig?.dashboardManager,
          flushToFrame,
        );
      } catch {
        // best-effort: ignore dashboard errors
      }
    } else {
      stagnantGenerationsCount += 1;
      if (completedGenerations % (opts.reportingConfig?.logEvery ?? 10) === 0) {
        try {
          await updateDashboardPeriodic(
            opts.mazeConfig.maze,
            bestRunResult,
            bestNetworkSoFar,
            completedGenerations,
            neat,
            opts.reportingConfig?.dashboardManager,
            flushToFrame,
          );
        } catch {
          // best-effort
        }
      }
    }

    // Step 7: persist snapshot if configured (best-effort)
    persistSnapshotIfNeeded(
      engineState,
      fs as { writeFileSync?: (path: string, data: string) => void } | null,
      path as { join?: (...paths: string[]) => string } | null,
      opts.persistDir,
      opts.persistTopK ?? 0,
      completedGenerations,
      opts.persistEvery ?? 0,
      neat,
      bestFitnessSoFar,
      simplifyMode,
      plateauCounter,
      scratchSnapshotObj,
      scratchSnapshotTop,
      collectTelemetryTail,
      getSortedIndicesByScore,
      isProfilingDetailsEnabled,
      profilingStartTimestamp,
      accumulateProfilingDuration,
    );

    // Step 8: check stop conditions
    const stopReason = await checkStopConditions(
      bestRunResult,
      bestNetworkSoFar,
      opts.mazeConfig.maze,
      completedGenerations,
      neat,
      opts.reportingConfig?.dashboardManager,
      flushToFrame,
      opts.minProgressToPass ?? 0,
      opts.autoPauseOnSolve ?? false,
      opts.stopOnlyOnSolve ?? false,
      stagnantGenerationsCount,
      opts.maxStagnantGenerations ?? 0,
      opts.maxGenerations ?? 0,
    );
    if (stopReason) break;

    // Step 9: periodic memory compaction and scratch shrinking
    if (
      (opts.memoryCompactionInterval ?? 0) > 0 &&
      completedGenerations - lastCompactionGeneration >=
        (opts.memoryCompactionInterval ?? 0)
    ) {
      const removedDisabled = compactPopulation(engineState, neat);
      if (removedDisabled > 0) {
        const currentPopulationSize = Array.isArray(neat?.population)
          ? neat.population.length
          : 0;
        maybeShrinkScratch(engineState, currentPopulationSize);
        safeWrite(
          `[COMPACT] gen=${completedGenerations} removedDisabledConns=${removedDisabled}\n`,
        );
      }
      lastCompactionGeneration = completedGenerations;
    }

    // Step 10: optionally yield to host between generations
    if (opts.reportingConfig?.paceEveryGeneration) {
      try {
        await flushToFrame();
      } catch {
        // ignore host-yield failures
      }
    }
  }

  // Prepare totals to return (read from pooled scratch to avoid ephemeral numbers earlier)
  const totalEvolveMs = Number(profileScratch[0] || 0);
  const totalLamarckMs = Number(profileScratch[1] || 0);
  const totalSimMs = Number(profileScratch[2] || 0);

  return {
    bestNetwork: bestNetworkSoFar,
    bestResult: bestRunResult,
    neat,
    completedGenerations,
    totalEvolveMs,
    totalLamarckMs,
    totalSimMs,
    updatedRingState: {
      logitsRingCap: updatedLogitsRingCap,
      logitsRingShared: updatedLogitsRingShared,
      scratchLogitsRingW: updatedScratchLogitsRingW,
    },
  } as EvolutionLoopResult;
};
