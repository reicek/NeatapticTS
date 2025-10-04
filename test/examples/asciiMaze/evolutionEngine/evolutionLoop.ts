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

import {
  makeFlushToFrame,
  initPersistence,
  makeSafeWriter,
} from './setupHelpers';

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
export function checkCancellation(
  options: any,
  bestResult?: any,
): string | undefined {
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
  } catch (err) {
    // Best-effort: swallow any unexpected errors to avoid breaking the caller.
  }
  // No cancellation detected.
  return undefined;
}

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
export function prepareLoopHelpers(opts: any, scratchBundle: any): any {
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
}

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
export async function checkStopConditions(
  bestResult: any,
  bestNetwork: any,
  maze: any,
  completedGenerations: number,
  neat: any,
  dashboardManager: any,
  flushToFrame: () => Promise<void>,
  minProgressToPass: number,
  autoPauseOnSolve: boolean,
  stopOnlyOnSolve: boolean,
  stagnantGenerations: number,
  maxStagnantGenerations: number,
  maxGenerations: number,
): Promise<string | undefined> {
  // Local convenience aliases for small, hot checks.
  const hasBest = Boolean(bestResult);
  const shouldConsiderStops = !stopOnlyOnSolve;

  // --- 1) Solved check ---
  if (
    bestResult?.success &&
    bestResult.progress >= (minProgressToPass ?? 0)
  ) {
    // Attempt dashboard update and yield; swallow any errors.
    try {
      dashboardManager?.update?.(
        maze,
        bestResult,
        bestNetwork,
        completedGenerations,
        neat,
      );
    } catch {}

    try {
      await flushToFrame?.();
    } catch {}

    // Optionally set a cooperative pause and emit a small event for UIs to react to.
    if (autoPauseOnSolve) {
      try {
        if (typeof window !== 'undefined') {
          (window as any).asciiMazePaused = true;
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
          } catch {}
        }
      } catch {}
    }

    if (hasBest) (bestResult as any).exitReason = 'solved';
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
        bestNetwork,
        completedGenerations,
        neat,
      );
    } catch {}
    try {
      await flushToFrame?.();
    } catch {}
    if (hasBest) (bestResult as any).exitReason = 'stagnation';
    return 'stagnation';
  }

  // --- 3) Max generations check ---
  if (
    shouldConsiderStops &&
    isFinite(maxGenerations) &&
    completedGenerations >= maxGenerations
  ) {
    if (hasBest) (bestResult as any).exitReason = 'maxGenerations';
    return 'maxGenerations';
  }

  // No stop condition matched.
  return undefined;
}

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
export function persistSnapshotIfNeeded(
  engineState: any,
  fs: any,
  pathModule: any,
  persistDir: string | undefined,
  persistTopK: number,
  completedGenerations: number,
  persistEvery: number,
  neat: any,
  bestFitness: number,
  simplifyMode: boolean,
  plateauCounter: number,
  scratchSnapshotObj: any,
  scratchSnapshotTop: any[],
  collectTelemetryTailFn: (state: any, neat: any, count: number) => any,
  getSortedIndicesByScoreFn: (state: any, population: any[]) => number[],
  isProfilingDetailsEnabledFn: (state: any) => boolean,
  profilingStartTimestampFn: (state: any) => number,
  accumulateProfilingDurationFn: (
    state: any,
    label: string,
    duration: number,
  ) => void,
) {
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

  if (
    !neat ||
    !Array.isArray(neat.population) ||
    neat.population.length === 0
  )
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
    const populationRef: any[] = neat.population ?? [];
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
        typeof genome?.toJSON === 'function' ? genome.toJSON() : undefined;
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
}

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
export async function updateDashboardAndMaybeFlush(
  maze: any,
  result: any,
  network: any,
  completedGenerations: number,
  neat: any,
  dashboardManager: any,
  flushToFrame?: () => Promise<void>,
) {
  // Step 0: Defensive local aliases with descriptive names to improve readability in hot paths.
  const manager = dashboardManager;
  const yieldFrame = flushToFrame;

  // Step 1: Call dashboard update if provided. This is intentionally best-effort and must not
  // throw — any error from the dashboard implementation is swallowed to keep the engine stable.
  if (manager?.update && typeof manager.update === 'function') {
    try {
      // Use the stable argument order so dashboard implementations are consistent.
      manager.update(maze, result, network, completedGenerations, neat);
    } catch (_updateError) {
      // Swallow dashboard errors — telemetry/UI must not break evolution.
    }
  }

  // Step 2: Optionally yield to the host/frame scheduler. Guard typeof to avoid an accidental
  // Promise rejection when a caller passes undefined/non-function.
  if (typeof yieldFrame === 'function') {
    try {
      await yieldFrame();
    } catch (_flushError) {
      // Swallow flush errors; a failed frame yield is non-fatal for the evolution loop.
    }
  }
}

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
export async function updateDashboardPeriodic(
  maze: any,
  bestResult: any,
  bestNetwork: any,
  completedGenerations: number,
  neat: any,
  dashboardManager: any,
  flushToFrame?: () => Promise<void>,
) {
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
      bestNetwork,
      completedGenerations,
      neat,
    );
  } catch (updateError) {
    // Intentionally ignore update errors — dashboard should not crash the engine.
  }

  // Step 3: Optionally yield to the host renderer/scheduler.
  if (typeof frameFlush === 'function') {
    try {
      await frameFlush();
    } catch (flushError) {
      // Ignore flush errors — non-critical for engine progress.
    }
  }
}

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
export function emitProfileSummary(
  engineState: any,
  safeWrite: (msg: string) => void,
  completedGenerations: number,
  totalEvolveMs: number,
  totalLamarckMs: number,
  totalSimMs: number,
  isProfilingDetailsEnabledFn: (state: any) => boolean,
  getProfilingAccumulatorsFn: (state: any) => any,
) {
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
        ? (detailAccum.telemetry / denom).toFixed(2)
        : '0.00';
      const avgSimplify = Number.isFinite(detailAccum?.simplify)
        ? (detailAccum.simplify / denom).toFixed(2)
        : '0.00';
      const avgSnapshot = Number.isFinite(detailAccum?.snapshot)
        ? (detailAccum.snapshot / denom).toFixed(2)
        : '0.00';
      const avgPrune = Number.isFinite(detailAccum?.prune)
        ? (detailAccum.prune / denom).toFixed(2)
        : '0.00';
      safeWrite(
        `[PROFILE_DETAIL] avgTelemetry=${avgTelemetry} avgSimplify=${avgSimplify} avgSnapshot=${avgSnapshot} avgPrune=${avgPrune}\n`,
      );
    }
  } catch {
    // Best-effort: never let profiling output disrupt the run.
  }
}

