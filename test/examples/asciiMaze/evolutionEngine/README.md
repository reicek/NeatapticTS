# evolutionEngine

## evolutionEngine/sampling.ts

### sampling

Sampling and history helpers extracted from the ASCII maze evolution façade.

Responsibilities:
1. Provide allocation-light array sampling utilities that reuse the shared `EngineState` scratch pools.
2. Expose history helpers that mirror the façade behaviour while keeping pooled buffers centralised.
3. Centralise RNG parameter resolution for sampling paths to keep behaviour deterministic under shared state.

### getTail

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, source: T[] | undefined, count: number) => T[]`

### pushHistory

`(buffer: T[] | undefined, value: T, maxLength: number) => T[]`

### sampleArray

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, source: T[], sampleCount: number) => T[]`

### sampleIntoScratch

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, source: T[], sampleCount: number) => number`

### sampleSegmentIntoScratch

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, source: T[], segmentStart: number, sampleCount: number) => number`

## evolutionEngine/engineState.ts

### engineState

Centralised shared state for the ASCII maze evolution façade.

Responsibilities:
1. Define the scratch-buffer schema consumed by telemetry, population, and inspection helpers.
2. Expose runtime toggle state (`EngineToggleState`) that drives optional phases and telemetry density.
3. Provide factory and maintenance helpers (`createEngineState`, `initialiseTelemetryScratch`, `ensureVisitedHashCapacity`, `ensureRngCacheBatch`, `reseedRngState`) that size buffers and keep deterministic RNG state in sync.
4. Export the project-wide singleton `engineState` so extracted modules can share the façade’s pooled resources while still accepting injected state for testing.

Callers mutate the returned scratch instances in place to avoid per-generation allocations; higher-level modules should treat the helpers as the sole entry point for sizing or resetting shared buffers.

### createEngineState

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState`

### createProfilingState

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineProfilingState`

### createScratchState

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineScratchState`

### createToggleState

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineToggleState`

### DEFAULT_RNG_CACHE_BATCH_SIZE

### DEFAULT_VISITED_HASH_LOAD_FACTOR

### EngineProfilingState

Aggregated profiling configuration and accumulators shared across the evolution run.

### EngineScratchState

Centralised shared state for the ASCII maze evolution façade.

Responsibilities:
1. Define the scratch-buffer schema consumed by telemetry, population, and inspection helpers.
2. Expose runtime toggle state (`EngineToggleState`) that drives optional phases and telemetry density.
3. Provide factory and maintenance helpers (`createEngineState`, `initialiseTelemetryScratch`, `ensureVisitedHashCapacity`, `ensureRngCacheBatch`, `reseedRngState`) that size buffers and keep deterministic RNG state in sync.
4. Export the project-wide singleton `engineState` so extracted modules can share the façade’s pooled resources while still accepting injected state for testing.

Callers mutate the returned scratch instances in place to avoid per-generation allocations; higher-level modules should treat the helpers as the sole entry point for sizing or resetting shared buffers.

### EngineState

Shared engine state instance combining pooled scratch buffers with toggle flags.

### EngineToggleState

Runtime switches that adjust telemetry verbosity and optional training phases.

### ensureRngCacheBatch

`(parameters: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").RngCacheParameters, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").RngCacheHandles`

### ensureVisitedHashCapacity

`(targetEntryCount: number, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").VisitedHashScratchHandles`

### initialiseTelemetryScratch

`(request: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").TelemetryScratchRequest, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").TelemetryScratchHandles`

### reseedRngState

`(seed: number, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => number`

### RngCacheHandles

Handles returned after ensuring the RNG cache is ready for consumption.

### RngCacheParameters

Parameters controlling the RNG cache refill process.

### setBaldwinPhaseDisabledFlag

`(isDisabled: boolean) => void`

### setReducedTelemetryFlag

`(isEnabled: boolean) => void`

### setTelemetryMinimalFlag

`(isMinimal: boolean) => void`

### TelemetryScratchHandles

Collection of scratch buffers handed back after initialisation for convenience.

### TelemetryScratchRequest

Configuration describing which telemetry scratch buffers require capacity guarantees.

### VisitedHashScratchHandles

Handles exposed after ensuring the visited-coordinate hash table capacity.

## evolutionEngine/rngAndTiming.ts

### rngAndTiming

RNG and timing helpers extracted from the evolution façade.
@remarks
All helpers accept an {@link EngineState} instance so they can read and mutate
shared state without relying on class statics.

### accumulateProfilingDuration

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, category: string, deltaMs: number) => void`

### clearDeterministicMode

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => void`

### drawFastRandom

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, parameters: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").RngCacheParameters) => number`

### getProfilingAccumulators

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => Record<string, number>`

### isDeterministicModeEnabled

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => boolean`

### isProfilingDetailsEnabled

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => boolean`

### profilingStartTimestamp

`() => number`

### readHighResolutionTime

`() => number`

### resolveRngParameters

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").RngCacheParameters`

### setDeterministicMode

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, seed: number | undefined) => void`

## evolutionEngine/scratchPools.ts

### scratchPools

Scratch pool management helpers extracted from the maze evolution façade.

These utilities centralise the logic that grows and shrinks pooled buffers attached to the
shared {@link EngineState}. They keep the façade lean by encapsulating heuristics for logits
ring sizing, telemetry scratch sizing, and connection flag pooling.

### allocateLogitsRing

`(capacity: number, actionDimension: number) => Float32Array<ArrayBufferLike>[]`

### ensureConnFlagsCapacity

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, minimumCapacity: number) => Int8Array<ArrayBufferLike> | null`

### ensureLogitsRingCapacity

`(capacityRequest: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/scratchPools").LogitsRingCapacityOptions) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/scratchPools").LogitsRingCapacityResult`

### ensureScratchCapacity

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, request: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/scratchPools").ScratchCapacityRequest) => void`

### initialiseSharedLogitsRing

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, config: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/scratchPools").SharedLogitsConfig) => boolean`

### LogitsRingCapacityOptions

Shape describing the parameters used when ensuring the logits ring capacity.

### LogitsRingCapacityResult

Result returned after resizing the logits ring.

### maybeShrinkScratch

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, populationSize: number) => void`

### ScratchCapacityRequest

Parameters describing the scratch sizing requirements for ensureScratchCapacity.

### SharedLogitsConfig

Parameters passed when attempting to initialise the shared logits ring buffers.

## evolutionEngine/setupHelpers.ts

### setupHelpers

setupHelpers.ts

Environment and setup utilities for the evolution engine.
   // Return  return async (): Promise<void> => {
    // Polling loop: after each tick, if the cooperative pause flag is set, wait another tick.
    // This keeps CPU usage minimal while allowing the host to pause/resume the evolution loop.
    while (true) {
      await preferredTick();
      // Note: using a permissive read of the global pause flag; undefined => not paused.
      if (!(globalThis as Record<string, unknown>).asciiMazePaused) return;
      // otherwise continue and await another tick before re-checking
    }
  };
};flush function used by the evolution loop.
  return async (): Promise<void> => {
    // Polling loop: after each tick, if the cooperative pause flag is set, wait another tick.
    // This keeps CPU usage minimal while allowing the host to pause/resume the evolution loop.
    while (true) {
      await preferredTick();
      // Note: using a permissive read of the global pause flag; undefined => not paused.
      if (!(globalThis as Record<string, unknown>).asciiMazePaused) return;
      // otherwise continue and await another tick before re-checking
    }
  };
};ibilities:
- Create cooperative frame-yielding functions for async evolution loops
- Initialize Node.js filesystem persistence helpers (fs, path)
- Build resilient logging writers with fallback chains

All functions are pure/side-effect-free except where explicitly documented.

@module setupHelpers

### DashboardManagerLike

Dashboard manager shape for logging (optional log function).

### FilesystemModule

setupHelpers.ts

Environment and setup utilities for the evolution engine.
   // Return  return async (): Promise<void> => {
    // Polling loop: after each tick, if the cooperative pause flag is set, wait another tick.
    // This keeps CPU usage minimal while allowing the host to pause/resume the evolution loop.
    while (true) {
      await preferredTick();
      // Note: using a permissive read of the global pause flag; undefined => not paused.
      if (!(globalThis as Record<string, unknown>).asciiMazePaused) return;
      // otherwise continue and await another tick before re-checking
    }
  };
};flush function used by the evolution loop.
  return async (): Promise<void> => {
    // Polling loop: after each tick, if the cooperative pause flag is set, wait another tick.
    // This keeps CPU usage minimal while allowing the host to pause/resume the evolution loop.
    while (true) {
      await preferredTick();
      // Note: using a permissive read of the global pause flag; undefined => not paused.
      if (!(globalThis as Record<string, unknown>).asciiMazePaused) return;
      // otherwise continue and await another tick before re-checking
    }
  };
};ibilities:
- Create cooperative frame-yielding functions for async evolution loops
- Initialize Node.js filesystem persistence helpers (fs, path)
- Build resilient logging writers with fallback chains

All functions are pure/side-effect-free except where explicitly documented.

### initPersistence

`(persistDir: string | undefined) => { fs: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/setupHelpers").FilesystemModule | null; path: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/setupHelpers").PathModule | null; }`

### makeFlushToFrame

`() => () => Promise<void>`

### makeSafeWriter

`(dashboardManager: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/setupHelpers").DashboardManagerLike | undefined) => (msg: string) => void`

### PathModule

Minimal path module shape for type safety (Node.js path module subset).

## evolutionEngine/evolutionLoop.ts

### evolutionLoop

Evolution Loop Module

Purpose:
-------
Provides utilities for running the main NEAT evolution loop, including
generation orchestration, cancellation checking, and loop helper preparation.

This module encapsulates:
 - Cancellation detection (AbortSignal and legacy cancellation API)
 - Loop helper preparation (frame flushing, persistence, logging)
 - Generation execution and orchestration
 - Stop condition evaluation

ES2023 Policy:
-------------
- Uses nullish coalescing `??` and optional chaining `?.`
- Descriptive variable names (no short identifiers)
- Async/await for generation loops
- Best-effort error handling (swallow non-fatal errors)

@module evolutionEngine/evolutionLoop

### checkCancellation

`(options: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").EvolutionOptions, bestResult: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult | undefined) => string | undefined`

### checkStopConditions

`(bestResult: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult | undefined, bestNetwork: import("C:/NeatapticTS/src/architecture/network").default | null, maze: string[], completedGenerations: number, neat: import("C:/NeatapticTS/src/neat").default, dashboardManager: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IDashboardManager | undefined, flushToFrame: () => Promise<void>, minProgressToPass: number, autoPauseOnSolve: boolean, stopOnlyOnSolve: boolean, stagnantGenerations: number, maxStagnantGenerations: number, maxGenerations: number) => Promise<string | undefined>`

### emitProfileSummary

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, safeWrite: (msg: string) => void, completedGenerations: number, totalEvolveMs: number, totalLamarckMs: number, totalSimMs: number, isProfilingDetailsEnabledFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => boolean, getProfilingAccumulatorsFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").ProfilingAccumulators) => void`

### EvolutionLoopResult

Evolution loop result

### GenerationOutcome

Generation outcome with profiling timings

### MazeWindow

Evolution Loop Module

Purpose:
-------
Provides utilities for running the main NEAT evolution loop, including
generation orchestration, cancellation checking, and loop helper preparation.

This module encapsulates:
 - Cancellation detection (AbortSignal and legacy cancellation API)
 - Loop helper preparation (frame flushing, persistence, logging)
 - Generation execution and orchestration
 - Stop condition evaluation

ES2023 Policy:
-------------
- Uses nullish coalescing `??` and optional chaining `?.`
- Descriptive variable names (no short identifiers)
- Async/await for generation loops
- Best-effort error handling (swallow non-fatal errors)

### MutableMazeResult

Mutable result object with exitReason field

### persistSnapshotIfNeeded

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, fs: { writeFileSync?: ((path: string, data: string) => void) | undefined; } | null, pathModule: { join?: ((...paths: string[]) => string) | undefined; } | null, persistDir: string | undefined, persistTopK: number, completedGenerations: number, persistEvery: number, neat: import("C:/NeatapticTS/src/neat").default, bestFitness: number, simplifyMode: boolean, plateauCounter: number, scratchSnapshotObj: Record<string, unknown>, scratchSnapshotTop: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").SnapshotEntry[], collectTelemetryTailFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: import("C:/NeatapticTS/src/neat").default, count: number) => unknown, getSortedIndicesByScoreFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, population: import("C:/NeatapticTS/src/architecture/network").default[]) => number[], isProfilingDetailsEnabledFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => boolean, profilingStartTimestampFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => number, accumulateProfilingDurationFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, label: string, duration: number) => void) => void`

### prepareLoopHelpers

`(opts: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").EvolutionOptions, scratchBundle: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").ScratchBundle) => import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").LoopHelpers`

### runEvolutionLoop

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: import("C:/NeatapticTS/src/neat").default, opts: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").EvolutionOptions, lamarckianTrainingSet: { input: number[]; output: number[]; }[], encodedMaze: number[][], startPosition: readonly [number, number], exitPosition: readonly [number, number], distanceMap: number[][], helpers: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").LoopHelpers, doProfile: boolean, scratchLogitsRing: Float32Array<ArrayBufferLike>[], logitsRingCap: number, logitsRingCapMax: number, actionDim: number, logitsRingShared: boolean, scratchLogitsShared: Float32Array<ArrayBufferLike> | undefined, scratchLogitsSharedW: Int32Array<ArrayBufferLike> | undefined, scratchLogitsRingW: number, emptyVec: import("C:/NeatapticTS/src/architecture/network").default[], scratchNodeIdx: Int32Array<ArrayBufferLike>, scratchSnapshotObj: Record<string, unknown>, scratchSnapshotTop: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").SnapshotEntry[], getNodeIndicesByType: (nodes: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode[], type: string) => number, collectHiddenToOutputConns: (hiddenNode: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode, nodesRef: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode[], outputCount: number) => import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkConnection[], constants: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").TrainingConstants & { DEFAULT_TRAIN_BATCH_LARGE: number; FITTEST_TRAIN_ITERATIONS: number; TELEMETRY_MINIMAL: boolean; SATURATION_PRUNE_THRESHOLD: number; RECENT_WINDOW: number; REDUCED_TELEMETRY: boolean; DISABLE_BALDWIN: boolean; }, speciesHistoryRef: number[]) => Promise<EvolutionLoopResult>`

### runGeneration

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: import("C:/NeatapticTS/src/neat").default, doProfile: boolean, lamarckianIterations: number, lamarckianTrainingSet: { input: number[]; output: number[]; }[], lamarckianSampleSize: number | undefined, safeWrite: (msg: string) => void, completedGenerations: number, dynamicPopEnabled: boolean, dynamicPopMax: number, plateauGenerations: number, plateauCounter: number, dynamicPopExpandInterval: number, dynamicPopExpandFactor: number, dynamicPopPlateauSlack: number, speciesHistoryRef: number[], emptyVec: import("C:/NeatapticTS/src/architecture/network").default[], scratchNodeIdx: Int32Array<ArrayBufferLike>, getNodeIndicesByType: (nodes: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode[], type: string) => number, constants: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").TrainingConstants) => Promise<GenerationOutcome>`

### RuntimeNetworkInstance

Network with dynamic runtime properties

### SimResultWithOutputs

Simulation result with step outputs

### simulateAndPostprocess

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, fittest: import("C:/NeatapticTS/src/architecture/network").default, encodedMaze: number[][], startPosition: readonly [number, number], exitPosition: readonly [number, number], distanceMap: number[][], maxSteps: number | undefined, doProfile: boolean, safeWrite: (msg: string) => void, logEvery: number, completedGenerations: number, neat: import("C:/NeatapticTS/src/neat").default, scratchLogitsRing: Float32Array<ArrayBufferLike>[], logitsRingCap: number, logitsRingCapMax: number, actionDim: number, logitsRingShared: boolean, scratchLogitsShared: Float32Array<ArrayBufferLike> | undefined, scratchLogitsSharedW: Int32Array<ArrayBufferLike> | undefined, scratchLogitsRingW: number, telemetryMinimal: boolean, saturationPruneThreshold: number, recentWindow: number, reducedTelemetry: boolean, getNodeIndicesByType: (nodes: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode[], type: string) => number, collectHiddenToOutputConns: (hiddenNode: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode, nodesRef: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode[], outputCount: number) => import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkConnection[]) => import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").SimulationResult`

### SimulationOutcome

Simulation result with profiling and ring state

### updateDashboardAndMaybeFlush

`(maze: string[], result: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult | undefined, network: import("C:/NeatapticTS/src/architecture/network").default | null, completedGenerations: number, neat: import("C:/NeatapticTS/src/neat").default, dashboardManager: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IDashboardManager | undefined, flushToFrame: (() => Promise<void>) | undefined) => Promise<void>`

### updateDashboardPeriodic

`(maze: string[], bestResult: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult | undefined, bestNetwork: import("C:/NeatapticTS/src/architecture/network").default | null, completedGenerations: number, neat: import("C:/NeatapticTS/src/neat").default, dashboardManager: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IDashboardManager | undefined, flushToFrame: (() => Promise<void>) | undefined) => Promise<void>`

## evolutionEngine/optionsAndSetup.ts

### optionsAndSetup

Options and Setup Module

Purpose:
-------
Provides utilities for normalizing run options, preparing maze environment,
and orchestrating NEAT driver creation with population seeding.

This module encapsulates:
 - Run options validation and normalization with sensible defaults
 - Environment preparation (maze encoding, distance maps, I/O sizing)
 - NEAT driver creation and population seeding orchestration

ES2023 Policy:
-------------
- Uses nullish coalescing `??` for default values (never `||`)
- Descriptive variable names (no short identifiers)
- Optional chaining `?.` for safe property access
- Spread operator for object composition

@module evolutionEngine/optionsAndSetup

### createAndSeedNeat

`(opts: any, inputSize: number, outputSize: number, fitnessContext: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IFitnessEvaluationContext, scratchPopClone: any[], scratchSample: any[]) => any`

### normalizeRunOptions

`(options: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IRunMazeEvolutionOptions, setDeterministic: (seed: number) => void, setReducedTelemetry: (enabled: boolean) => void, setMinimalTelemetry: (enabled: boolean) => void, setDisableBaldwin: (disabled: boolean) => void) => any`

### prepareEnvironmentForRun

`(opts: any, scratchBundle: any) => any`

## evolutionEngine/telemetryMetrics.ts

### telemetryMetrics

Telemetry logging and metric computation helpers extracted from the ASCII maze evolution engine.

The helpers in this module operate on the shared {@link EngineState} scratch buffers to avoid
per-call allocations while keeping the main façade lighter. All telemetry is best-effort: any
internal error is swallowed so that logging never impacts the evolution loop.

### ActionEntropyStats

Structure describing the result of action-entropy computation.

### collectTelemetryTail

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: unknown, tailLength: number) => unknown`

### GenerationResult

Minimal structure representing a generation evolution result.
Expected to have a path property containing the movement history.

### LOG_TAG_ACTION_ENTROPY

### LOG_TAG_LOGITS

### LOG_TAG_OUTPUT_BIAS

### logActionEntropy

`(__0: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/telemetryMetrics").LogActionEntropyParams) => void`

### LogActionEntropyParams

Parameters required to emit action-entropy telemetry.

### logDiversity

`(__0: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/telemetryMetrics").LogDiversityParams) => void`

### LogDiversityParams

Parameters required to emit population diversity telemetry.

### logExploration

`(__0: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/telemetryMetrics").LogExplorationParams) => void`

### LogExplorationParams

Parameters required to emit exploration telemetry.

### logGenerationTelemetry

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: import("C:/NeatapticTS/src/neat").default, fittest: import("C:/NeatapticTS/src/architecture/network").default | undefined, genResult: GenerationResult | undefined, generationIndex: number, writeLog: (msg: string) => void, actionDimension: number, recentWindow: number, reducedTelemetry: boolean, telemetryMinimal: boolean, onCollapseRecovery: () => void, isProfilingDetailsEnabledFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => boolean, profilingStartTimestampFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => number, accumulateProfilingDurationFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, label: string, duration: number) => void) => void`

### LogitStatsParams

Input parameters for computing logit statistics.

### LogitStatsResult

Structure describing aggregated logit statistics.

### logLogitsAndCollapse

`(__0: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/telemetryMetrics").LogLogitsParams) => void`

### LogLogitsParams

Parameters required to emit logits statistics, perform collapse detection and trigger recovery.

### LogOutputBiasParams

Parameters required to emit output-bias telemetry.

### logOutputBiasStats

`(__0: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/telemetryMetrics").LogOutputBiasParams) => void`

### RuntimeConnection

Runtime type for network connection with dynamic properties.
Connections may have enabled, weight, and other runtime-added fields.

### RuntimeGenome

Runtime type for NEAT genome/network with dynamic properties.
Genomes may have nodes, connections, score, species and runtime tracking fields.

### RuntimeNeat

Runtime type for NEAT instance with dynamic properties.
NEAT instances may have population, getTelemetry and other runtime methods.

### RuntimeNode

Runtime type for network node with dynamic properties.
Nodes may have type, bias, and other runtime-added fields.

### TelemetryBaseParams

Parameters shared by telemetry helpers that require access to the shared state and writer.

### TelemetryWriter

`(message: string) => void`

Writer signature reused across telemetry helpers.

## evolutionEngine/neatConfiguration.ts

### neatConfiguration

NEAT Configuration Module

Purpose:
-------
Provides utilities for instantiating and seeding NEAT (NeuroEvolution of Augmenting Topologies)
instances with standardized configuration and optional initial populations.

This module encapsulates:
 - NEAT driver creation with opinionated defaults (elitism, provenance, mutation operators)
 - Population seeding with defensive cloning and best-effort error handling
 - Configuration normalization and validation

ES2023 Policy:
-------------
- Uses nullish coalescing `??` for default values (never `||`)
- Descriptive variable names (no short identifiers like `i`, `c`, `p`)
- Optional chaining `?.` for safe property access
- Spread operator for array/object operations

@module evolutionEngine/neatConfiguration

### createNeat

`(inputCount: number, outputCount: number, fitnessCallback: (net: import("C:/NeatapticTS/src/architecture/network").default) => number, cfg: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/neatConfiguration").NeatConfig | undefined) => import("C:/NeatapticTS/src/neat").default`

### NeatConfig

NEAT configuration object shape for type safety.

### seedInitialPopulation

`(neat: import("C:/NeatapticTS/src/neat").default, initialPopulation: import("C:/NeatapticTS/src/architecture/network").default[] | undefined, initialBestNetwork: import("C:/NeatapticTS/src/architecture/network").default | undefined, targetPopSize: number, scratchPopClone: import("C:/NeatapticTS/src/architecture/network").default[]) => import("C:/NeatapticTS/src/architecture/network").default[]`

## evolutionEngine/networkInspection.ts

### networkInspection

Network Inspection Module

Purpose:
-------
Provides utilities for inspecting and analyzing neural network topology,
including node classification, activation function detection, and connection
analysis (recurrent/gated detection).

This module encapsulates:
 - Node classification (input/hidden/output buckets)
 - Activation function name gathering
 - Recurrent and gated connection detection
 - Network structure printing for debugging

ES2023 Policy:
-------------
- Uses nullish coalescing `??` and optional chaining `?.`
- Descriptive variable names (no short identifiers)
- Pooled scratch buffers to avoid allocations
- Best-effort error handling (swallow non-fatal errors)

@module evolutionEngine/networkInspection

### printNetworkStructure

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, network: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork) => void`

### swallowError

`(error: unknown) => void`

## evolutionEngine/populationPruning.ts

### populationPruning

Population pruning and network warm-start helpers for the ASCII maze evolution engine.

This module centralizes genome connection pruning (simplification phase), output bias maintenance,
and warm-start initialization logic. All helpers operate on the shared {@link EngineState} scratch
buffers to avoid per-call allocations while keeping pruning logic modular and testable.

Responsibilities:
1. Apply simplify-phase pruning to entire populations via strategy-driven connection disabling.
2. Collect, sort, and selectively disable weak connections based on absolute weight.
3. Initialize compass warm-start wiring for directional inputs.
4. Re-center and clamp output node biases to prevent drift.
5. Provide allocation-light helpers that reuse pooled buffers (connection candidates, node indices).

All functions are best-effort: internal errors are swallowed to avoid destabilizing the evolution loop.

### applyCompassWarmStart

`(__0: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/populationPruning").ApplyCompassWarmStartParams) => void`

### ApplyCompassWarmStartParams

Parameters for warm-starting compass wiring.

### ApplySimplifyPruningParams

Parameters for applying simplify pruning to a population.

### applySimplifyPruningToPopulation

`(__0: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/populationPruning").ApplySimplifyPruningParams) => void`

### centerOutputBiases

`(__0: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/populationPruning").CenterOutputBiasesParams) => void`

### CenterOutputBiasesParams

Parameters for re-centering output biases.

## evolutionEngine/trainingWarmStart.ts

### trainingWarmStart

trainingWarmStart.ts

Lamarckian warm-start and population pretraining subsystem.

Responsibilities:
- Build supervised training datasets for compass-guided navigation
- Apply Lamarckian backpropagation training to populations
- Adjust output biases after training to maintain exploration
- Orchestrate conditional warm-start pretraining with buffer management

All functions are pure/side-effect-free except where explicitly documented.
Mutations are limited to NEAT population networks and engine scratch buffers.

@module trainingWarmStart

### adjustOutputBiasesAfterTraining

`(network: import("C:/NeatapticTS/src/architecture/network").default, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, constants: { DEFAULT_STD_SMALL: number; DEFAULT_STD_ADJUST_MULT: number; }, scratchNodeIdx: Int32Array<ArrayBufferLike>, getNodeIndicesByType: (nodes: any[], nodeType: string) => number) => void`

### applyLamarckianTraining

`(neat: import("C:/NeatapticTS/src/neat").default, trainingSet: any[], iterations: number, sampleSize: number | undefined, safeWrite: (msg: string) => void, profileEnabled: boolean, completedGenerations: number, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, constants: { DEFAULT_TRAIN_ERROR: number; DEFAULT_TRAIN_RATE: number; DEFAULT_TRAIN_MOMENTUM: number; DEFAULT_TRAIN_BATCH_SMALL: number; }, adjustOutputBiases: (network: any) => void) => number`

### buildLamarckianTrainingSet

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, constants: { TRAIN_OUT_PROB_HIGH: number; TRAIN_OUT_PROB_LOW: number; PROGRESS_MEDIUM: number; PROGRESS_STRONG: number; PROGRESS_JUNCTION: number; PROGRESS_FOURWAY: number; PROGRESS_REGRESS: number; PROGRESS_MIN_SIGNAL: number; PROGRESS_MILD_REGRESS: number; DEFAULT_JITTER_PROB: number; AUGMENT_JITTER_BASE: number; AUGMENT_JITTER_RANGE: number; AUGMENT_PROGRESS_JITTER_PROB: number; AUGMENT_PROGRESS_DELTA_RANGE: number; AUGMENT_PROGRESS_DELTA_HALF: number; RNG_PARAMETERS: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").RngCacheParameters; }) => { input: number[]; output: number[]; }[]`

### pretrainPopulationWarmStart

`(neat: import("C:/NeatapticTS/src/neat").default, lamarckianTrainingSet: any[], constants: { PRETRAIN_MAX_ITER: number; PRETRAIN_BASE_ITER: number; DEFAULT_TRAIN_ERROR: number; DEFAULT_PRETRAIN_RATE: number; DEFAULT_PRETRAIN_MOMENTUM: number; DEFAULT_TRAIN_BATCH_SMALL: number; }, applyCompassWarmStart: (network: any) => void, centerOutputBiases: (network: any) => void) => void`

### warmStartPopulationIfNeeded

`(neat: import("C:/NeatapticTS/src/neat").default, trainingSet: any[], state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, pretrainPopulation: (neat: any, trainingSet: any[]) => void) => void`

## evolutionEngine/populationDynamics.ts

### populationDynamics

populationDynamics.ts

Population-level dynamics for NEAT evolution: generation state management,
population expansion/compaction, mutation application, sorting, species tracking,
and anti-collapse recovery.

Responsibilities:
- Plateau detection and simplify-phase orchestration
- Population expansion with parent sampling and mutation
- Genome sorting by fitness (iterative quicksort with pooled scratch)
- Species history tracking and collapse detection
- Connection compaction and anti-collapse recovery

All functions accept `EngineState` to access shared scratch buffers and RNG.
Follows ES2023 idioms: `toSorted`, `.at(-1)`, numeric separators, etc.

### antiCollapseRecovery

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any, completedGenerations: number, safeWrite: (msg: string) => void, sampleSegmentIntoScratchFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, array: any[], startIdx: number, count: number) => number) => void`

### applyMutationsToClone

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, clone: any, neat: any, mutateCount: number) => void`

### compactGenomeConnections

`(genome: any) => number`

### compactPopulation

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any) => number`

### createChildFromParent

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any, parent: any) => any`

### determineMutateCount

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState) => number`

### ensureOutputIdentity

`(neat: any) => void`

### expandPopulation

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any, targetAdd: number, safeWrite: (msg: string) => void, completedGenerations: number) => void`

### getSortedIndicesByScore

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, population: any[]) => number[]`

### handleSimplifyState

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any, plateauCounter: number, plateauGenerations: number, simplifyDuration: number, simplifyMode: boolean, simplifyRemaining: number, simplifyStrategy: string, simplifyPruneFraction: number) => { simplifyMode: boolean; simplifyRemaining: number; plateauCounter: number; }`

### handleSpeciesHistory

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any, speciesHistory: number[]) => boolean`

### IndexBuffer

Typed or array-based index buffer for sorting

### maybeExpandPopulation

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any, dynamicPopEnabled: boolean, completedGenerations: number, dynamicPopMax: number, plateauGenerations: number, plateauCounter: number, dynamicPopExpandInterval: number, dynamicPopExpandFactor: number, dynamicPopPlateauSlack: number, safeWrite: (msg: string) => void) => void`

### maybeStartSimplify

`(plateauCounter: number, plateauGenerations: number, simplifyDuration: number) => number`

### MutationOperation

Mutation operation from NEAT driver

### prepareExpansion

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any) => { populationRef: any[]; sortedIdx: number[]; parentPoolSize: number; }`

### pruneSaturatedHiddenOutputs

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, genome: unknown, getNodeIndicesByType: (nodes: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode[], nodeType: string) => number, collectHiddenToOutputConns: (hiddenNode: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode, nodes: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkNode[], outputCount: number) => import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").NetworkConnection[]) => void`

### registerClone

`(neat: any, clone: any, parentId: any) => void`

### reinitializeGenomeOutputsAndWeights

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, genome: unknown) => { connReset: number; biasReset: number; }`

### runSimplifyCycle

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState").EngineState, neat: any, simplifyRemaining: number, simplifyStrategy: string, simplifyPruneFraction: number) => number`

### RuntimeGenome

Network genome with dynamic runtime properties

### updatePlateauState

`(fitness: number, lastBestFitnessForPlateau: number, plateauCounter: number, plateauImprovementThreshold: number) => { plateauCounter: number; lastBestFitnessForPlateau: number; }`
