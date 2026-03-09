# evolutionEngine

## evolutionEngine/engineState.types.ts

### engineState.types

Shared contracts for the ASCII maze evolution engine state boundary.

This file owns the exported type surface consumed by the engine-state facade,
telemetry helpers, sampling utilities, and RNG/timing adapters.

### EngineProfilingState

Aggregated profiling configuration and accumulators shared across the evolution run.

### EngineScratchState

Shared contracts for the ASCII maze evolution engine state boundary.

This file owns the exported type surface consumed by the engine-state facade,
telemetry helpers, sampling utilities, and RNG/timing adapters.

### EngineState

Shared engine state instance combining pooled scratch buffers with toggle flags.

### EngineToggleState

Runtime switches that adjust telemetry verbosity and optional training phases.

### RngCacheHandles

Handles returned after ensuring the RNG cache is ready for consumption.

### RngCacheParameters

Parameters controlling the RNG cache refill process.

### TelemetryScratchHandles

Collection of scratch buffers handed back after initialisation for convenience.

### TelemetryScratchRequest

Configuration describing which telemetry scratch buffers require capacity guarantees.

### VisitedHashScratchHandles

Handles exposed after ensuring the visited-coordinate hash table capacity.

## evolutionEngine/evolutionEngine.types.ts

### DistanceMap

Distance map for maze navigation.

### EncodedMaze

Encoded maze representation with cell values.

### EncodedMazeData

Encoded maze for simulation.

### EvolutionGenomeLike

Loose genome shape shared by engine telemetry and population-dynamics helpers.

### EvolutionHelpers

Helper functions object passed to evolution loop orchestration.

### EvolutionHostAdapter

Narrow host adapter used by the engine for pause polling and stop notifications.

### EvolutionHostStopEvent

Host-facing stop event emitted by the engine when a run finishes for a concrete reason.

### EvolutionLoopHelpers

Helper functions for evolution.

### EvolutionOptions

Options object passed to evolution functions.

### EvolutionStopReason

Canonical stop reasons reported by the engine to host adapters.

### FileSystem

Node.js fs module type for file operations.

### IAgentSimulationConfig

Agent simulation configuration.

### IEvolutionAlgorithmConfig

Configuration options for the evolutionary algorithm used in the ASCII Maze demos.

### IMazeConfig

Maze configuration used by the ASCII Maze evolution helpers.

### IReportingConfig

Reporting configuration used to control logging, dashboard updates and UI pacing.

### IRunMazeEvolutionOptions

Main options for running a single maze-evolution experiment.

### LogitsRingState

Ring state for logits tracking.

### LoopHelpers

Loop helpers returned by prepareLoopHelpers.

### MazeDistanceMap

Distance map for pathfinding.

### MazeEvolutionCurriculumPhaseOutcome

Shared curriculum-facing summary derived from one completed evolution phase.

### MazeEvolutionRunResult

Stable result returned by `EvolutionEngine.runMazeEvolution()`.

### MazePosition

Position in maze.

### MutationOperationLike

Mutation-operation surface read from the NEAT driver at runtime.

### NeatInstance

Type for Neat class instance from the neataptic library.

### NetworkConnection

Network connection representation used by engine-side runtime adaptation helpers.

### NetworkInstance

Type for Network class instance from the neataptic library.

### NetworkNode

Network node representation used by engine-side runtime adaptation helpers.

### PathModule

Node.js path module type.

### Position

2D position in maze coordinates.

### ProfilingAccumulators

Profiling accumulator structure.

### ScratchBundle

Scratch bundle containing reusable buffers.

### SimulationResult

Simulation result returned by generation evaluation helpers.

### SnapshotEntry

Snapshot entry for persistence.

### SpeciesHistoryHost

Static host used to read optional species-history state from the engine facade.

### TelemetryNeatLike

NEAT runtime shape needed by telemetry helpers that inspect the population.

### TrackedNetworkInstance

Network instance annotated with telemetry fields during a generation.

### TrainingConstants

Training constants used by Lamarckian warm-start and refinement helpers.

## evolutionEngine/sampling.ts

### sampling

Sampling and history helpers extracted from the ASCII maze evolution façade.

Responsibilities:
1. Provide allocation-light array sampling utilities that reuse the shared `EngineState` scratch pools.
2. Expose history helpers that mirror the façade behaviour while keeping pooled buffers centralised.
3. Centralise RNG parameter resolution for sampling paths to keep behaviour deterministic under shared state.

### getTail

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, source: T[] | undefined, count: number) => T[]`

### pushHistory

`(buffer: T[] | undefined, value: T, maxLength: number) => T[]`

### sampleArray

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, source: T[], sampleCount: number) => T[]`

### sampleIntoScratch

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, source: T[], sampleCount: number) => number`

### sampleSegmentIntoScratch

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, source: T[], segmentStart: number, sampleCount: number) => number`

## evolutionEngine/engineState.ts

### engineState

### createEngineState

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState`

### createProfilingState

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineProfilingState`

### createScratchState

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineScratchState`

### createToggleState

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineToggleState`

### DEFAULT_RNG_CACHE_BATCH_SIZE

### DEFAULT_VISITED_HASH_LOAD_FACTOR

### EngineProfilingState

Aggregated profiling configuration and accumulators shared across the evolution run.

### EngineScratchState

Shared contracts for the ASCII maze evolution engine state boundary.

This file owns the exported type surface consumed by the engine-state facade,
telemetry helpers, sampling utilities, and RNG/timing adapters.

### EngineState

Shared engine state instance combining pooled scratch buffers with toggle flags.

### EngineToggleState

Runtime switches that adjust telemetry verbosity and optional training phases.

### ensureRngCacheBatch

`(parameters: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").RngCacheParameters, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").RngCacheHandles`

### ensureVisitedHashCapacity

`(targetEntryCount: number, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").VisitedHashScratchHandles`

### initialiseTelemetryScratch

`(request: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").TelemetryScratchRequest, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").TelemetryScratchHandles`

### reseedRngState

`(seed: number, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => number`

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

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, category: string, deltaMs: number) => void`

### clearDeterministicMode

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => void`

### drawFastRandom

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, parameters: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").RngCacheParameters) => number`

### getProfilingAccumulators

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => Record<string, number>`

### isDeterministicModeEnabled

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => boolean`

### isProfilingDetailsEnabled

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => boolean`

### profilingStartTimestamp

`() => number`

### readHighResolutionTime

`() => number`

### resolveRngParameters

`() => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").RngCacheParameters`

### setDeterministicMode

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, seed: number | undefined) => void`

## evolutionEngine/scratchPools.ts

### scratchPools

Scratch pool management helpers extracted from the maze evolution façade.

These utilities centralise the logic that grows and shrinks pooled buffers attached to the
shared {@link EngineState}. They keep the façade lean by encapsulating heuristics for logits
ring sizing, telemetry scratch sizing, and connection flag pooling.

### allocateLogitsRing

`(capacity: number, actionDimension: number) => Float32Array<ArrayBufferLike>[]`

### ensureConnFlagsCapacity

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, minimumCapacity: number) => Int8Array<ArrayBufferLike> | null`

### ensureLogitsRingCapacity

`(capacityRequest: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/scratchPools").LogitsRingCapacityOptions) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/scratchPools").LogitsRingCapacityResult`

### ensureScratchCapacity

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, request: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/scratchPools").ScratchCapacityRequest) => void`

### initialiseSharedLogitsRing

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, config: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/scratchPools").SharedLogitsConfig) => boolean`

### LogitsRingCapacityOptions

Shape describing the parameters used when ensuring the logits ring capacity.

### LogitsRingCapacityResult

Result returned after resizing the logits ring.

### maybeShrinkScratch

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, populationSize: number) => void`

### ScratchCapacityRequest

Parameters describing the scratch sizing requirements for ensureScratchCapacity.

### SharedLogitsConfig

Parameters passed when attempting to initialise the shared logits ring buffers.

## evolutionEngine/setupHelpers.ts

### DashboardManagerLike

Dashboard manager shape for logging (optional log function).

### FilesystemModule

Setup helpers for the ASCII maze evolution engine.

Responsibilities:
- Create cooperative frame-yielding helpers for async evolution loops.
- Initialize Node.js persistence helpers when available.
- Build resilient logging writers with dashboard and console fallbacks.

### initPersistence

`(persistDir: string | undefined) => { fs: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/setupHelpers").FilesystemModule | null; path: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/setupHelpers").PathModule | null; }`

### isPauseRequested

`(hostAdapter: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").EvolutionHostAdapter | undefined) => boolean`

Read host-controlled pause state without letting host errors break the engine.

Parameters:
- `hostAdapter` - - Optional host adapter implementing pause polling.

Returns: True when the host asks the engine to remain paused.

### makeFlushToFrame

`(hostAdapter: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").EvolutionHostAdapter | undefined) => () => Promise<void>`

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

`(options: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").EvolutionOptions, bestResult: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult | undefined) => string | undefined`

### checkStopConditions

`(bestResult: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult | undefined, bestNetwork: import("C:/NeatapticTS/src/architecture/network").default | null, maze: string[], completedGenerations: number, neat: import("C:/NeatapticTS/src/neat").default, dashboardManager: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IDashboardManager | undefined, flushToFrame: () => Promise<void>, hostAdapter: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").EvolutionHostAdapter | undefined, minProgressToPass: number, autoPauseOnSolve: boolean, stopOnlyOnSolve: boolean, stagnantGenerations: number, maxStagnantGenerations: number, maxGenerations: number) => Promise<string | undefined>`

### emitProfileSummary

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, safeWrite: (msg: string) => void, completedGenerations: number, totalEvolveMs: number, totalLamarckMs: number, totalSimMs: number, isProfilingDetailsEnabledFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => boolean, getProfilingAccumulatorsFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").ProfilingAccumulators) => void`

### EvolutionLoopResult

Evolution loop result

### GenerationOutcome

Generation outcome with profiling timings

### MutableMazeResult

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

### persistSnapshotIfNeeded

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, fs: { writeFileSync?: ((path: string, data: string) => void) | undefined; } | null, pathModule: { join?: ((...paths: string[]) => string) | undefined; } | null, persistDir: string | undefined, persistTopK: number, completedGenerations: number, persistEvery: number, neat: import("C:/NeatapticTS/src/neat").default, bestFitness: number, simplifyMode: boolean, plateauCounter: number, scratchSnapshotObj: Record<string, unknown>, scratchSnapshotTop: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").SnapshotEntry[], collectTelemetryTailFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: import("C:/NeatapticTS/src/neat").default, count: number) => unknown, getSortedIndicesByScoreFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, population: import("C:/NeatapticTS/src/architecture/network").default[]) => number[], isProfilingDetailsEnabledFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => boolean, profilingStartTimestampFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => number, accumulateProfilingDurationFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, label: string, duration: number) => void) => void`

### prepareLoopHelpers

`(opts: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").EvolutionOptions, scratchBundle: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").ScratchBundle) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").LoopHelpers`

### runEvolutionLoop

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: import("C:/NeatapticTS/src/neat").default, opts: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").EvolutionOptions, lamarckianTrainingSet: { input: number[]; output: number[]; }[], encodedMaze: number[][], startPosition: readonly [number, number], exitPosition: readonly [number, number], distanceMap: number[][], helpers: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").LoopHelpers, doProfile: boolean, scratchLogitsRing: Float32Array<ArrayBufferLike>[], logitsRingCap: number, logitsRingCapMax: number, actionDim: number, logitsRingShared: boolean, scratchLogitsShared: Float32Array<ArrayBufferLike> | undefined, scratchLogitsSharedW: Int32Array<ArrayBufferLike> | undefined, scratchLogitsRingW: number, emptyVec: import("C:/NeatapticTS/src/architecture/network").default[], scratchNodeIdx: Int32Array<ArrayBufferLike>, scratchSnapshotObj: Record<string, unknown>, scratchSnapshotTop: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").SnapshotEntry[], getNodeIndicesByType: (nodes: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode[], type: string) => number, collectHiddenToOutputConns: (hiddenNode: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode, nodesRef: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode[], outputCount: number) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkConnection[], constants: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").TrainingConstants & { DEFAULT_TRAIN_BATCH_LARGE: number; FITTEST_TRAIN_ITERATIONS: number; TELEMETRY_MINIMAL: boolean; SATURATION_PRUNE_THRESHOLD: number; RECENT_WINDOW: number; REDUCED_TELEMETRY: boolean; DISABLE_BALDWIN: boolean; }, speciesHistoryRef: number[]) => Promise<EvolutionLoopResult>`

### runGeneration

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: import("C:/NeatapticTS/src/neat").default, doProfile: boolean, lamarckianIterations: number, lamarckianTrainingSet: { input: number[]; output: number[]; }[], lamarckianSampleSize: number | undefined, safeWrite: (msg: string) => void, completedGenerations: number, dynamicPopEnabled: boolean, dynamicPopMax: number, plateauGenerations: number, plateauCounter: number, dynamicPopExpandInterval: number, dynamicPopExpandFactor: number, dynamicPopPlateauSlack: number, speciesHistoryRef: number[], emptyVec: import("C:/NeatapticTS/src/architecture/network").default[], scratchNodeIdx: Int32Array<ArrayBufferLike>, getNodeIndicesByType: (nodes: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode[], type: string) => number, constants: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").TrainingConstants) => Promise<GenerationOutcome>`

### SimResultWithOutputs

Simulation result with step outputs

### simulateAndPostprocess

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, fittest: import("C:/NeatapticTS/src/architecture/network").default, encodedMaze: number[][], startPosition: readonly [number, number], exitPosition: readonly [number, number], distanceMap: number[][], maxSteps: number | undefined, doProfile: boolean, safeWrite: (msg: string) => void, logEvery: number, completedGenerations: number, neat: import("C:/NeatapticTS/src/neat").default, scratchLogitsRing: Float32Array<ArrayBufferLike>[], logitsRingCap: number, logitsRingCapMax: number, actionDim: number, logitsRingShared: boolean, scratchLogitsShared: Float32Array<ArrayBufferLike> | undefined, scratchLogitsSharedW: Int32Array<ArrayBufferLike> | undefined, scratchLogitsRingW: number, telemetryMinimal: boolean, saturationPruneThreshold: number, recentWindow: number, reducedTelemetry: boolean, getNodeIndicesByType: (nodes: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode[], type: string) => number, collectHiddenToOutputConns: (hiddenNode: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode, nodesRef: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode[], outputCount: number) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkConnection[]) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").SimulationResult`

### SimulationOutcome

Simulation result with profiling and ring state

### updateDashboardAndMaybeFlush

`(maze: string[], result: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult | undefined, network: import("C:/NeatapticTS/src/architecture/network").default | null, completedGenerations: number, neat: import("C:/NeatapticTS/src/neat").default, dashboardManager: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IDashboardManager | undefined, flushToFrame: (() => Promise<void>) | undefined) => Promise<void>`

### updateDashboardPeriodic

`(maze: string[], bestResult: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult | undefined, bestNetwork: import("C:/NeatapticTS/src/architecture/network").default | null, completedGenerations: number, neat: import("C:/NeatapticTS/src/neat").default, dashboardManager: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IDashboardManager | undefined, flushToFrame: (() => Promise<void>) | undefined) => Promise<void>`

## evolutionEngine/curriculumPhase.ts

### hasMazeEvolutionReachedCurriculumThreshold

`(progress: unknown, minProgressToPass: number) => boolean`

### refineMazeEvolutionCarryOverNetwork

`(bestNetwork: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork | undefined, previousBestNetwork: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork | undefined) => import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork | undefined`

### resolveMazeEvolutionPhaseOutcome

`(evolutionResult: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").MazeEvolutionRunResult, previousBestNetwork: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork | undefined, minProgressToPass: number) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").MazeEvolutionCurriculumPhaseOutcome`

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

`(opts: any, inputSize: number, outputSize: number, fitnessContext: import("C:/NeatapticTS/test/examples/asciiMaze/fitness.types").IFitnessEvaluationContext, scratchPopClone: any[], scratchSample: any[]) => any`

### normalizeRunOptions

`(options: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").IRunMazeEvolutionOptions, setDeterministic: (seed: number) => void, setReducedTelemetry: (enabled: boolean) => void, setMinimalTelemetry: (enabled: boolean) => void, setDisableBaldwin: (disabled: boolean) => void) => any`

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

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: unknown, tailLength: number) => unknown`

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

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: import("C:/NeatapticTS/src/neat").default, fittest: import("C:/NeatapticTS/src/architecture/network").default | undefined, genResult: GenerationResult | undefined, generationIndex: number, writeLog: (msg: string) => void, actionDimension: number, recentWindow: number, reducedTelemetry: boolean, telemetryMinimal: boolean, onCollapseRecovery: () => void, isProfilingDetailsEnabledFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => boolean, profilingStartTimestampFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => number, accumulateProfilingDurationFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, label: string, duration: number) => void) => void`

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

`(engineState: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, network: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork) => void`

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

`(network: import("C:/NeatapticTS/src/architecture/network").default, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, constants: { DEFAULT_STD_SMALL: number; DEFAULT_STD_ADJUST_MULT: number; }, scratchNodeIdx: Int32Array<ArrayBufferLike>, getNodeIndicesByType: (nodes: any[], nodeType: string) => number) => void`

### applyLamarckianTraining

`(neat: import("C:/NeatapticTS/src/neat").default, trainingSet: any[], iterations: number, sampleSize: number | undefined, safeWrite: (msg: string) => void, profileEnabled: boolean, completedGenerations: number, state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, constants: { DEFAULT_TRAIN_ERROR: number; DEFAULT_TRAIN_RATE: number; DEFAULT_TRAIN_MOMENTUM: number; DEFAULT_TRAIN_BATCH_SMALL: number; }, adjustOutputBiases: (network: any) => void) => number`

### buildLamarckianTrainingSet

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, constants: { TRAIN_OUT_PROB_HIGH: number; TRAIN_OUT_PROB_LOW: number; PROGRESS_MEDIUM: number; PROGRESS_STRONG: number; PROGRESS_JUNCTION: number; PROGRESS_FOURWAY: number; PROGRESS_REGRESS: number; PROGRESS_MIN_SIGNAL: number; PROGRESS_MILD_REGRESS: number; DEFAULT_JITTER_PROB: number; AUGMENT_JITTER_BASE: number; AUGMENT_JITTER_RANGE: number; AUGMENT_PROGRESS_JITTER_PROB: number; AUGMENT_PROGRESS_DELTA_RANGE: number; AUGMENT_PROGRESS_DELTA_HALF: number; RNG_PARAMETERS: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").RngCacheParameters; }) => { input: number[]; output: number[]; }[]`

### pretrainPopulationWarmStart

`(neat: import("C:/NeatapticTS/src/neat").default, lamarckianTrainingSet: any[], constants: { PRETRAIN_MAX_ITER: number; PRETRAIN_BASE_ITER: number; DEFAULT_TRAIN_ERROR: number; DEFAULT_PRETRAIN_RATE: number; DEFAULT_PRETRAIN_MOMENTUM: number; DEFAULT_TRAIN_BATCH_SMALL: number; }, applyCompassWarmStart: (network: any) => void, centerOutputBiases: (network: any) => void) => void`

### warmStartPopulationIfNeeded

`(neat: import("C:/NeatapticTS/src/neat").default, trainingSet: any[], state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, pretrainPopulation: (neat: any, trainingSet: any[]) => void) => void`

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

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any, completedGenerations: number, safeWrite: (msg: string) => void, sampleSegmentIntoScratchFn: (state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, array: any[], startIdx: number, count: number) => number) => void`

### applyMutationsToClone

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, clone: any, neat: any, mutateCount: number) => void`

### compactGenomeConnections

`(genome: any) => number`

### compactPopulation

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any) => number`

### createChildFromParent

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any, parent: any) => any`

### determineMutateCount

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState) => number`

### ensureOutputIdentity

`(neat: any) => void`

### expandPopulation

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any, targetAdd: number, safeWrite: (msg: string) => void, completedGenerations: number) => void`

### getSortedIndicesByScore

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, population: any[]) => number[]`

### handleSimplifyState

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any, plateauCounter: number, plateauGenerations: number, simplifyDuration: number, simplifyMode: boolean, simplifyRemaining: number, simplifyStrategy: string, simplifyPruneFraction: number) => { simplifyMode: boolean; simplifyRemaining: number; plateauCounter: number; }`

### handleSpeciesHistory

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any, speciesHistory: number[]) => boolean`

### IndexBuffer

Typed or array-based index buffer for sorting

### maybeExpandPopulation

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any, dynamicPopEnabled: boolean, completedGenerations: number, dynamicPopMax: number, plateauGenerations: number, plateauCounter: number, dynamicPopExpandInterval: number, dynamicPopExpandFactor: number, dynamicPopPlateauSlack: number, safeWrite: (msg: string) => void) => void`

### maybeStartSimplify

`(plateauCounter: number, plateauGenerations: number, simplifyDuration: number) => number`

### prepareExpansion

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any) => { populationRef: any[]; sortedIdx: number[]; parentPoolSize: number; }`

### pruneSaturatedHiddenOutputs

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, genome: unknown, getNodeIndicesByType: (nodes: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode[], nodeType: string) => number, collectHiddenToOutputConns: (hiddenNode: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode, nodes: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkNode[], outputCount: number) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/evolutionEngine.types").NetworkConnection[]) => void`

### registerClone

`(neat: any, clone: any, parentId: any) => void`

### reinitializeGenomeOutputsAndWeights

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, genome: unknown) => { connReset: number; biasReset: number; }`

### runSimplifyCycle

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineState, neat: any, simplifyRemaining: number, simplifyStrategy: string, simplifyPruneFraction: number) => number`

### updatePlateauState

`(fitness: number, lastBestFitnessForPlateau: number, plateauCounter: number, plateauImprovementThreshold: number) => { plateauCounter: number; lastBestFitnessForPlateau: number; }`

## evolutionEngine/engineState.constants.ts

### engineState.constants

Constant values shared by the ASCII maze engine-state facade and its helper utilities.

### ACTION_OUTPUT_DIMENSION

### DEFAULT_CONNECTION_FLAG_CAPACITY

### DEFAULT_HISTORY_BUFFER_CAPACITY

### DEFAULT_LOGITS_RING_CAPACITY

### DEFAULT_NODE_INDEX_BUFFER_CAPACITY

### DEFAULT_QUICKSORT_STACK_CAPACITY

### DEFAULT_RNG_CACHE_BATCH_SIZE

### DEFAULT_SAMPLE_POOL_SIZE

### DEFAULT_SMALL_EXPLORE_TABLE_CAPACITY

### DEFAULT_SORTED_INDEX_CAPACITY

### DEFAULT_SPECIES_SCRATCH_CAPACITY

### DEFAULT_STRING_BUFFER_CAPACITY

### DEFAULT_VISITED_HASH_LOAD_FACTOR

### MAX_VISITED_HASH_LOAD_FACTOR

### MIN_VISITED_HASH_LOAD_FACTOR

### RNG_GOLDEN_RATIO_SEED

## evolutionEngine/engineState.utils.ts

### engineState.utils

Pure utility helpers shared by the ASCII maze engine-state facade.

### buildTelemetryHandles

`(scratch: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineScratchState) => import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").TelemetryScratchHandles`

### createLogitsRing

`() => Float32Array<ArrayBufferLike>[]`

### createProfilingAccumulators

`() => Record<string, number>`

### createSnapshotReusableObject

`() => { generation: number; bestFitness: number; simplifyMode: boolean; plateauCounter: number; timestamp: number; telemetryTail: undefined; top: undefined; }`

### ensureTelemetryFloatPools

`(scratch: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").EngineScratchState, hints: TelemetryCapacityHints) => void`

### ensureTelemetryStringBuffer

`(buffer: string[], required: number) => string[]`

### nextPowerOfTwo

`(candidate: number) => number`

### normaliseRngBatchSize

`(requestedBatchSize: number) => number`

### normaliseRngSeed

`(rawSeed: number) => number`

### normaliseTelemetryCapacityHints

`(request: import("C:/NeatapticTS/test/examples/asciiMaze/evolutionEngine/engineState.types").TelemetryScratchRequest) => TelemetryCapacityHints`

### normaliseVisitedHashEntries

`(requestedEntries: number) => number`

### normaliseVisitedHashLoad

`(requestedLoadFactor: number) => number`

### resolveProfilingEnabled

`() => boolean`
