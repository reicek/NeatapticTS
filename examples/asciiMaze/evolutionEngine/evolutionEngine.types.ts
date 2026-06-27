/**
 * Population-level runtime contracts for the ASCII Maze evolution engine.
 *
 * This folder is where one maze, one fitness story, and one NEAT controller
 * turn into a repeatable training program. The public `EvolutionEngine` facade
 * uses these contracts to normalize options, coordinate host callbacks, keep
 * deterministic state reproducible, and decide when a curriculum phase has
 * produced a winner worth carrying forward.
 *
 * The important distinction is scale. `mazeMovement/` explains one agent run.
 * `fitness.ts` explains how that run is scored. `dashboardManager/` explains
 * how progress is shown to a human. `evolutionEngine/` explains how many runs
 * across many generations become one population-level search loop with stop
 * reasons, telemetry, warm starts, and phase outcomes that the next maze can
 * reuse.
 *
 * This file is the right chapter opening because it names the public nouns of
 * that loop before the reader hits pooled scratch buffers or hot-path helpers.
 * It answers four questions quickly: what a caller can configure, what a host
 * may observe or interrupt, what result the engine returns, and which shared
 * runtime contexts keep the hot path allocation-light.
 *
 * A useful mental model is to treat the engine as a control tower rather than
 * the aircraft itself. The engine does not move the agent through one maze cell
 * at a time. It schedules phases, batches generations, preserves deterministic
 * state, and hands structured outcomes back to browser or terminal hosts.
 *
 * Read the chapter in three passes. Start here for the public contracts and the
 * meaning of a run result. Continue to `engineState.types.ts` when you want the
 * shared scratch and toggle state that keeps the loop cheap. Finish with
 * `evolutionLoop.ts`, `evolutionEngine.services.ts`, and `sampling.ts` when you
 * want the actual orchestration and telemetry mechanics.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Caller["Caller or host"]:::base --> Options["IRunMazeEvolutionOptions\nrun inputs"]:::base
 *   Options --> Engine["EvolutionEngine facade\npopulation-level control"]:::accent
 *   Engine --> Loop["generation loop\nevaluate mutate telemetry"]:::base
 *   Loop --> Result["MazeEvolutionRunResult\nbest network + exit reason"]:::base
 *   Result --> Phase["MazeEvolutionCurriculumPhaseOutcome\ncarry winner forward"]:::base
 *   Engine --> Host["EvolutionHostAdapter\npause and stop hooks"]:::base
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   EngineFolder["evolutionEngine/"]:::accent --> Contracts["evolutionEngine.types.ts\npublic run contracts"]:::base
 *   EngineFolder --> Scratch["engineState.types.ts\nshared scratch and toggles"]:::base
 *   EngineFolder --> Loop["evolutionLoop.ts\nmain generation orchestration"]:::base
 *   EngineFolder --> Services["evolutionEngine.services.ts\nand helpers"]:::base
 *   EngineFolder --> Sampling["sampling.ts\nand telemetry support"]:::base
 * ```
 *
 * For background reading on the staged difficulty idea behind the browser and
 * curriculum-style runs, see Wikipedia contributors,
 * [Curriculum learning](https://en.wikipedia.org/wiki/Curriculum_learning),
 * which captures the broader teaching idea of solving easier tasks before
 * harder ones.
 *
 * Example: describe the host adapter and reporting hooks the engine may call.
 *
 * ```ts
 * const hostAdapter: EvolutionHostAdapter = {
 *   isPauseRequested: () => window.asciiMazePaused === true,
 *   handleStop: ({ reason, completedGenerations }) => {
 *     console.log(reason, completedGenerations);
 *   },
 * };
 * ```
 *
 * Example: sketch one engine run configuration before execution begins.
 *
 * ```ts
 * const runOptions: IRunMazeEvolutionOptions = {
 *   mazeConfig: { maze },
 *   agentSimConfig: { maxSteps: 160 },
 *   evolutionAlgorithmConfig: { popSize: 120, deterministic: true },
 *   reportingConfig: { dashboardManager, logEvery: 5 },
 * };
 * ```
 */

import type Network from '../../../src/architecture/network';
import type { FitnessEvaluatorFn } from '../fitness.types';
import type {
  IDashboardManager,
  IMazeRunResult,
  INetwork,
} from '../interfaces';
import type { ExampleArchitectureProfileId } from '../../architectureProfiles';

/** Maze configuration used by the ASCII Maze evolution helpers. */
export interface IMazeConfig {
  /** The maze layout, each element is a row string. Rows should be equal length. */
  maze: string[];
}

/** Agent simulation configuration. */
export interface IAgentSimulationConfig {
  /** Maximum number of steps the agent can take in a single episode. */
  maxSteps: number;
}

/**
 * Adaptive mutation tuning forwarded to the underlying NEAT runtime.
 *
 * These knobs let host layers increase exploration pressure without needing to
 * know the internal controller implementation details. The ASCII Maze browser
 * curriculum uses this to make the first phase more willing to grow topology
 * when early generations stall.
 */
export interface EvolutionAdaptiveMutationConfig {
  enabled?: boolean;
  strategy?: string;
  adaptEvery?: number;
  sigma?: number;
  minRate?: number;
  maxRate?: number;
  initialRate?: number;
  adaptAmount?: boolean;
  minAmount?: number;
  maxAmount?: number;
  amountSigma?: number;
  [key: string]: unknown;
}

/** Browser-worker evaluation controls for parallel ASCII Maze genome scoring. */
export interface EvolutionWorkerEvaluationConfig {
  /** Attempt browser worker evaluation when true. */
  enabled?: boolean;
  /** Dedicated worker bundle URL used for maze-fitness evaluation. */
  workerUrl?: string;
  /** Maximum number of concurrently active evaluation workers. */
  workerCount?: number;
}

/** Configuration options for the evolutionary algorithm used in the ASCII Maze demos. */
export interface IEvolutionAlgorithmConfig {
  allowRecurrent?: boolean;
  architectureProfileId?: ExampleArchitectureProfileId;
  adaptiveMutation?: EvolutionAdaptiveMutationConfig;
  workerEvaluation?: EvolutionWorkerEvaluationConfig;
  popSize?: number;
  maxStagnantGenerations?: number;
  minProgressToPass?: number;
  maxGenerations?: number;
  stopOnlyOnSolve?: boolean;
  autoPauseOnSolve?: boolean;
  randomSeed?: number;
  initialPopulation?: INetwork[];
  initialBestNetwork?: INetwork | Network;
  lamarckianIterations?: number;
  lamarckianSampleSize?: number;
  plateauGenerations?: number;
  plateauImprovementThreshold?: number;
  simplifyDuration?: number;
  simplifyPruneFraction?: number;
  simplifyStrategy?: 'weakWeight' | 'weakRecurrentPreferred';
  persistEvery?: number;
  persistDir?: string;
  persistTopK?: number;
  dynamicPopEnabled?: boolean;
  dynamicPopMax?: number;
  dynamicPopExpandInterval?: number;
  dynamicPopExpandFactor?: number;
  dynamicPopPlateauSlack?: number;
  deterministic?: boolean;
  memoryCompactionInterval?: number;
  telemetryReduceStats?: boolean;
  telemetryMinimal?: boolean;
  disableBaldwinianRefinement?: boolean;
}

/** Canonical stop reasons reported by the engine to host adapters. */
export type EvolutionStopReason =
  'solved' | 'stagnation' | 'maxGenerations' | 'cancelled' | 'aborted';

/**
 * Host-facing stop event emitted by the engine when a run finishes for a concrete reason.
 *
 * @remarks
 * The engine uses this narrow payload so browser or terminal hosts can react to
 * solve, stop, and pause-adjacent lifecycle events without the engine depending
 * on browser globals or DOM APIs.
 */
export interface EvolutionHostStopEvent {
  /** Canonical reason that caused the run to stop. */
  reason: EvolutionStopReason;
  /** Maze layout associated with the stopping run. */
  maze: string[];
  /** Number of generations completed before the stop reason fired. */
  completedGenerations: number;
  /** Latest best result available at stop time, when one exists. */
  result?: IMazeRunResult;
  /** Convenience progress mirror from the latest result. */
  progress?: number;
  /** Whether the host should cooperatively enter a paused state. */
  requestHostPause?: boolean;
}

/**
 * Narrow host adapter used by the engine for pause polling and stop notifications.
 *
 * @remarks
 * Browser-entry and other host boundaries should implement this contract when
 * they need host-specific pause control or solve/stop side effects.
 *
 * @example
 * ```ts
 * const hostAdapter: EvolutionHostAdapter = {
 *   isPauseRequested: () => window.asciiMazePaused === true,
 *   handleStop: ({ reason }) => console.log('maze run stopped because', reason),
 * };
 * ```
 */
export interface EvolutionHostAdapter {
  /** Return true while the host wants cooperative frame flushing to stay paused. */
  isPauseRequested?: () => boolean;
  /** React to a solved, aborted, or otherwise stopped run. */
  handleStop?: (event: EvolutionHostStopEvent) => void | Promise<void>;
}

/** Reporting configuration used to control logging, dashboard updates and UI pacing. */
export interface IReportingConfig {
  /** How frequently, in generations, to emit logs or telemetry updates. */
  logEvery?: number;
  /** Dashboard manager instance responsible for receiving per-generation updates. */
  dashboardManager: IDashboardManager;
  /** Optional human-readable label identifying this run. */
  label?: string;
  /** When true, yield to the host after each generation. */
  paceEveryGeneration?: boolean;
  /** Optional host adapter that owns pause polling and stop side effects. */
  hostAdapter?: EvolutionHostAdapter;
}

/** Main options for running a single maze-evolution experiment. */
export interface IRunMazeEvolutionOptions {
  mazeConfig: IMazeConfig;
  agentSimConfig: IAgentSimulationConfig;
  evolutionAlgorithmConfig: IEvolutionAlgorithmConfig;
  reportingConfig: IReportingConfig;
  fitnessEvaluator?: FitnessEvaluatorFn;
  cancellation?: { isCancelled: () => boolean };
  signal?: AbortSignal;
}

/**
 * Stable result returned by `EvolutionEngine.runMazeEvolution()`.
 *
 * @remarks
 * Browser curriculum code, terminal demos, and tooling should depend on this
 * shared engine-owned contract instead of recreating local runtime result
 * adapters just to read winner carry-over or solve progress.
 */
export interface MazeEvolutionRunResult {
  /** Highest-scoring evolved network available at the end of the run. */
  bestNetwork: NetworkInstance | null;
  /** Best simulation result captured during the run, when one exists. */
  bestResult: IMazeRunResult | undefined;
  /** Final NEAT instance after all completed generations. */
  neat: NeatInstance;
  /** Canonical or compatibility exit reason describing why the run ended. */
  exitReason: string;
  /** Shared architecture profile id that seeded the run population, when one was used. */
  architectureProfileId?: ExampleArchitectureProfileId;
}

/**
 * Shared curriculum-facing summary derived from one completed evolution phase.
 *
 * @remarks
 * Browser curriculum code and curriculum-style tests should depend on this
 * engine-owned contract so host boundaries do not recreate local refinement or
 * result-interpretation shims when carrying winners across phases.
 */
export interface MazeEvolutionCurriculumPhaseOutcome {
  /** Original engine result for callers that still need the underlying run details. */
  result: MazeEvolutionRunResult;
  /** Convenience mirror of the best-run progress percentage for logging or gating. */
  progress: number | undefined;
  /** Whether the phase met the caller's curriculum advancement threshold. */
  solved: boolean;
  /** Refined winner to seed into the next curriculum phase, when one is available. */
  nextBestNetwork: INetwork | undefined;
}

/** Type for Neat class instance from the neataptic library. */
export type NeatInstance = import('../../../src/neat').default;

/** Type for Network class instance from the neataptic library. */
export type NetworkInstance =
  import('../../../src/architecture/network').default;

/**
 * Network instance annotated with telemetry fields during a generation.
 *
 * @remarks
 * This keeps telemetry post-processing and dashboard reporting aligned on one
 * engine-owned contract instead of file-local runtime casts.
 */
export interface TrackedNetworkInstance extends NetworkInstance {
  _lastStepOutputs?: Float32Array[];
  _saturationFraction?: number;
  _actionEntropy?: number;
}

/**
 * Loose genome shape shared by engine telemetry and population-dynamics helpers.
 *
 * @remarks
 * The concrete runtime objects are still `Network` instances, but the engine
 * occasionally needs to acknowledge a small amount of evolution-time metadata
 * that the plain public type does not emphasize: score, species membership,
 * clone hooks, lineage IDs, and telemetry scratch fields.
 *
 * This interface deliberately stays loose because it exists to document and
 * centralize that adaptation layer, not to replace the underlying runtime class.
 */
export interface EvolutionGenomeLike {
  nodes?: NetworkNode[];
  connections?: NetworkConnection[];
  score?: number;
  species?: number | null;
  clone?: () => EvolutionGenomeLike;
  mutate?: (method: unknown) => void;
  _lastStepOutputs?: unknown[];
  _id?: number;
  _parentId?: number;
  [key: string]: unknown;
}

/**
 * NEAT runtime shape needed by telemetry helpers that inspect the population.
 *
 * The telemetry helpers intentionally ask for very little here: access to the
 * population plus optional telemetry export. That keeps them portable across the
 * engine's internal helpers without binding them to the full driver surface.
 */
export interface TelemetryNeatLike {
  population?: EvolutionGenomeLike[];
  getTelemetry?: () => unknown;
  [key: string]: unknown;
}

/**
 * Mutation-operation surface read from the NEAT driver at runtime.
 *
 * The engine treats mutation operations as opaque descriptors because the
 * concrete driver owns how those operations are interpreted. The helpers only
 * need enough structure to cache, count, and hand them back into `mutate(...)`.
 */
export interface MutationOperationLike {
  length?: number;
  [key: string]: unknown;
}

/** Static host used to read optional species-history state from the engine facade. */
export interface SpeciesHistoryHost {
  _speciesHistory?: unknown[];
  [key: string]: unknown;
}

/** Encoded maze representation with cell values. */
export interface EncodedMaze {
  width: number;
  height: number;
  cells: number[];
  maze: string[];
}

/** 2D position in maze coordinates. */
export interface Position {
  x: number;
  y: number;
}

/** Distance map for maze navigation. */
export interface DistanceMap {
  width: number;
  height: number;
  distances: number[];
}

/** Options object passed to evolution functions. */
export interface EvolutionOptions {
  cancellation?: {
    isCancelled?: () => boolean;
    isCancellationRequested?: () => boolean;
    reason?: string;
  };
  signal?: AbortSignal;
  maxGenerations?: number;
  maxStagnantGenerations?: number;
  minProgressToPass?: number;
  logEvery?: number;
  enableProfiling?: boolean;
  persistDir?: string;
  persistTopK?: number;
  persistEvery?: number;
  flushToFrame?: () => Promise<void>;
  reportingConfig?: {
    dashboardManager?: IDashboardManager;
    logEvery?: number;
    paceEveryGeneration?: boolean;
    hostAdapter?: EvolutionHostAdapter;
    [key: string]: unknown;
  };
  mazeConfig: {
    maze: string[];
    [key: string]: unknown;
  };
  agentSimConfig?: {
    maxSteps: number;
    [key: string]: unknown;
  };
  lamarckianIterations?: number;
  lamarckianSampleSize?: number;
  dynamicPopEnabled?: boolean;
  dynamicPopMax?: number;
  plateauGenerations?: number;
  dynamicPopExpandInterval?: number;
  dynamicPopExpandFactor?: number;
  dynamicPopPlateauSlack?: number;
  plateauImprovementThreshold?: number;
  simplifyDuration?: number;
  simplifyStrategy?: 'weakWeight' | 'weakRecurrentPreferred';
  simplifyPruneFraction?: number;
  memoryCompactionInterval?: number;
  autoPauseOnSolve?: boolean;
  stopOnlyOnSolve?: boolean;
  initialBestNetwork?: unknown;
  [key: string]: unknown;
}

/** Helper functions object passed to evolution loop orchestration. */
export interface EvolutionHelpers {
  write: (msg: string) => void;
  isProfilingDetailsEnabled: (state: unknown) => boolean;
  getProfilingAccumulators: (state: unknown) => ProfilingAccumulators;
}

/** Profiling accumulator structure. */
export interface ProfilingAccumulators {
  totalEvolveMs?: number;
  totalLamarckMs?: number;
  totalSimMs?: number;
  mutate?: number;
  crossover?: number;
  select?: number;
  snapshot?: number;
  telemetry?: number;
  simplify?: number;
  prune?: number;
  [key: string]: number | undefined;
}

/** Node.js fs module type for file operations. */
export interface FileSystem {
  writeFileSync: (path: string, data: string) => void;
  readFileSync: (path: string, encoding: string) => string;
  existsSync: (path: string) => boolean;
  mkdirSync: (path: string, options?: { recursive?: boolean }) => void;
}

/** Node.js path module type. */
export interface PathModule {
  join: (...paths: string[]) => string;
  resolve: (...paths: string[]) => string;
  dirname: (path: string) => string;
}

/** Loop helpers returned by prepareLoopHelpers. */
export interface LoopHelpers {
  flushToFrame: () => Promise<void>;
  fs: unknown;
  path: unknown;
  safeWrite: (msg: string) => void;
}

/** Scratch bundle containing reusable buffers. */
export interface ScratchBundle {
  samplePool?: unknown[];
  profilingScratch?: Float64Array;
  exps?: Float64Array;
  [key: string]: unknown;
}

/** Snapshot entry for persistence. */
export interface SnapshotEntry {
  idx?: number;
  json?: string;
  score?: number;
  nodes?: number;
  connections?: number;
  [key: string]: unknown;
}

/** Training constants used by Lamarckian warm-start and refinement helpers. */
export interface TrainingConstants {
  DEFAULT_TRAIN_ERROR: number;
  DEFAULT_TRAIN_RATE: number;
  DEFAULT_TRAIN_MOMENTUM: number;
  DEFAULT_TRAIN_BATCH_SMALL: number;
  DEFAULT_STD_SMALL: number;
  DEFAULT_STD_ADJUST_MULT: number;
}

/** Helper functions for evolution. */
export interface EvolutionLoopHelpers {
  getNodeIndicesByType: (nodes: NetworkNode[], type: string) => number;
  collectHiddenToOutputConns: (
    hiddenNode: NetworkNode,
    nodes: NetworkNode[],
    outputCount: number,
  ) => NetworkConnection[];
}

/**
 * Shared runtime buffers and limits consumed by the evolution loop hot path.
 *
 * This context keeps the loop and simulation helpers from passing a long list
 * of pooled ring buffers, scratch arrays, and capacity limits positionally.
 */
export interface EvolutionLoopRuntimeContext {
  scratchLogitsRing: Float32Array[];
  logitsRingCapMax: number;
  actionDim: number;
  scratchLogitsShared?: Float32Array;
  scratchLogitsSharedW?: Int32Array;
}

/**
 * Shared telemetry thresholds consumed by the evolution loop simulation pass.
 *
 * The loop owns these switches conceptually, but grouping them as one context
 * keeps telemetry policy changes from widening hot-path function signatures.
 */
export interface EvolutionLoopTelemetryContext {
  telemetryMinimal: boolean;
  saturationPruneThreshold: number;
  recentWindow: number;
  reducedTelemetry: boolean;
}

/**
 * Shared scratch buffers and helper callbacks used across evolution-loop stages.
 *
 * This context groups the scratch arrays and analysis helpers that travel
 * together through generation, simulation, and snapshot paths.
 */
export interface EvolutionLoopSupportContext {
  emptyVec: NetworkInstance[];
  scratchNodeIdx: Int32Array;
  scratchSnapshotObj: Record<string, unknown>;
  scratchSnapshotTop: SnapshotEntry[];
  speciesHistoryRef: number[];
  loopHelpers: EvolutionLoopHelpers;
}

/** Network node representation used by engine-side runtime adaptation helpers. */
export interface NetworkNode {
  type?: string;
  bias?: number;
  squash?:
    string | { name?: string } | ((x: number, derivate?: boolean) => number);
  connections?: {
    in?: NetworkConnection[];
    out?: NetworkConnection[];
  };
  [key: string]: unknown;
}

/** Network connection representation used by engine-side runtime adaptation helpers. */
export interface NetworkConnection {
  from?: NetworkNode;
  to?: NetworkNode;
  gater?: NetworkNode | null;
  weight?: number;
  gain?: number;
  enabled?: boolean;
  [key: string]: unknown;
}

/** Encoded maze for simulation. */
export interface EncodedMazeData {
  width: number;
  height: number;
  cells: number[];
  maze?: string[];
  [key: string]: unknown;
}

/** Position in maze. */
export interface MazePosition {
  x: number;
  y: number;
}

/** Distance map for pathfinding. */
export interface MazeDistanceMap {
  width: number;
  height: number;
  distances: number[];
}

/** Ring state for logits tracking. */
export interface LogitsRingState {
  logitsRingCap: number;
  logitsRingShared: boolean;
  scratchLogitsRingW: number;
}

/**
 * Mutable runtime state owned by the public EvolutionEngine facade.
 *
 * The extracted engine modules already share pooled buffers through
 * `engineState`. This narrower state exists only for the facade-specific
 * logits-ring bookkeeping that must survive across runs while keeping the
 * class boundary orchestration-first.
 */
export type EvolutionEngineFacadeRuntimeState = LogitsRingState;

/** Simulation result returned by generation evaluation helpers. */
export interface SimulationResult {
  generationResult: IMazeRunResult;
  simTime: number;
  updatedRingState: LogitsRingState;
}
