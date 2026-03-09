import type Network from '../../../../src/architecture/network';
import type { FitnessEvaluatorFn } from '../fitness.types';
import type {
  IDashboardManager,
  IMazeRunResult,
  INetwork,
} from '../interfaces';

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

/** Configuration options for the evolutionary algorithm used in the ASCII Maze demos. */
export interface IEvolutionAlgorithmConfig {
  allowRecurrent?: boolean;
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
  | 'solved'
  | 'stagnation'
  | 'maxGenerations'
  | 'cancelled'
  | 'aborted';

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
export type NeatInstance = import('../../../../src/neat').default;

/** Type for Network class instance from the neataptic library. */
export type NetworkInstance =
  import('../../../../src/architecture/network').default;

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
 * The NEAT runtime exposes additional mutable fields during evolution, so the
 * engine centralizes those optional members here rather than duplicating local
 * runtime helper interfaces across multiple files.
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

/** NEAT runtime shape needed by telemetry helpers that inspect the population. */
export interface TelemetryNeatLike {
  population?: EvolutionGenomeLike[];
  getTelemetry?: () => unknown;
  [key: string]: unknown;
}

/** Mutation-operation surface read from the NEAT driver at runtime. */
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

/** Network node representation used by engine-side runtime adaptation helpers. */
export interface NetworkNode {
  type?: string;
  bias?: number;
  squash?:
    | string
    | { name?: string }
    | ((x: number, derivate?: boolean) => number);
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

/** Simulation result returned by generation evaluation helpers. */
export interface SimulationResult {
  generationResult: IMazeRunResult;
  simTime: number;
  updatedRingState: LogitsRingState;
}
