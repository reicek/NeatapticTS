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

/** Type for Neat class instance from the neataptic library. */
export type NeatInstance = import('../../../../src/neat').default;

/** Type for Network class instance from the neataptic library. */
export type NetworkInstance =
  import('../../../../src/architecture/network').default;

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
