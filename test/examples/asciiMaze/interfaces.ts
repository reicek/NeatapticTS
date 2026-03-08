/**
 * Shared ASCII Maze contracts plus compatibility re-exports for the Step 5 type split.
 *
 * The ownership of run/configuration contracts now lives in `evolutionEngine/`
 * and fitness evaluation contracts now live in `fitness.types.ts`. This file
 * intentionally keeps only the genuinely cross-cutting network, result, and
 * terminal-oriented shapes while re-exporting the moved contracts to preserve
 * the established public import surface.
 */

import type { NeatInstance } from './evolutionEngine/evolutionEngine.types';

export type {
  DistanceMap,
  EncodedMaze,
  EncodedMazeData,
  EvolutionHelpers,
  EvolutionLoopHelpers,
  EvolutionOptions,
  FileSystem,
  IAgentSimulationConfig,
  IEvolutionAlgorithmConfig,
  IMazeConfig,
  IReportingConfig,
  IRunMazeEvolutionOptions,
  LogitsRingState,
  LoopHelpers,
  MazeDistanceMap,
  MazePosition,
  NeatInstance,
  NetworkConnection,
  NetworkInstance,
  NetworkNode,
  PathModule,
  Position,
  ProfilingAccumulators,
  ScratchBundle,
  SimulationResult,
  SnapshotEntry,
  TrainingConstants,
} from './evolutionEngine/evolutionEngine.types';
export type {
  FitnessEvaluatorFn,
  IFitnessEvaluationContext,
} from './fitness.types';

/**
 * Interface for dashboard manager abstraction.
 * Used for dependency inversion and testability.
 */
export interface IDashboardManager {
  /**
   * Update the dashboard with the latest simulation/evolution state.
   *
   * @param maze - The current maze layout represented as an array of ASCII strings.
   * @param result - Result object produced by the agent run.
   * @param network - The network instance used for the run.
   * @param generation - The current generation number.
   * @param neatInstance - Optional NEAT instance for advanced telemetry display.
   */
  update(
    maze: string[],
    result: IMazeRunResult | undefined,
    network: INetwork | null,
    generation: number,
    neatInstance?: NeatInstance,
  ): void;

  /** Optional log function for dashboard messages. */
  logFunction?: (msg: string) => void;

  /** Allow additional properties for extensibility. */
  [key: string]: unknown;
}

/**
 * Result structure returned by the maze simulation and evolution helpers.
 */
export interface IMazeRunResult {
  /** Whether the agent solved the maze during this run. */
  success: boolean;
  /** Number of steps executed before termination. */
  steps: number;
  /** Materialised path as [x, y] coordinates visited sequentially. */
  path: Array<[number, number]>;
  /** Scalar fitness assigned to the run. */
  fitness: number;
  /** Progress metric (usually 0-100) representing completion percentage. */
  progress: number;
  /** Optional saturation fraction of outputs during the run. */
  saturationFraction?: number;
  /** Optional action-entropy metric derived from movement distribution. */
  actionEntropy?: number;
  /** Optional exit reason string used by the evolution loop. */
  exitReason?: string;
  /** Additional diagnostics or telemetry fields supplied by callers. */
  [key: string]: unknown;
}

/**
 * Visualization node used by ASCII and graph renderers to present a network node.
 */
export interface IVisualizationNode {
  uuid: string;
  id: number;
  type: string;
  activation: number;
  bias?: number;
  isAverage?: boolean;
  avgCount?: number;
  label?: string;
}

/**
 * Visualization connection used by ASCII and graph renderers to present an edge between two nodes.
 */
export interface IVisualizationConnection {
  fromUUID: string;
  toUUID: string;
  gaterUUID?: string | null;
  weight: number;
  enabled: boolean;
}

/** Structural connection descriptor referencing resolved node structures. */
export interface IConnectionWithStructRefs {
  from?: INodeStruct | null;
  to?: INodeStruct | null;
  gater?: INodeStruct | null;
  weight?: number;
  enabled?: boolean;
  [key: string]: unknown;
}

/** Aggregates incoming and outgoing link arrays for a node snapshot. */
export interface INodeConnectionRegistry {
  in?: IConnectionWithStructRefs[];
  out?: IConnectionWithStructRefs[];
  gated?: IConnectionWithStructRefs[];
  self?: IConnectionWithStructRefs[];
  [key: string]: unknown;
}

/** Type representing a node activation (squash) function with optional metadata. */
export type ActivationFunctionWithName = ((
  input: number,
  derivate?: boolean,
) => number) & {
  name?: string;
  originalName?: string;
};

/** Structure describing a single network node for visualization, serialization and tooling. */
export interface INodeStruct {
  type: string;
  bias?: number;
  squash?: ActivationFunctionWithName;
  activation?: number;
  name?: string;
  index?: number;
  [key: string]: unknown;
}

/** Extended node snapshot including connection registries for visualisation utilities. */
export interface INodeWithConnectionInfo extends INodeStruct {
  connections?: INodeConnectionRegistry;
}

/**
 * Lightweight neural-network abstraction used across the ASCII Maze example.
 */
export interface INetwork {
  activate: (inputs: number[]) => number[];
  propagate?: (
    rate: number,
    momentum: number,
    update: boolean,
    target: number[],
  ) => void;
  clear?: () => void;
  clone?: () => INetwork;
  nodes?: INodeStruct[];
  connections?: {
    from: INodeStruct;
    to: INodeStruct;
    weight: number;
    gater?: INodeStruct | null;
    enabled?: boolean;
    [key: string]: unknown;
  }[];
  input?: number | INodeStruct[];
  output?: number | INodeStruct[];
}

/** Represents the outcome of a single logical step or checkpoint in the evolution process. */
export interface IEvolutionStepResult {
  success: boolean;
  progress: number;
}

/** Represents the overall result of an evolution function call. */
export interface IEvolutionFunctionResult {
  finalResult: IEvolutionStepResult;
}
