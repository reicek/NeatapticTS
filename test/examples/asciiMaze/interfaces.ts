// Interfaces for ASCII Maze Neuroevolution System
// This file centralizes shared interfaces and types for consistency and maintainability.

import { Network } from '../../../src/neataptic';

/**
 * Interface for dashboard manager abstraction.
 * Used for dependency inversion and testability.
 */
export interface IDashboardManager {
  /**
   * Updates the dashboard with the latest maze, result, network, and generation.
   * @param maze - The maze being solved.
   * @param result - The result object from the agent's run.
   * @param network - The neural network used for the run.
   * @param generation - The current generation number.
   */
  update(
    maze: string[],
    result: any,
    network: INetwork,
    generation: number,
    neatInstance?: any // optional Neat instance for advanced telemetry display
  ): void;
}

/**
 * Maze configuration interface.
 * Used to specify the maze layout for evolution or simulation.
 */
export interface IMazeConfig {
  maze: string[];
}

/**
 * Agent simulation configuration interface.
 * Specifies agent simulation parameters such as maximum allowed steps.
 */
export interface IAgentSimulationConfig {
  maxSteps: number;
}

/**
 * Evolution algorithm configuration interface.
 * Specifies parameters for the evolutionary algorithm.
 */
export interface IEvolutionAlgorithmConfig {
  allowRecurrent?: boolean;
  popSize?: number;
  maxStagnantGenerations?: number;
  minProgressToPass?: number;
  maxGenerations?: number; // Safety cap on total generations
  randomSeed?: number;
  initialPopulation?: INetwork[];
  initialBestNetwork?: INetwork;
  lamarckianIterations?: number; // Per-individual refinement steps
  lamarckianSampleSize?: number; // Subsample training patterns for speed
  // Adaptive simplify/pruning phase triggers
  plateauGenerations?: number; // generations with < plateauImprovementThreshold improvement to trigger simplify
  plateauImprovementThreshold?: number; // minimum fitness delta to count as improvement (default 1e-6)
  simplifyDuration?: number; // number of generations to remain in simplify mode
  simplifyPruneFraction?: number; // fraction of weakest connections to disable each simplify generation (0-1)
  simplifyStrategy?: 'weakWeight' | 'weakRecurrentPreferred'; // pruning heuristic
  // Persistence
  persistEvery?: number; // save snapshot every N generations
  persistDir?: string; // directory for saving snapshots
  persistTopK?: number; // number of top genomes to save
  // Dynamic population growth controls
  dynamicPopEnabled?: boolean;
  dynamicPopMax?: number;
  dynamicPopExpandInterval?: number;
  dynamicPopExpandFactor?: number;
  dynamicPopPlateauSlack?: number;
}

/**
 * Fitness evaluation context interface.
 * Provides all necessary information for evaluating a network's fitness in a maze.
 */
export interface IFitnessEvaluationContext {
  encodedMaze: number[][];
  startPosition: [number, number];
  exitPosition: [number, number];
  agentSimConfig: IAgentSimulationConfig;
  distanceMap?: number[][]; // optional cached distance map for performance
}

/**
 * Fitness evaluator function signature.
 * Used for dependency injection of custom fitness functions.
 */
export type FitnessEvaluatorFn = (
  network: INetwork,
  context: IFitnessEvaluationContext
) => number;

/**
 * Reporting and dashboard configuration interface.
 * Used to configure logging and dashboard reporting during evolution.
 */
export interface IReportingConfig {
  logEvery?: number;
  dashboardManager: IDashboardManager;
  label?: string;
}

/**
 * Main options for running maze evolution.
 * Composes all configuration interfaces for a single evolution run.
 */
export interface IRunMazeEvolutionOptions {
  mazeConfig: IMazeConfig;
  agentSimConfig: IAgentSimulationConfig;
  evolutionAlgorithmConfig: IEvolutionAlgorithmConfig;
  reportingConfig: IReportingConfig;
  fitnessEvaluator?: FitnessEvaluatorFn;
}

/**
 * Visualization node for network visualization (for ASCII/graph rendering).
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
 * Visualization connection for network visualization.
 */
export interface IVisualizationConnection {
  fromUUID: string;
  toUUID: string;
  gaterUUID?: string | null;
  weight: number;
  enabled: boolean;
}

/**
 * Type for activation functions with optional name properties.
 */
export type ActivationFunctionWithName = ((
  input: number,
  derivate?: boolean
) => number) & {
  name?: string;
  originalName?: string;
};

/**
 * Structure for a node within the INetwork interface.
 * Represents a neuron in the network.
 */
export interface INodeStruct {
  type: string; // 'input', 'hidden', 'output', 'constant'
  bias?: number;
  squash?: ActivationFunctionWithName;
  activation?: number;
  name?: string;
  index?: number;
  [key: string]: any;
}

/**
 * Interface for a neural network, used within the asciiMaze example
 * to decouple from the concrete Network class.
 */
export interface INetwork {
  /**
   * Activates the network with the given inputs and returns the outputs.
   * @param inputs - Input values for the network.
   */
  activate: (inputs: number[]) => number[];
  /**
   * Optionally propagates error for supervised learning.
   */
  propagate?: (
    rate: number,
    momentum: number,
    update: boolean,
    target: number[]
  ) => void;
  /**
   * Optionally clears the network's state.
   */
  clear?: () => void;
  /**
   * Optionally clones the network.
   */
  clone?: () => INetwork;
  /**
   * List of nodes in the network.
   */
  nodes?: INodeStruct[];
  /**
   * List of connections in the network.
   */
  connections?: {
    from: INodeStruct;
    to: INodeStruct;
    weight: number;
    gater?: INodeStruct | null;
    enabled?: boolean;
    [key: string]: any;
  }[];
  /**
   * Input layer size or nodes.
   */
  input?: number | INodeStruct[];
  /**
   * Output layer size or nodes.
   */
  output?: number | INodeStruct[];
  // Add any other methods/properties from the concrete Network class that are used by the asciiMaze example
}

/**
 * Represents the result of a single step or phase of an evolution process.
 */
export interface IEvolutionStepResult {
  success: boolean;
  progress: number;
}

/**
 * Represents the overall result of an evolution function call.
 */
export interface IEvolutionFunctionResult {
  finalResult: IEvolutionStepResult;
}
