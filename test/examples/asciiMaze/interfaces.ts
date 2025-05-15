// Interfaces for ASCII Maze Neuroevolution System
// This file centralizes shared interfaces and types for consistency and maintainability.

import { Network } from '../../../src/neataptic';

/**
 * Interface for dashboard manager abstraction (DIP)
 */
export interface IDashboardManager {
  update(maze: string[], result: any, network: Network, generation: number): void;
}

/**
 * Maze configuration (ISP)
 */
export interface IMazeConfig {
  maze: string[];
}

/**
 * Agent simulation configuration (ISP)
 */
export interface IAgentSimulationConfig {
  maxSteps: number;
}

/**
 * Evolution algorithm configuration (ISP)
 */
export interface IEvolutionAlgorithmConfig {
  allowRecurrent?: boolean;
  popSize?: number;
  maxStagnantGenerations?: number;
  minProgressToPass?: number;
  randomSeed?: number;
  initialPopulation?: Network[];
  initialBestNetwork?: Network;
}

/**
 * Fitness evaluation context (OCP)
 */
export interface IFitnessEvaluationContext {
  encodedMaze: number[][];
  startPosition: [number, number];
  exitPosition: [number, number];
  agentSimConfig: IAgentSimulationConfig;
}

/**
 * Fitness evaluator function signature (OCP)
 */
export type FitnessEvaluatorFn = (network: Network, context: IFitnessEvaluationContext) => number;

/**
 * Reporting and dashboard configuration (ISP)
 */
export interface IReportingConfig {
  logEvery?: number;
  dashboardManager: IDashboardManager;
  label?: string;
}

/**
 * Main options for running maze evolution (composed for SRP/ISP)
 */
export interface IRunMazeEvolutionOptions {
  mazeConfig: IMazeConfig;
  agentSimConfig: IAgentSimulationConfig;
  evolutionAlgorithmConfig: IEvolutionAlgorithmConfig;
  reportingConfig: IReportingConfig;
  fitnessEvaluator?: FitnessEvaluatorFn;
}

/**
 * Visualization node for network visualization (for ASCII/graph rendering)
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
 * Visualization connection for network visualization
 */
export interface IVisualizationConnection {
  fromUUID: string;
  toUUID: string;
  gaterUUID?: string | null;
  weight: number;
  enabled: boolean;
}
