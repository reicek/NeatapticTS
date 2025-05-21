/**
 * ASCII Maze - Main export file
 *
 * This file exports all necessary components from the ASCII maze modules,
 * providing a unified interface for external consumers.
 * 
 * All core classes, utilities, and interfaces for the ASCII Maze neuroevolution system
 * are re-exported here for convenient import in other modules or applications.
 */

// Re-export core utility and logic classes for maze solving
export { MazeUtils } from './mazeUtils';
export { MazeVision } from './mazeVision';
export { MazeMovement } from './mazeMovement';
export { MazeVisualization } from './mazeVisualization';
export { NetworkVisualization } from './networkVisualization';
export { DashboardManager } from './dashboardManager';
export { TerminalUtility } from './terminalUtility';
export { FitnessEvaluator } from './fitness';
export { EvolutionEngine } from './evolutionEngine';
export { NetworkRefinement } from './networkRefinement';

// Re-export interfaces and configuration objects for external use
export * from './interfaces';
export { colors } from './colors';
export * from './mazes';