/**
 * ASCII Maze Example - Main file
 * 
 * This module brings together all components of the ASCII Maze solving system,
 * providing the core functionality for the neuroevolution-based maze solver.
 * 
 * The system demonstrates how a neural network can be evolved to solve
 * increasingly complex mazes through reinforcement learning principles.
 */

// Re-export components from specialized modules
export * from './mazeUtils';
export * from './mazeVision';
export * from './mazeMovement';
export * from './mazeEnhancements';

// Export utility and visualization functions
export { colors } from './colors';
export * from './mazeVisualization';
export * from './dashboardManager';
export * from './terminalUtility';

// Export maze collection
export * as mazes from './mazes';

/**
 * Main entry point for the ASCII Maze example.
 * Creates a demonstration environment for running the maze solving system.
 */
if (require.main === module) {
  // If this file is being run directly (not imported), run the enhanced demo
  require('./enhancedMazeDemo');
}
