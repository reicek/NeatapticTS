/**
 * ASCII Maze Example - Main file
 *
 * This module brings together all components of the ASCII Maze solving system,
 * providing the core functionality for the neuroevolution-based maze solver.
 *
 * The system demonstrates how a neural network can be evolved to solve
 * increasingly complex mazes through reinforcement learning principles.
 */

// Re-export components from specialized modules as classes for external use.
/** Utility functions for maze manipulation and analysis. */
export { MazeUtils } from './mazeUtils';
/** Provides sensory input (vision) for the agent in the maze. */
export { MazeVision } from './mazeVision';
/** Handles agent movement logic within the maze. */
export { MazeMovement } from './mazeMovement';
/** Visualization utilities for rendering the maze and agent progress. */
export { MazeVisualization } from './mazeVisualization';
/** Manages dashboard output and progress reporting. */
export { DashboardManager } from './dashboardManager';
/** Terminal utilities for clearing and formatting output. */
export { TerminalUtility } from './terminalUtility';

// Export interfaces and maze collection for external configuration and extension.
/** All shared interfaces for the ASCII Maze system. */
export * from './interfaces';
/** Color constants for terminal output. */
export { colors } from './colors';
/** Collection of available maze definitions. */
export * as mazes from './mazes';

/**
 * Main entry point for the ASCII Maze example.
 * If this file is executed directly, it runs the enhanced demonstration environment.
 */
if (require.main === module) {
  // If this file is being run directly (not imported), run the enhanced demo.
  // This sets up the demonstration environment for the ASCII Maze solver.
  require('./enhancedMazeDemo');
}
