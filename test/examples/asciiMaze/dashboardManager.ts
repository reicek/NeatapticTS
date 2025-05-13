/**
 * Dashboard Manager - Handles the visualization dashboard
 * 
 * This module contains the DashboardManager class, which manages the
 * state of the dynamic terminal dashboard that displays maze solving progress.
 */

import { Network } from '../../../src/neataptic';
import { findPosition, manhattanDistance } from './mazeUtils';
import { visualizeMaze, printMazeStats, displayProgressBar } from './mazeVisualization';
import { visualizeNetworkSummary } from './networkVisualization';
import { colors } from './colors';

// Import centerLine directly for better type resolution
// We explicitly define a local version of the function with matching signature
function centerLine(text: string, width: number = 150, padChar: string = '='): string {
  const pad = Math.max(0, width - text.length);
  const left = Math.floor(pad / 2);
  const right = pad - left;
  return padChar.repeat(left) + text + padChar.repeat(right);
}

/**
 * Stores successful maze solutions and current best networks
 */
export class DashboardManager {
  // Array to store successful solutions
  private solvedMazes: Array<{
    maze: string[];
    result: any;
    network: Network;
    generation: number;
  }> = [];
  
  // Tracks which mazes have already been solved to prevent duplicates
  private solvedMazeKeys: Set<string> = new Set<string>();
  
  // Current best solution for the active maze
  private currentBest: {
    result: any;
    network: Network;
    generation: number;
  } | null = null;
  
  // Terminal clearing function
  private clearFunction: () => void;
  private logFunction: (...args: any[]) => void;
  
  constructor(clearFn: () => void, logFn: (...args: any[]) => void) {
    this.clearFunction = clearFn;
    this.logFunction = logFn;
  }
  
  /**
   * Creates a unique key for a maze to prevent duplicate solutions
   * 
   * @param maze - The maze array to create a key for
   * @returns A string that uniquely identifies this maze
   */
  private getMazeKey(maze: string[]): string {
    // Create a simple hash of the maze by joining all rows
    // This gives us a unique identifier for each unique maze layout
    return maze.join('');
  }
  
  /**
   * Updates the dashboard with new results
   * 
   * @param maze - The current maze being solved
   * @param result - The result of the latest attempt
   * @param network - The neural network that produced the result
   * @param generation - Current generation number
   */
  update(maze: string[], result: any, network: Network, generation: number): void {
    // Save this as current best
    this.currentBest = {
      result,
      network,
      generation
    };
    
    // If maze was solved and we haven't solved this specific maze before, add to solved mazes list
    if (result.success) {
      const mazeKey = this.getMazeKey(maze);
      
      if (!this.solvedMazeKeys.has(mazeKey)) {
        // This is a new solved maze, add it to our records
        this.solvedMazes.push({
          maze,
          result,
          network,
          generation
        });
        this.solvedMazeKeys.add(mazeKey);
      }
    }
    
    // Redraw the dashboard
    this.redraw(maze);
  }
  
  /**
   * Clears the terminal and redraws the dashboard with all content
   */
  redraw(currentMaze: string[]): void {
    // Clear the screen
    this.clearFunction();
    
    // Print header - using local centerLine function
    this.logFunction(colors.bright + colors.teal + centerLine(' NEUROEVOLUTION MAZE SOLVER DASHBOARD ', 150, '=') + colors.reset);
    
    // Print current best for active maze
    if (this.currentBest) {
      this.logFunction(`\n${colors.bright}${colors.teal}${centerLine(` BEST NETWORK (GEN ${this.currentBest.generation}) `, 150, '-')}${colors.reset}`);
      
      // Visualize the network
      this.logFunction(visualizeNetworkSummary(this.currentBest.network));
      
      // Show maze visualization with agent's path
      const startPos = findPosition(currentMaze, 'S');
      const lastPos = this.currentBest.result.path[this.currentBest.result.path.length - 1];
      this.logFunction(`\n${colors.bright}${colors.teal}${centerLine(' CURRENT MAZE ', 150, '-')}${colors.reset}`);
      this.logFunction(visualizeMaze(currentMaze, lastPos, this.currentBest.result.path));
      
      // Print stats
      printMazeStats(this.currentBest.result, currentMaze, this.logFunction);
      
      // Show progress bar
      this.logFunction(`\nProgress to exit: ${displayProgressBar(this.currentBest.result.progress)}`);
    }
    
    // Print all solved mazes
    if (this.solvedMazes.length > 0) {
      this.logFunction(`\n${colors.bright}${colors.green}${centerLine(' SOLVED MAZES ', 150, '=')}${colors.reset}`);
      
      // Loop through solved mazes in reverse order (last solved first)
      for (let i = this.solvedMazes.length - 1; i >= 0; i--) {
        const solved = this.solvedMazes[i];
        const endPos = solved.result.path[solved.result.path.length - 1];
        
        // Calculate display number (last solved is #1)
        const displayNumber = this.solvedMazes.length - i;
        this.logFunction(`\n${colors.bright}${colors.green}${centerLine(` SOLVED MAZE #${displayNumber} (GEN ${solved.generation}) `, 150, '-')}${colors.reset}`);
        this.logFunction(visualizeMaze(solved.maze, endPos, solved.result.path));
        
        // Print efficiency and other stats
        const startPos = findPosition(solved.maze, 'S');
        const exitPos = findPosition(solved.maze, 'E');
        const optimalLength = manhattanDistance(startPos, exitPos);
        const efficiency = ((optimalLength / (solved.result.path.length - 1)) * 100).toFixed(1);
        this.logFunction(`${colors.bright}Path efficiency:${colors.reset} ${optimalLength}/${solved.result.path.length - 1} (${efficiency}%)`);
        this.logFunction(`${colors.bright}Steps:${colors.reset} ${solved.result.steps}`);
        this.logFunction(`${colors.bright}Fitness:${colors.reset} ${solved.result.fitness.toFixed(2)}`);
        
        // Add network visualization for the winner network
        this.logFunction(`\n${colors.bright}${colors.indigo}${centerLine(' WINNER NETWORK ', 150, '·')}${colors.reset}`);
        this.logFunction(visualizeNetworkSummary(solved.network));
      }
    }
    
    // Print footer
    this.logFunction(`\n${colors.bright}${colors.teal}${centerLine(' EVOLUTION IN PROGRESS ', 150, '=')}${colors.reset}`);
    if (this.solvedMazes.length > 0) {
      this.logFunction(`${colors.bright}${colors.green}Solved mazes: ${this.solvedMazes.length}${colors.reset}`);
    }
    this.logFunction(`Current generation: ${this.currentBest?.generation || 0}`);
    this.logFunction(`Best fitness so far: ${this.currentBest?.result.fitness.toFixed(2) || 0}`);
    this.logFunction(`Best progress so far: ${displayProgressBar(this.currentBest?.result.progress || 0)}`);
  }
  
  /**
   * Clears all saved state
   */
  reset(): void {
    this.solvedMazes = [];
    this.solvedMazeKeys.clear();
    this.currentBest = null;
  }
}