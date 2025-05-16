/**
 * Dashboard Manager - Handles the visualization dashboard
 * 
 * This module contains the DashboardManager class, which manages the
 * state of the dynamic terminal dashboard that displays maze solving progress.
 */

import { Network } from '../../../src/neataptic';
import { findPosition, bfsDistance, encodeMaze } from './mazeUtils';
import { visualizeMaze, printMazeStats, displayProgressBar } from './mazeVisualization';
import { visualizeNetworkSummary } from './networkVisualization';
import { colors } from './colors';
import { IDashboardManager } from './interfaces';

// Import centerLine directly for better type resolution
// We explicitly define a local version of the function with matching signature
function centerLine(text: string, width: number = 150, padChar: string = '═'): string {
  const pad = Math.max(0, width - text.length);
  const left = Math.floor(pad / 2);
  const right = pad - left;
  return padChar.repeat(left) + text + padChar.repeat(right);
}

/**
 * Stores successful maze solutions and current best networks
 */
export class DashboardManager implements IDashboardManager {
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
      this.logFunction(`\n${colors.blueCore}${centerLine(`═`, 150, '═')}${colors.reset}`);
      this.logFunction(`\n${colors.blueCore}${centerLine(`  ASCII maze  `, 150, '═')}${colors.reset}`);
    
    // Print current best for active maze
    if (this.currentBest) {
      this.logFunction(`\n${colors.neonCoral}${centerLine(`EVOLVING (GEN ${this.currentBest.generation}) `, 150, '|')}${colors.reset}`);
      
      // Visualize the network
      this.logFunction(visualizeNetworkSummary(this.currentBest.network));
      
      // Show maze visualization with agent's path
      const lastPos = this.currentBest.result.path[this.currentBest.result.path.length - 1];
      this.logFunction(`\n${colors.neonSilver}${centerLine(' CURRENT MAZE ', 150, '═')}${colors.reset}`);
      const currentMazeVisualization = visualizeMaze(currentMaze, lastPos, this.currentBest.result.path);
      const currentMazeLines = Array.isArray(currentMazeVisualization) ? currentMazeVisualization : currentMazeVisualization.split('\n');
      const centeredCurrentMaze = currentMazeLines.map(line => line).join('\n');
      this.logFunction(centeredCurrentMaze);
      
      // Print stats
      printMazeStats(this.currentBest.result, currentMaze, this.logFunction);
      
      // Show progress bar
      this.logFunction(`\n${colors.neonSilver}Progress to exit: ${displayProgressBar(this.currentBest.result.progress)}`);
    }
    
    // Print footer
    if (this.solvedMazes.length > 0) {
      this.logFunction(`${colors.neonSilver}Solved mazes: ${this.solvedMazes.length}${colors.reset}`);
    }
    this.logFunction(`Current generation: ${this.currentBest?.generation || 0}`);
    this.logFunction(`Best fitness so far: ${this.currentBest?.result.fitness.toFixed(2) || 0}`);
    this.logFunction(`\n${colors.neonCoral}${centerLine('|', 150, '|')}${colors.reset}`);
    this.logFunction(`\n${colors.blueCore}${centerLine(`_`, 150, '_')}${colors.reset}`);
    
    // Print all solved mazes
    if (this.solvedMazes.length > 0) {
      this.logFunction(`\n${colors.neonSilver}${centerLine(' SOLVED MAZES ', 150, '═')}${colors.reset}`);
      
      // Loop through solved mazes in reverse order (last solved first)
      for (let i = this.solvedMazes.length - 1; i >= 0; i--) {
        const solved = this.solvedMazes[i];
        const endPos = solved.result.path[solved.result.path.length - 1];
        
        // Calculate display number (last solved is #1)
        const displayNumber = this.solvedMazes.length - i;
        this.logFunction(`\n${colors.bright}${colors.cyanNeon}${centerLine(` SOLVED MAZE #${displayNumber} (GEN ${solved.generation}) `, 150, '═')}${colors.reset}`);
        const solvedMazeVisualization = visualizeMaze(solved.maze, endPos, solved.result.path);
        const solvedMazeLines = Array.isArray(solvedMazeVisualization) ? solvedMazeVisualization : solvedMazeVisualization.split('\n');
        const centeredSolvedMaze = solvedMazeLines.map(line => centerLine(line, 150, ' ')).join('\n');
        this.logFunction(centeredSolvedMaze);
        
        // Print efficiency and other stats
        const startPos = findPosition(solved.maze, 'S');
        const exitPos = findPosition(solved.maze, 'E');        
        const optimalLength = bfsDistance(encodeMaze(solved.maze), startPos, exitPos);
        const pathLength = solved.result.path.length - 1;
        // Efficiency is the percentage of optimal length to actual (lower = more roundabout path)
        const efficiency = Math.min(100, Math.round((optimalLength / pathLength) * 100)).toFixed(1);
        // Overhead is how much longer than optimal the path is (100% means twice as long as optimal)
        const overhead = ((pathLength / optimalLength) * 100 - 100).toFixed(1);
        
        // Calculate unique cells visited vs revisited cells
        const uniqueCells = new Set<string>();
        let revisitedCells = 0;
        
        for (const [x, y] of solved.result.path) {
          const cellKey = `${x},${y}`;
          if (uniqueCells.has(cellKey)) {
            revisitedCells++;
          } else {
            uniqueCells.add(cellKey);
          }
        }
        
        this.logFunction(`${colors.neonSilver}Path efficiency:${colors.reset} ${optimalLength}/${pathLength} (${efficiency}%)`);
        this.logFunction(`${colors.neonSilver}Path overhead:${colors.reset} ${overhead}% longer than optimal`);
        this.logFunction(`${colors.neonSilver}Unique cells visited:${colors.reset} ${uniqueCells.size}`);
        this.logFunction(`${colors.neonSilver}Cells revisited:${colors.reset} ${revisitedCells} times`);
        this.logFunction(`${colors.neonSilver}Steps:${colors.reset} ${solved.result.steps}`);
        this.logFunction(`${colors.neonSilver}Fitness:${colors.reset} ${solved.result.fitness.toFixed(2)}`);
        
        // Add network visualization for the winner network
        this.logFunction(`\n${colors.bright}${colors.whiteNeon}${centerLine(' WINNER NETWORK ', 150, '·')}${colors.reset}`);
        this.logFunction(visualizeNetworkSummary(solved.network));
      }
    }
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