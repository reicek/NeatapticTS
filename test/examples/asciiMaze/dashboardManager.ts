/**
 * Dashboard Manager - Handles the visualization dashboard
 *
 * This module contains the DashboardManager class, which manages the
 * state of the dynamic terminal dashboard that displays maze solving progress.
 * It tracks solved mazes, current best solutions, and handles all terminal output
 * for visualizing the agent's progress, network, and statistics.
 */

import { Network } from '../../../src/neataptic';
import { MazeUtils } from './mazeUtils';
import { MazeVisualization } from './mazeVisualization';
import { NetworkVisualization } from './networkVisualization';
import { colors } from './colors';
import { INetwork, IDashboardManager } from './interfaces'; // Added INetwork

/**
 * Stores successful maze solutions and current best networks,
 * and manages all dashboard output and visualization.
 */
export class DashboardManager implements IDashboardManager {
  // Array to store successful solutions for each unique maze
  private solvedMazes: Array<{
    maze: string[];
    result: any;
    network: INetwork; // Changed Network to INetwork
    generation: number;
  }> = [];

  // Tracks which mazes have already been solved to prevent duplicates
  private solvedMazeKeys: Set<string> = new Set<string>();

  // Current best solution for the active maze
  private currentBest: {
    result: any;
    network: INetwork; // Changed Network to INetwork
    generation: number;
  } | null = null;

  // Terminal clearing function
  private clearFunction: () => void;
  // Terminal logging function
  private logFunction: (...args: any[]) => void;

  /**
   * Constructs a DashboardManager.
   * @param clearFn - Function to clear the terminal.
   * @param logFn - Function to log output to the terminal.
   */
  constructor(clearFn: () => void, logFn: (...args: any[]) => void) {
    this.clearFunction = clearFn;
    this.logFunction = logFn;
  }

  /**
   * Creates a unique key for a maze to prevent duplicate solutions.
   * @param maze - The maze array to create a key for.
   * @returns A string that uniquely identifies this maze.
   */
  private getMazeKey(maze: string[]): string {
    // Create a simple hash of the maze by joining all rows
    // This gives us a unique identifier for each unique maze layout
    return maze.join('');
  }

  /**
   * Updates the dashboard with new results.
   * Tracks the current best solution and records solved mazes.
   * @param maze - The current maze being solved.
   * @param result - The result of the latest attempt.
   * @param network - The neural network that produced the result.
   * @param generation - Current generation number.
   */
  update(
    maze: string[],
    result: any,
    network: INetwork,
    generation: number
  ): void {
    // Changed Network to INetwork
    // Save this as current best
    this.currentBest = {
      result,
      network,
      generation,
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
          generation,
        });
        this.solvedMazeKeys.add(mazeKey);
      }
    }

    // Redraw the dashboard
    this.redraw(maze);
  }

  /**
   * Clears the terminal and redraws the dashboard with all content.
   * Displays current best solution, solved mazes, and statistics.
   * @param currentMaze - The maze currently being solved.
   */
  redraw(currentMaze: string[]): void {
    // Clear the screen
    this.clearFunction();

    // Draw dashboard header
    this.logFunction(
      `${colors.blueCore}╔${NetworkVisualization.pad('═', 148, '═')}╗${
        colors.reset
      }`
    );
    this.logFunction(
      `${colors.blueCore}╚${NetworkVisualization.pad(
        '╦════════════╦',
        148,
        '═'
      )}╝${colors.reset}`
    );
    this.logFunction(
      `${colors.blueCore}${NetworkVisualization.pad(
        `║ ${colors.neonYellow}ASCII maze${colors.blueCore} ║`,
        150,
        ' '
      )}${colors.reset}`
    );
    this.logFunction(
      `${colors.blueCore}╔${NetworkVisualization.pad(
        '╩════════════╩',
        148,
        '═'
      )}╗${colors.reset}`
    );

    // Print current best for active maze
    if (this.currentBest) {
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '══════════════════════',
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(
          `${colors.orangeNeon}EVOLVING (GEN ${this.currentBest.generation})`,
          148,
          ' '
        )}${colors.blueCore}║${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '══════════════════════',
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );

      // Visualize the network
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
      this.logFunction(
        NetworkVisualization.visualizeNetworkSummary(this.currentBest.network)
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );

      // Show maze visualization with agent's path
      const lastPos = this.currentBest.result.path[
        this.currentBest.result.path.length - 1
      ];
      const currentMazeVisualization = MazeVisualization.visualizeMaze(
        currentMaze,
        lastPos,
        this.currentBest.result.path
      );
      const currentMazeLines = Array.isArray(currentMazeVisualization)
        ? currentMazeVisualization
        : currentMazeVisualization.split('\n');
      const centeredCurrentMaze = currentMazeLines
        .map(
          (line) =>
            `${colors.blueCore}║${NetworkVisualization.pad(line, 148, ' ')}${
              colors.blueCore
            }║`
        )
        .join('\n');
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
      this.logFunction(centeredCurrentMaze);
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );

      // Print stats for the current best solution
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
      MazeVisualization.printMazeStats(
        this.currentBest,
        currentMaze,
        this.logFunction
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );

      // Show progress bar for agent's progress to exit
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
      this.logFunction(
        `${colors.blueCore}║ ${
          colors.neonSilver
        }Progress to exit: ${MazeVisualization.displayProgressBar(
          this.currentBest.result.progress
        )}`
      );
      this.logFunction(
        `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
          colors.reset
        }`
      );
    }

    // Print footer if there are solved mazes
    if (this.solvedMazes.length > 0) {
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '╦══════════════╦',
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          `╣ ${colors.orangeNeon}SOLVED MAZES${colors.blueCore} ╠`,
          148,
          '═'
        )}╣${colors.reset}`
      );
      this.logFunction(
        `${colors.blueCore}╠${NetworkVisualization.pad(
          '╩══════════════╩',
          148,
          '═'
        )}╣${colors.reset}`
      );
    }

    this.logFunction(
      `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
        colors.reset
      }`
    );
    this.logFunction(
      `${colors.blueCore}║        ${
        colors.neonSilver
      }${NetworkVisualization.pad(
        `Current generation: ${this.currentBest?.generation || 0}`,
        140,
        ' ',
        'left'
      )}${colors.blueCore}║${colors.reset}`
    );
    this.logFunction(
      `${colors.blueCore}║        ${
        colors.neonSilver
      }${NetworkVisualization.pad(
        `Best fitness so far: ${
          this.currentBest?.result.fitness.toFixed(2) || 0
        }`,
        140,
        ' ',
        'left'
      )}${colors.blueCore}║${colors.reset}`
    );
    this.logFunction(
      `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
        colors.reset
      }`
    );

    // Print all solved mazes with their statistics and network summaries
    if (this.solvedMazes.length > 0) {
      // Loop through solved mazes in reverse order (last solved first)
      for (let i = this.solvedMazes.length - 1; i >= 0; i--) {
        const solved = this.solvedMazes[i];
        const endPos = solved.result.path[solved.result.path.length - 1];
        // Calculate display number (last solved is #1)
        const displayNumber = this.solvedMazes.length - i;
        const solvedMazeVisualization = MazeVisualization.visualizeMaze(
          solved.maze,
          endPos,
          solved.result.path
        );
        const solvedMazeLines = Array.isArray(solvedMazeVisualization)
          ? solvedMazeVisualization
          : solvedMazeVisualization.split('\n');
        const centeredSolvedMaze = solvedMazeLines
          .map((line) => NetworkVisualization.pad(line, 150, ' '))
          .join('\n');
        this.logFunction(centeredSolvedMaze);

        // Print efficiency and other stats for the solved maze
        const startPos = MazeUtils.findPosition(solved.maze, 'S');
        const exitPos = MazeUtils.findPosition(solved.maze, 'E');
        const optimalLength = MazeUtils.bfsDistance(
          MazeUtils.encodeMaze(solved.maze),
          startPos,
          exitPos
        );
        const pathLength = solved.result.path.length - 1;
        // Efficiency is the percentage of optimal length to actual (lower = more roundabout path)
        const efficiency = Math.min(
          100,
          Math.round((optimalLength / pathLength) * 100)
        ).toFixed(1);
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

        this.logFunction(
          `${colors.blueCore}║       ${NetworkVisualization.pad(
            `${colors.neonSilver}Path efficiency:${colors.reset} ${optimalLength}/${pathLength} (${efficiency}%)`,
            140,
            ' ',
            'left'
          )} ${colors.blueCore}║${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}║       ${NetworkVisualization.pad(
            `${colors.neonSilver}Path overhead:${colors.reset} ${overhead}% longer than optimal`,
            140,
            ' ',
            'left'
          )} ${colors.blueCore}║${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}║       ${NetworkVisualization.pad(
            `${colors.neonSilver}Unique cells visited:${colors.reset} ${uniqueCells.size}`,
            140,
            ' ',
            'left'
          )} ${colors.blueCore}║${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}║       ${NetworkVisualization.pad(
            `${colors.neonSilver}Cells revisited:${colors.reset} ${revisitedCells} times`,
            140,
            ' ',
            'left'
          )} ${colors.blueCore}║${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}║       ${NetworkVisualization.pad(
            `${colors.neonSilver}Steps:${colors.reset} ${solved.result.steps}`,
            140,
            ' ',
            'left'
          )} ${colors.blueCore}║${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}║       ${NetworkVisualization.pad(
            `${colors.neonSilver}Fitness:${
              colors.reset
            } ${solved.result.fitness.toFixed(2)}`,
            140,
            ' ',
            'left'
          )} ${colors.blueCore}║${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}╠${NetworkVisualization.pad(
            '╦════════════════╦',
            148,
            '═'
          )}╣${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}╠${NetworkVisualization.pad(
            `╣ ${colors.orangeNeon}WINNER NETWORK${colors.blueCore} ╠`,
            148,
            '═'
          )}╣${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}╠${NetworkVisualization.pad(
            '╩════════════════╩',
            148,
            '═'
          )}╣${colors.reset}`
        );
        this.logFunction(
          `${colors.blueCore}║${NetworkVisualization.pad(' ', 148, ' ')}║${
            colors.reset
          }`
        );

        this.logFunction(
          NetworkVisualization.visualizeNetworkSummary(solved.network)
        );
        // Optionally, print detailed network structure for debugging
        // if (this.currentBest?.network) printNetworkStructure(this.currentBest.network);
      }
    }
  }

  /**
   * Clears all saved state, including solved mazes and current best.
   */
  reset(): void {
    this.solvedMazes = [];
    this.solvedMazeKeys.clear();
    this.currentBest = null;
  }
}
