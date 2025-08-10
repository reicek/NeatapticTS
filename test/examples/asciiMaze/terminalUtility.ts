/**
 * Terminal Utility - Handles terminal-specific functions
 *
 * This module contains utility functions for terminal interactions,
 * such as clearing the terminal screen and managing long-running
 * simulation processes.
 */

import {
  INetwork,
  IEvolutionFunctionResult,
  IEvolutionStepResult,
} from './interfaces';

/**
 * TerminalUtility provides static methods for terminal management and
 * simulation control in the ASCII Maze environment.
 */
export class TerminalUtility {
  /**
   * Creates a function that clears the terminal in node.js.
   *
   * @returns A function that when called, will clear the terminal display.
   */
  static createTerminalClearer(): () => void {
    return () => {
      // Clear the terminal using ANSI escape code
      process.stdout.write('\x1Bc');
    };
  }

  /**
   * Helper function for simulation that evolves the agent until it solves
   * the maze or reaches a progress threshold.
   *
   * This function will repeatedly call the provided evolution function
   * until either:
   * - The maze is successfully solved (success=true)
   * - The agent reaches at least the minimum progress threshold
   * - The maximum number of attempts is reached
   *
   * @param evolveFn - Function that evolves the agent and returns results.
   * @param minProgressToPass - Minimum progress (%) to consider evolution successful.
   * @param maxTries - Maximum number of evolution attempts to try.
   * @returns Object containing the final evolution result and number of tries performed.
   */
  static async evolveUntilSolved(
    evolveFn: () => Promise<IEvolutionFunctionResult>,
    minProgressToPass: number = 60,
    maxTries: number = 10
  ): Promise<{ finalResult: IEvolutionStepResult; tries: number }> {
    // Track number of tries and last result
    let tries = 0;
    let lastResult: IEvolutionStepResult = { success: false, progress: 0 };
    while (tries < maxTries) {
      tries++;
      const { finalResult } = await evolveFn();
      lastResult = finalResult;
      if (finalResult.success || finalResult.progress >= minProgressToPass) {
        return { finalResult, tries };
      }
    }
    // Return the last result if unsuccessful after maxTries
    return { finalResult: lastResult, tries };
  }
}
