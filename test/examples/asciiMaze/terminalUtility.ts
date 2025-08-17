/**
 * Terminal Utility - Handles terminal-specific functions
 *
 * This module contains utility functions for terminal interactions,
 * such as clearing the terminal screen and managing long-running
 * simulation processes.
 */

import { IEvolutionFunctionResult, IEvolutionStepResult } from './interfaces';

/**
 * TerminalUtility provides static methods for terminal management and
 * simulation control in the ASCII Maze environment.
 */
export class TerminalUtility {
  /**
   * Returns a function that clears the terminal screen using ANSI escape codes.
   *
   * This is useful for refreshing the terminal display between simulation steps.
   *
   * @returns Function that clears the terminal when called.
   */
  static createTerminalClearer(): () => void {
    return () => {
      // Clear the terminal using ANSI escape code (\x1Bc resets screen)
      process.stdout.write('\x1Bc');
    };
  }

  /**
   * Evolves the agent until it solves the maze or meets a progress threshold.
   *
   * This function repeatedly calls the provided evolution function until one of the following occurs:
   * - The agent successfully solves the maze (success=true)
   * - The agent reaches at least the minimum progress threshold
   * - The maximum number of attempts is reached
   *
   * @param evolveFn - Async function that evolves the agent and returns results.
   * @param minProgressToPass - Minimum progress (%) to consider evolution successful (default: 60).
   * @param maxTries - Maximum number of evolution attempts to try (default: 10).
   * @returns Object containing the final evolution result and number of tries performed.
   */
  static async evolveUntilSolved(
    evolveFn: () => Promise<IEvolutionFunctionResult>,
    minProgressToPass: number = 60,
    maxTries: number = 10
  ): Promise<{ finalResult: IEvolutionStepResult; tries: number }> {
    /**
     * Tracks the number of tries performed.
     */
    let tries = 0;
    /**
     * Stores the last evolution result (used if all attempts fail).
     */
    let lastResult: IEvolutionStepResult = { success: false, progress: 0 };
    while (tries < maxTries) {
      tries++;
      // Await the result of the evolution function
      const { finalResult } = await evolveFn();
      lastResult = finalResult;
      // If solved or progress threshold met, return immediately
      if (finalResult.success || finalResult.progress >= minProgressToPass) {
        return { finalResult, tries };
      }
    }
    // Return the last result if unsuccessful after maxTries
    return { finalResult: lastResult, tries };
  }
}
