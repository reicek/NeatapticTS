/**
 * Terminal Utility - Handles terminal-specific functions
 * 
 * This module contains utility functions for terminal interactions,
 * such as clearing the terminal screen and managing long-running
 * simulation processes.
 */

/**
 * Creates a function that clears the terminal in node.js
 * 
 * @returns A function that when called, will clear the terminal display
 */
export function createTerminalClearer(): () => void {
  return () => {
    // Clear the terminal
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
 * @param evolveFn - Function that evolves the agent and returns results
 * @param minProgressToPass - Minimum progress (%) to consider evolution successful
 * @param maxTries - Maximum number of evolution attempts to try
 * @returns Object containing the final evolution result and number of tries performed
 */
export async function evolveUntilSolved(
  evolveFn: () => Promise<{ finalResult: { success: boolean; progress: number } }>,
  minProgressToPass: number = 60,
  maxTries: number = 10
): Promise<{ finalResult: { success: boolean; progress: number }, tries: number }> {
  let tries = 0;
  let lastResult: { success: boolean; progress: number } = { success: false, progress: 0 };
  while (tries < maxTries) {
    tries++;
    const { finalResult } = await evolveFn();
    lastResult = finalResult;
    if (finalResult.success || finalResult.progress >= minProgressToPass) {
      return { finalResult, tries };
    }
  }
  return { finalResult: lastResult, tries };
}