// Network refinement logic (backpropagation after evolution)
// Exports: NetworkRefinement class with static method

import Network from '../../../src/architecture/network'; // Corrected import for default export

/**
 * NetworkRefinement provides static methods for refining evolved networks
 * using supervised backpropagation after neuro-evolution.
 */
export class NetworkRefinement {
  /**
   * Refines a winning neural network using backpropagation.
   *
   * This function takes a neural network that has successfully solved a maze (a "winner" from
   * neuro-evolution) and further trains it using a supervised learning approach. The goal is to
   * reinforce the associations between specific sensory inputs and the desired motor outputs (actions).
   *
   * The training dataset is predefined and maps idealized sensory states (representing clear environmental
   * perceptions like "path open to the North") to the corresponding optimal action (e.g., "move North").
   * This supervised refinement helps to solidify the network's decision-making logic, potentially
   * improving its robustness and ability to generalize to new, unseen maze configurations or serve
   * as a better starting point for evolving solutions to subsequent, more complex mazes.
   *
   * @param winner - The `Network` instance that previously succeeded in a task, to be refined.
   * @returns A new `Network` instance that is a clone of the winner, further trained via backpropagation.
   * @throws Error if no `winner` network is provided.
   */
  static refineWinnerWithBackprop(winner?: Network): Network {
    if (!winner) {
      throw new Error('A winner network must be provided for refinement.');
    }

    // Clone the network to avoid mutating the original winner
    /**
     * The network instance to be refined (clone of the winner).
     */
    const networkToRefine = winner.clone();

    /**
     * Training set: maps idealized sensory states to optimal actions.
     * Each input/output pair represents a scenario the agent should learn.
     * - input: [N, E, S, W, ...other sensors]
     * - output: [N, E, S, W] (one-hot for direction)
     */
    const trainingSet: ReadonlyArray<{
      input: readonly number[];
      output: readonly number[];
    }> = [
      // If vision indicates North is clear (1,0,0,0), output should be North (1,0,0,0)
      { input: [1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5], output: [1, 0, 0, 0] }, // Move North
      { input: [0, 1, 0, 0, 0.5, 0.5, 0.5, 0.5], output: [0, 1, 0, 0] }, // Move East
      { input: [0, 0, 1, 0, 0.5, 0.5, 0.5, 0.5], output: [0, 0, 1, 0] }, // Move South
      { input: [0, 0, 0, 1, 0.5, 0.5, 0.5, 0.5], output: [0, 0, 0, 1] }, // Move West
      // Add more nuanced scenarios if necessary
    ];

    /**
     * Training parameters for backpropagation:
     * - learningRate: step size for weight updates
     * - momentum: helps accelerate learning and avoid local minima
     * - iterations: number of times to train over the dataset
     */
    const learningRate = 0.05;
    const momentum = 0.01;
    const iterations = 100;

    // Perform backpropagation training for the specified number of iterations
    for (let iter = 0; iter < iterations; iter++) {
      for (const { input, output } of trainingSet) {
        // Use a small defensive wrapper so refinement remains best-effort and
        // any unexpected propagation errors don't abort the overall process.
        NetworkRefinement.#safePropagate(
          networkToRefine,
          learningRate,
          momentum,
          output
        );
      }
    }

    // Return the refined network
    return networkToRefine;
  }

  /**
   * Defensive propagate wrapper used during refinement.
   * Returns true when propagation succeeded, false when an error was caught.
   */
  static #safePropagate(
    net: Network,
    learningRate: number,
    momentum: number,
    target: readonly number[]
  ): boolean {
    try {
      // `propagate` is a concrete implementation detail on `Network` instances.
      // Keep the original call signature (learningRate, momentum, clear, target).
      (net as any).propagate(learningRate, momentum, true, target);
      return true;
    } catch (e) {
      // Best-effort: swallow errors but keep optional debugging available via console when needed.
      // eslint-disable-next-line no-console
      if ((globalThis as any).DEBUG)
        console.warn('Refinement propagate failed:', e);
      return false;
    }
  }
}
