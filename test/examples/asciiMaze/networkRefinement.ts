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

    // Clone the network to avoid mutating the original
    const networkToRefine = winner.clone();

    // Training set: maps idealized sensory states to optimal actions.
    // Each input/output pair represents a scenario the agent should learn.
    const trainingSet = [
      // Example: If vision indicates North is clear (1,0,0,0), output should be North (1,0,0,0)
      { input: [1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5], output: [1, 0, 0, 0] }, // Ideal: Move North
      { input: [0, 1, 0, 0, 0.5, 0.5, 0.5, 0.5], output: [0, 1, 0, 0] }, // Ideal: Move East
      { input: [0, 0, 1, 0, 0.5, 0.5, 0.5, 0.5], output: [0, 0, 1, 0] }, // Ideal: Move South
      { input: [0, 0, 0, 1, 0.5, 0.5, 0.5, 0.5], output: [0, 0, 0, 1] }, // Ideal: Move West
      // Add more nuanced scenarios if necessary
    ];

    // Training parameters for backpropagation
    const learningRate = 0.05;
    const momentum = 0.01;
    const iterations = 100;

    // Perform backpropagation training for the specified number of iterations
    for (let i = 0; i < iterations; i++) {
      for (const data of trainingSet) {
        // The `propagate` method is part of the concrete `Network` class.
        networkToRefine.propagate(learningRate, momentum, true, data.output);
      }
    }
    // Return the refined network
    return networkToRefine;
  }
}
