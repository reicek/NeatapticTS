// Network refinement logic (backpropagation after evolution)
// Exports: refineWinnerWithBackprop

import { Network } from '../../../src/neataptic';

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
export function refineWinnerWithBackprop(winner?: Network) {
  if (!winner) {
    throw new Error('No winning network provided for refinement.');
  }

  // Training set: maps idealized sensory states to optimal actions
  const trainingSet = [
    // Input: [best direction encoding, N_open, E_open, S_open, W_open]
    // Output: [N, E, S, W] (one-hot)
    { input: [0, 1, 0, 0, 0], output: [1, 0, 0, 0] },       // North
    { input: [0.25, 0, 1, 0, 0], output: [0, 1, 0, 0] },    // East
    { input: [0.5, 0, 0, 1, 0], output: [0, 0, 1, 0] },     // South
    { input: [0.75, 0, 0, 0, 1], output: [0, 0, 0, 1] },    // West
    // Ambiguous: two directions open, best direction encoding points to one
    { input: [0, 1, 1, 0, 0], output: [1, 0, 0, 0] },       // N and E open, best is N
    { input: [0.25, 1, 1, 0, 0], output: [0, 1, 0, 0] },    // N and E open, best is E
    { input: [0.5, 0, 1, 1, 0], output: [0, 0, 1, 0] },     // E and S open, best is S
    { input: [0.75, 0, 0, 1, 1], output: [0, 0, 0, 1] },    // S and W open, best is W
  ];
  // Clone the winner to avoid mutating the original
  const clone = winner.clone();
  // Train using backpropagation on the idealized dataset
  clone.train(trainingSet, { iterations: 1000, error: 0.001, log: false });
  return clone;
}
