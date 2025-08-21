// Fitness evaluation logic for maze solving
// Exports: FitnessEvaluator class with static methods

import { INetwork } from './interfaces'; // Added INetwork import
import { MazeUtils } from './mazeUtils';
import { MazeMovement } from './mazeMovement';
import { IFitnessEvaluationContext } from './interfaces';

/**
 * The `FitnessEvaluator` class is responsible for calculating the fitness of a neural network
 * in the context of solving a maze. Fitness is a numerical score that quantifies how "good" a
 * particular network is at the task. The NEAT algorithm uses this score to select the best
 * networks for reproduction. This class provides static methods, so it doesn't need to be instantiated.
 */
export class FitnessEvaluator {
  /**
   * Evaluates the fitness of a single neural network based on its performance in a maze simulation.
   *
   * This is the core of the fitness calculation. It runs a simulation of the agent controlled
   * by the given network and then calculates a score based on a combination of factors.
   * A well-designed fitness function is crucial for guiding the evolution towards the desired behavior.
   *
   * The fitness function rewards several key behaviors:
   * - **Progress**: How close did the agent get to the exit? This is the primary driver.
   * - **Success**: A large, fixed bonus is awarded for successfully reaching the exit.
   * - **Efficiency**: If the exit is reached, an additional bonus is given for shorter paths.
   *   This encourages the agent to find the most direct route.
   * - **Exploration**: A bonus is given for each unique cell the agent visits. This encourages
   *   the agent to explore the maze rather than getting stuck in a small area. The exploration
   *   bonus is weighted by the cell's proximity to the exit, rewarding exploration in promising areas.
   *
   * @param network - The neural network to be evaluated.
   * @param encodedMaze - A 2D array representing the maze layout.
   * @param startPosition - The agent's starting coordinates `[x, y]`.
   * @param exitPosition - The maze's exit coordinates `[x, y]`.
   * @param distanceMap - A pre-calculated map of distances from each cell to the exit, for performance.
   * @param maxSteps - The maximum number of steps the agent is allowed to take in the simulation.
   * @returns The final computed fitness score for the network.
   */
  static evaluateNetworkFitness(
    network: INetwork,
    encodedMaze: number[][],
    startPosition: readonly [number, number],
    exitPosition: readonly [number, number],
    distanceMap: number[][] | undefined,
    maxSteps: number
  ): number {
    // Step 1: Simulate the agent's journey through the maze using its network "brain".
    // The result object contains detailed statistics about the run, like the path taken,
    // whether the exit was reached, and a base fitness score.
    const result = MazeMovement.simulateAgent(
      network,
      encodedMaze,
      startPosition,
      exitPosition,
      distanceMap,
      maxSteps
    );

    /**
     * @var {number} explorationBonus - A bonus rewarding the agent for exploring new territory.
     */
    let explorationBonus = 0;

    // Step 2: Calculate the exploration bonus.
    // Build a frequency map of visited cells so we can cheaply test whether
    // a cell was visited exactly once. The previous implementation used
    // `result.path.filter(...).length === 1` inside the loop which is O(n^2)
    // for long paths. This preserves behavior but reduces complexity to O(n).
    const visitCounts = new Map<string, number>();
    for (const [x, y] of result.path) {
      const key = `${x},${y}`;
      visitCounts.set(key, (visitCounts.get(key) || 0) + 1);
    }
    for (const [x, y] of result.path) {
      const key = `${x},${y}`;
      // Determine the distance of the current cell from the exit.
      const distToExit = distanceMap
        ? distanceMap[y]?.[x] ?? Infinity
        : MazeUtils.bfsDistance(encodedMaze, [x, y], exitPosition);

      // Cells closer to the exit are more valuable to explore.
      const proximityMultiplier =
        1.5 - 0.5 * (distToExit / (encodedMaze.length + encodedMaze[0].length));

      // Reward only if this cell was visited exactly once (first visit bonus).
      if (visitCounts.get(key) === 1)
        explorationBonus += 200 * proximityMultiplier;
    }

    // Step 3: Combine the base fitness with the exploration bonus.
    let fitness = result.fitness + explorationBonus;

    // Step 4: Apply large bonuses for success and efficiency.
    if (result.success) {
      // A significant, constant bonus for reaching the exit.
      fitness += 5000;

      // An additional bonus for path efficiency.
      // The closer the agent's path length is to the optimal path length, the higher the bonus.
      const optimal = distanceMap
        ? distanceMap[startPosition[1]]?.[startPosition[0]] ?? Infinity
        : MazeUtils.bfsDistance(encodedMaze, startPosition, exitPosition);
      const pathOverhead = ((result.path.length - 1) / optimal) * 100 - 100;
      fitness += Math.max(0, 8000 - pathOverhead * 80);
    }

    // Step 5: Return the final, comprehensive fitness score.
    return fitness;
  }

  /**
   * A wrapper function that serves as the default fitness evaluator for the NEAT evolution process.
   *
   * This function acts as an adapter. The main evolution engine (`EvolutionEngine`) works with a
   * standardized `context` object that bundles all the necessary information for an evaluation.
   * This method simply unpacks that context object and passes the individual parameters to the
   * core `evaluateNetworkFitness` function.
   *
   * @param network - The neural network to be evaluated.
   * @param context - An object containing all the necessary data for the fitness evaluation,
   *                  such as the maze, start/exit positions, and simulation configuration.
   * @returns The computed fitness score for the network.
   */
  static defaultFitnessEvaluator(
    network: INetwork,
    context: IFitnessEvaluationContext
  ): number {
    // Call the main fitness evaluation function with the parameters unpacked from the context object.
    return FitnessEvaluator.evaluateNetworkFitness(
      network,
      context.encodedMaze,
      context.startPosition,
      context.exitPosition,
      context.distanceMap,
      context.agentSimConfig.maxSteps
    );
  }
}
