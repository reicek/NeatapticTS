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
  /** Base bonus applied for each unique (visited-once) cell, scaled by proximity. */
  static #EXPLORATION_UNIQUE_CELL_BONUS = 200;
  /** Proximity multiplier base (max factor when distance fraction is 0). */
  static #PROXIMITY_MULTIPLIER_BASE = 1.5;
  /** Proximity multiplier slope (value subtracted times normalized distance). */
  static #PROXIMITY_MULTIPLIER_SLOPE = 0.5;
  /** Fixed success bonus when the exit is reached. */
  static #SUCCESS_BONUS = 5000;
  /** Baseline efficiency bonus before subtracting overhead penalty. */
  static #EFFICIENCY_BASE = 8000;
  /** Scale factor converting path overhead percent into penalty. */
  static #EFFICIENCY_PENALTY_SCALE = 80;

  // --- Typed-array scratch buffers (non-reentrant) -------------------------
  /** @internal Scratch array for visit counts (flattened y*width + x). */
  static #VISIT_COUNT_SCRATCH: Uint16Array = new Uint16Array(0);
  static #SCRATCH_WIDTH = 0;
  static #SCRATCH_HEIGHT = 0;

  /** Ensure scratch visit-count buffer has capacity for given maze dims. */
  static #ensureVisitScratch(width: number, height: number) {
    if (width <= 0 || height <= 0) return;
    if (width === this.#SCRATCH_WIDTH && height === this.#SCRATCH_HEIGHT) {
      // Existing buffer is correct size – just clear.
      this.#VISIT_COUNT_SCRATCH.fill(0);
      return;
    }
    this.#SCRATCH_WIDTH = width;
    this.#SCRATCH_HEIGHT = height;
    this.#VISIT_COUNT_SCRATCH = new Uint16Array(width * height); // zeroed
  }
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
    maxAllowedSteps: number
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
      maxAllowedSteps
    );

    // Step 2: Calculate exploration bonus using a pooled typed array (avoids Map + strings).
    // @remarks Not reentrant – shared scratch buffer reused each call.
    let explorationBonus = 0;
    const mazeHeight = encodedMaze.length;
    const mazeWidth = encodedMaze[0]?.length || 0;
    FitnessEvaluator.#ensureVisitScratch(mazeWidth, mazeHeight);
    const visitCountsScratch = FitnessEvaluator.#VISIT_COUNT_SCRATCH;
    const strideWidth = FitnessEvaluator.#SCRATCH_WIDTH; // alias

    // Pass 1: count visits.
    for (let pathIndex = 0; pathIndex < result.path.length; pathIndex++) {
      const [cellX, cellY] = result.path[pathIndex];
      const flatIndex = cellY * strideWidth + cellX;
      visitCountsScratch[flatIndex]++;
    }

    // Pass 2: accumulate bonus for unique cells.
    const dimensionSum = mazeHeight + mazeWidth;
    for (let pathIndex = 0; pathIndex < result.path.length; pathIndex++) {
      const [cellX, cellY] = result.path[pathIndex];
      const flatIndex = cellY * strideWidth + cellX;
      if (visitCountsScratch[flatIndex] !== 1) continue; // only unique cells
      const distanceToExit = distanceMap
        ? distanceMap[cellY]?.[cellX] ?? Infinity
        : MazeUtils.bfsDistance(encodedMaze, [cellX, cellY], exitPosition);
      const proximityMultiplier =
        FitnessEvaluator.#PROXIMITY_MULTIPLIER_BASE -
        FitnessEvaluator.#PROXIMITY_MULTIPLIER_SLOPE *
          (distanceToExit / dimensionSum);
      explorationBonus +=
        FitnessEvaluator.#EXPLORATION_UNIQUE_CELL_BONUS * proximityMultiplier;
    }
    // Optional: zero only touched cells to reduce work; here we clear whole buffer for simplicity.
    visitCountsScratch.fill(0);

    // Step 3: Combine the base fitness with the exploration bonus.
    let fitness = result.fitness + explorationBonus;

    // Step 4: Apply large bonuses for success and efficiency.
    if (result.success) {
      // Success bonus (constant).
      fitness += FitnessEvaluator.#SUCCESS_BONUS;
      // Efficiency bonus scaled by path overhead.
      const optimalPathLength = distanceMap
        ? distanceMap[startPosition[1]]?.[startPosition[0]] ?? Infinity
        : MazeUtils.bfsDistance(encodedMaze, startPosition, exitPosition);
      const pathOverheadPercent =
        ((result.path.length - 1) / optimalPathLength) * 100 - 100;
      const efficiencyBonus = Math.max(
        0,
        FitnessEvaluator.#EFFICIENCY_BASE -
          pathOverheadPercent * FitnessEvaluator.#EFFICIENCY_PENALTY_SCALE
      );
      fitness += efficiencyBonus;
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
