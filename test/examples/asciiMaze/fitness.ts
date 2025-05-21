// Fitness evaluation logic for maze solving
// Exports: FitnessEvaluator class with static methods

import { INetwork } from './interfaces'; // Added INetwork import
import { MazeUtils } from './mazeUtils';
import { MazeMovement } from './mazeMovement';
import { IFitnessEvaluationContext } from './interfaces';

/**
 * FitnessEvaluator class for evaluating neural network fitness in maze solving.
 * Provides static methods for both direct fitness calculation and NEAT-compatible evaluation.
 */
export class FitnessEvaluator {
  /**
   * Evaluates the fitness of a neural network in solving a maze.
   *
   * The fitness function rewards:
   * - Progress toward the exit (result.fitness)
   * - Exploration of unique cells (explorationBonus)
   * - Success in reaching the exit (large bonus)
   * - Efficiency (shorter paths get more bonus)
   *
   * @param network - The neural network to evaluate.
   * @param encodedMaze - Numeric representation of the maze.
   * @param startPosition - Agent's starting [x, y] position.
   * @param exitPosition - Maze exit [x, y] position.
   * @param maxSteps - Maximum steps allowed in simulation.
   * @returns The computed fitness score for the network.
   */
  static evaluateNetworkFitness(
    network: INetwork, // Changed Network to INetwork
    encodedMaze: number[][],
    startPosition: [number, number],
    exitPosition: [number, number],
    maxSteps: number,
  ): number {
    // Simulate the agent's run through the maze and collect the result
    const result = MazeMovement.simulateAgent(network, encodedMaze, startPosition, exitPosition, maxSteps);
    let explorationBonus = 0;
    // Reward for visiting unique cells, scaled by proximity to exit
    for (const [x, y] of result.path) {
      const distToExit = MazeUtils.bfsDistance(encodedMaze, [x, y], exitPosition);
      // Cells closer to the exit get a higher multiplier (max 1.5x, min 1x)
      const proximityMultiplier = 1.5 - 0.5 * (distToExit / (encodedMaze.length + encodedMaze[0].length));
      // Only reward the first visit to each cell
      if (result.path.filter(([px, py]: [number, number]) => px === x && py === y).length === 1) {
        explorationBonus += 200 * proximityMultiplier;
      }
    }
    // Base fitness from simulation result plus exploration bonus
    let fitness = result.fitness + explorationBonus;
    // Large bonus for reaching the exit
    if (result.success) fitness += 5000;
    // Additional bonus for path efficiency (shorter is better)
    if (result.success) {
      const optimal = MazeUtils.bfsDistance(encodedMaze, startPosition, exitPosition);
      const pathOverhead = ((result.path.length - 1) / optimal) * 100 - 100;
      fitness += Math.max(0, 8000 - pathOverhead * 80);
    }
    return fitness;
  }

  /**
   * Default fitness evaluator for NEAT evolution.
   *
   * This function adapts the generic fitness function to the context object
   * expected by the evolution engine, extracting all required parameters.
   *
   * @param network - The neural network to evaluate.
   * @param context - Fitness evaluation context (maze, positions, agent config).
   * @returns The computed fitness score for the network.
   */
  static defaultFitnessEvaluator(network: INetwork, context: IFitnessEvaluationContext): number {
    return FitnessEvaluator.evaluateNetworkFitness(
      network,
      context.encodedMaze,
      context.startPosition,
      context.exitPosition,
      context.agentSimConfig.maxSteps
    );
  }
}
