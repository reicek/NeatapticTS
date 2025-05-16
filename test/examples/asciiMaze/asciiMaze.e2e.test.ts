import { Neat, Network, methods } from '../../../src/neataptic';
import seedrandom from 'seedrandom';
import {
  tiny,
  spiralSmall,
  spiral,
  small,
  medium,
  medium2,
  large,
  minotaur
} from './mazes';
import { colors } from './colors';
import {
  encodeMaze,
  findPosition,
  bfsDistance,
} from './mazeUtils';
import {
  simulateAgent
} from './mazeMovement';
import {
  DashboardManager
} from './dashboardManager';
import {
  createTerminalClearer
} from './terminalUtility';
import {
  IDashboardManager,
  IFitnessEvaluationContext,
  IRunMazeEvolutionOptions
} from './interfaces';

// Force console output for this test by writing directly to stdout/stderr
const forceLog = (...args: any[]): void => {
  const message = args.join(' ') + '\n';
  process.stdout.write(message);
};

// Create the dashboard manager
const dashboardManagerInstance: IDashboardManager = new DashboardManager(createTerminalClearer(), forceLog);

// Utility to count walkable tiles in a maze
function countWalkableTiles(maze: string[]): number {
  let count = 0;
  for (const row of maze) {
    for (const cell of row) {
      if (cell === '.' || cell === 'S' || cell === 'E') count++;
    }
  }
  return count + 5;
}

/**
 * Educational Note:
 * The first input neuron encodes the "best direction" as perceived by the agent's vision (0=N, 0.25=E, 0.5=S, 0.75=W).
 * The network learns to correlate this input with the correct output neuron (N/E/S/W).
 * The softmax+argmax output scheme ensures the network's outputs are interpreted as probabilities,
 * making the action selection robust and allowing the agent to confidently choose the best direction.
 * This synergy between input encoding and output selection helps the agent learn more efficiently and robustly.
 */

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
function refineWinnerWithBackprop(winner?: Network) {
  if (!winner) {
    throw new Error('No winning network provided for refinement.');
  }

  const trainingSet = [
    // Input: [best direction encoding, N_open, E_open, S_open, W_open]
    // Output: [N, E, S, W] (one-hot)

    // Ideal cases
    { input: [0, 1, 0, 0, 0], output: [1, 0, 0, 0] },       // North
    { input: [0.25, 0, 1, 0, 0], output: [0, 1, 0, 0] },    // East
    { input: [0.5, 0, 0, 1, 0], output: [0, 0, 1, 0] },     // South
    { input: [0.75, 0, 0, 0, 1], output: [0, 0, 0, 1] },    // West

    // Ambiguous: two directions open, best direction encoding points to one
    { input: [0, 1, 1, 0, 0], output: [1, 0, 0, 0] },       // N and E open, best is N
    { input: [0.25, 1, 1, 0, 0], output: [0, 1, 0, 0] },    // N and E open, best is E
    { input: [0.5, 0, 1, 1, 0], output: [0, 0, 1, 0] },     // E and S open, best is S
    { input: [0.75, 0, 0, 1, 1], output: [0, 0, 0, 1] },    // S and W open, best is W

    // All directions open, best direction encoding points to one
    { input: [0, 1, 1, 1, 1], output: [1, 0, 0, 0] },       // All open, best is N
    { input: [0.25, 1, 1, 1, 1], output: [0, 1, 0, 0] },    // All open, best is E
    { input: [0.5, 1, 1, 1, 1], output: [0, 0, 1, 0] },     // All open, best is S
    { input: [0.75, 1, 1, 1, 1], output: [0, 0, 0, 1] },    // All open, best is W

    // Only one direction open, but best direction encoding is ambiguous
    { input: [0.25, 1, 0, 0, 0], output: [1, 0, 0, 0] },    // Only N open, best encoding is E (should still pick N)
    { input: [0.5, 0, 1, 0, 0], output: [0, 1, 0, 0] },     // Only E open, best encoding is S (should still pick E)

    // No direction open (should not move)
    { input: [0, 0, 0, 0, 0], output: [0, 0, 0, 0] },       // Surrounded by walls

    // Three directions open, best direction encoding points to the closed one (should pick any open)
    { input: [0, 0, 1, 1, 1], output: [0, 1, 0, 0] },       // N closed, best encoding is N (should pick E)
  ];
  const clone = winner.clone();
  
  clone.train(trainingSet, { iterations: 100, error: 0.01, log: false });
  
  return clone;
}

/**
 * Evaluates the fitness of a neural network in solving a maze.
 * This is the specific fitness logic for the current maze problem.
 *
 * @param network - The neural network to evaluate.
 * @param encodedMaze - The numeric representation of the maze.
 * @param startPosition - The agent's starting position [x, y].
 * @param exitPosition - The maze's exit position [x, y].
 * @param maxSteps - Maximum steps the agent can take in one simulation.
 * @returns The fitness score of the network.
 */
function evaluateNetworkFitness(
  network: Network,
  encodedMaze: number[][],
  startPosition: [number, number],
  exitPosition: [number, number],
  maxSteps: number,
): number {
  const result = simulateAgent(network, encodedMaze, startPosition, exitPosition, maxSteps);
  // Eager exploration: reward unique cells, scaled by proximity to exit
  let explorationBonus = 0;
  for (const [x, y] of result.path) {
    const distToExit = bfsDistance(encodedMaze, [x, y], exitPosition);
    // Cells closer to exit get a higher multiplier (max 1.5x, min 1x)
    const proximityMultiplier = 1.5 - 0.5 * (distToExit / (encodedMaze.length + encodedMaze[0].length));
    if (result.path.filter(([px, py]: [number, number]) => px === x && py === y).length === 1) {
      explorationBonus += 200 * proximityMultiplier;
    }
  }
  let fitness = result.fitness + explorationBonus;
  if (result.success) fitness += 5000;
  if (result.success) {
    const optimal = bfsDistance(encodedMaze, startPosition, exitPosition);
    const pathOverhead = ((result.path.length - 1) / optimal) * 100 - 100;
    fitness += Math.max(0, 8000 - pathOverhead * 80);
  }
  return fitness;
}

// Default fitness evaluator using the extracted specific fitness function
function defaultFitnessEvaluator(network: Network, context: IFitnessEvaluationContext): number {
  return evaluateNetworkFitness(
    network,
    context.encodedMaze,
    context.startPosition,
    context.exitPosition,
    context.agentSimConfig.maxSteps
  );
}

describe('ASCII Maze Solver using Neuro-Evolution', () => {
  beforeAll(() => {
    // Print instructions for running with visible logs
    console.log = jest.fn((...args) => process.stdout.write(args.join(' ') + '\n'));
    console.log('\n');
    console.log('=================================================================');
    console.log('To see all ASCII maze visualization and logs, run:');
    console.log(`${colors.bright}${colors.cyan}npm run test:e2e:logs${colors.reset}`);
    console.log('=================================================================');
    console.log('\n');
  });

  /**
   * Executes the neuro-evolution process for an agent to solve a given ASCII maze.
   *
   * This asynchronous function orchestrates the evolutionary algorithm (NEAT) to train a population
   * of neural networks. Each network controls an agent attempting to navigate the maze from a
   * starting point ('S') to an exit ('E'). The function manages generations of agents, evaluates their
   * fitness based on maze-solving performance (e.g., reaching the exit, progress made, steps taken),
   * and applies genetic operators (selection, crossover, mutation) to evolve better solutions over time.
   *
   * It supports various configuration options to tailor the evolution process, such as population size,
   * maximum steps per agent, criteria for stagnation, and logging frequency. It can also start
   * from a pre-existing population or a specific high-performing network.
   *
   * @param options - Grouped configuration options for the maze evolution process, adhering to ISP.
   * @returns A Promise that resolves with the best network, its simulation result, and the NEAT instance.
   */
  async function runMazeEvolution(options: IRunMazeEvolutionOptions) {
    const { mazeConfig, agentSimConfig, evolutionAlgorithmConfig, reportingConfig, fitnessEvaluator } = options;
    const { maze } = mazeConfig;

    const {
      allowRecurrent = true,
      popSize = 500,
      maxStagnantGenerations = 500,
      minProgressToPass = 95,
      randomSeed,
      initialPopulation,
      initialBestNetwork,
    } = evolutionAlgorithmConfig;
    const { logEvery = 1, dashboardManager, label } = reportingConfig;

    const currentFitnessEvaluator = fitnessEvaluator || defaultFitnessEvaluator;

    if (randomSeed !== undefined) {
      seedrandom(randomSeed.toString(), { global: true });
    }
    const inputSize = 1 + 4; // 5 inputs (best direction encoding + open N/E/S/W)
    const outputSize = 4; // 4 outputs for N/E/S/W (softmax/argmax)
    const encodedMaze = encodeMaze(maze);
    const startPosition = findPosition(maze, 'S');
    const exitPosition = findPosition(maze, 'E');

    const fitnessContext: IFitnessEvaluationContext = {
      encodedMaze,
      startPosition,
      exitPosition,
      agentSimConfig, // Pass the whole agentSimConfig
    };

    const neatFitnessCallback = (network: Network): number => {
      return currentFitnessEvaluator(network, fitnessContext);
    };

    const neat = new Neat(inputSize, outputSize, neatFitnessCallback, {
      popsize: popSize,
      mutation: [
        methods.mutation.FFW,
        methods.mutation.ALL,
        methods.mutation.MOD_ACTIVATION,
        methods.mutation.ADD_CONN,
        methods.mutation.SUB_NODE,
        methods.mutation.SUB_CONN,
        methods.mutation.MOD_WEIGHT,
        methods.mutation.MOD_BIAS,
        methods.mutation.MOD_CONNECTION,
        methods.mutation.ADD_LSTM_NODE, // Added for memory evolution
      ],
      mutationRate: 0.1,
      mutationAmount: 0.1,
      elitism: Math.max(1, Math.floor(popSize * 0.1)),
      provenance: Math.max(1, Math.floor(popSize * 0.1)),
      equal: false,
      allowRecurrent,
      minHidden: 24,
    });

    if (initialPopulation && initialPopulation.length > 0) {
      neat.population = initialPopulation.map(net => net.clone());
    }
    if (initialBestNetwork) {
      neat.population[0] = initialBestNetwork.clone();
    }

    let bestNetwork: Network | undefined;
    let bestFitness = -Infinity;
    let bestResult: any;
    let stagnantGenerations = 0;
    let completedGenerations = 0;

    while (true) {
      const fittest = await neat.evolve();
      const fitness = fittest.score ?? 0;
      completedGenerations++;

      const generationResult = simulateAgent(fittest, encodedMaze, startPosition, exitPosition, agentSimConfig.maxSteps);

      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestNetwork = fittest;
        bestResult = generationResult;
        stagnantGenerations = 0;
        dashboardManager.update(maze, generationResult, fittest, completedGenerations);
      } else {
        stagnantGenerations++;
        if (completedGenerations % logEvery === 0) {
          if (bestNetwork && bestResult) {
            dashboardManager.update(maze, bestResult, bestNetwork, completedGenerations);
          }
        }
      }

      if (bestResult?.success && bestResult.progress >= minProgressToPass) {
        if (bestNetwork && bestResult) {
          dashboardManager.update(maze, bestResult, bestNetwork, completedGenerations);
        }
        break;
      }

      if (stagnantGenerations >= maxStagnantGenerations) {
        if (bestNetwork && bestResult) {
          dashboardManager.update(maze, bestResult, bestNetwork, completedGenerations);
        }
        break;
      }
    }

    return {
      bestNetwork,
      bestResult,
      neat,
    };
  }

  // --- TESTS ---
  test('Evolve agent: curriculum learning from small to medium to large to minotaur maze', async () => {
    // Train on tiny
    const tinyResult = await runMazeEvolution({
      mazeConfig: { maze: tiny },
      agentSimConfig: { maxSteps: countWalkableTiles(tiny) + 5 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 1000,
        maxStagnantGenerations: 500,
        minProgressToPass: 99,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'tiny',
      }
    });
    const tinyRefined = refineWinnerWithBackprop(tinyResult?.bestNetwork);

    // spiralSmall
    const spiralSmallResult = await runMazeEvolution({
      mazeConfig: { maze: spiralSmall },
      agentSimConfig: { maxSteps: countWalkableTiles(spiralSmall) + 5 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 400,
        maxStagnantGenerations: 500,
        minProgressToPass: 99,
        initialBestNetwork: tinyRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'spiralSmall',
      }
    });
    const spiralSmallRefined = refineWinnerWithBackprop(spiralSmallResult?.bestNetwork);

    // Spiral
    const spiralResult = await runMazeEvolution({
      mazeConfig: { maze: spiral },
      agentSimConfig: { maxSteps: countWalkableTiles(spiral) + 5 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 500,
        maxStagnantGenerations: 500,
        minProgressToPass: 99,
        initialBestNetwork: spiralSmallRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'spiral',
      }
    });
    const spiralRefined = refineWinnerWithBackprop(spiralResult.bestNetwork);

    // Train on small
    const smallResult = await runMazeEvolution({
      mazeConfig: { maze: small },
      agentSimConfig: { maxSteps: countWalkableTiles(small) + 5 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 500,
        maxStagnantGenerations: 2000,
        minProgressToPass: 99,
        initialBestNetwork: spiralRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'small',
      }
    });
    const smallRefined = refineWinnerWithBackprop(smallResult?.bestNetwork);

    // Medium
    const mediumResult = await runMazeEvolution({
      mazeConfig: { maze: medium },
      agentSimConfig: { maxSteps: 300 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 500,
        maxStagnantGenerations: 2000,
        minProgressToPass: 99,
        initialBestNetwork: smallRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'medium',
      }
    });
    const mediumRefined = refineWinnerWithBackprop(mediumResult?.bestNetwork);

    // Medium 2
    const medium2Result = await runMazeEvolution({
      mazeConfig: { maze: medium2 },
      agentSimConfig: { maxSteps: 300 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 500,
        maxStagnantGenerations: 2000,
        minProgressToPass: 99,
        initialBestNetwork: mediumRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'medium2',
      }
    });
    const medium2Refined = refineWinnerWithBackprop(medium2Result?.bestNetwork);

    // Large
    const largeResult = await runMazeEvolution({
      mazeConfig: { maze: large },
      agentSimConfig: { maxSteps: 800 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 500,
        maxStagnantGenerations: 2000,
        minProgressToPass: 95,
        initialBestNetwork: medium2Refined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'large',
      }
    });
    const largeRefined = refineWinnerWithBackprop(largeResult?.bestNetwork);

    // Minotaur
    await runMazeEvolution({
      mazeConfig: { maze: minotaur },
      agentSimConfig: { maxSteps: 800 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 500,
        maxStagnantGenerations: 2000,
        minProgressToPass: 95,
        initialBestNetwork: largeRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'minotaur',
      }
    });
  }, 0);
});
