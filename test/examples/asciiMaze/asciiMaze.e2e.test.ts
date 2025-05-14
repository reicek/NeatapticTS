import { Neat, Network, methods } from '../../../src/neataptic';
import seedrandom from 'seedrandom';
import {
  tiny,
  spiralSmall,
  spiral,
  small,
  medium,
  large,
  minotaur
} from './mazes';
import { colors } from './colors';
import {
  encodeMaze,
  findPosition,
  manhattanDistance,
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

// Force console output for this test by writing directly to stdout/stderr
const forceLog = (...args: any[]): void => {
  const message = args.join(' ') + '\n';
  process.stdout.write(message);
};

// Create the dashboard manager
const dashboardManager = new DashboardManager(createTerminalClearer(), forceLog);

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
   * Run NEAT evolution for the ASCII maze.
   * Optionally accepts an initial population to continue evolution.
   */
  async function runMazeEvolution({
    maze,
    allowRecurrent = true,
    popSize = 500,
    maxSteps = 2500,
    maxStagnantGenerations = 500,
    logEvery = 1,
    minProgressToPass = 95,
    randomSeed,
    initialPopulation,
    initialBestNetwork,
    agentOptsOverride,
    label,
  }: {
    maze: string[];
    allowRecurrent?: boolean;
    popSize?: number;
    maxSteps?: number;
    maxStagnantGenerations?: number;
    logEvery?: number;
    minProgressToPass?: number;
    randomSeed?: number;
    initialPopulation?: Network[];
    initialBestNetwork?: Network;
    agentOptsOverride?: any;
    label?: string;
  }) {
    if (randomSeed !== undefined) {
      seedrandom(randomSeed.toString(), { global: true });
    }    
    // Use the same input vector and agent logic for all mazes, including tiny
    const inputSize = 1 + 4; // 5 inputs (junction factor (1), open N/E/S/W (4))
    const outputSize = 4;
    const encodedMaze = encodeMaze(maze);
    const startPosition = findPosition(maze, 'S');
    const exitPosition = findPosition(maze, 'E');

    function evaluateFitness(network: Network): number {
      const result = simulateAgent(network, encodedMaze, startPosition, exitPosition, maxSteps);
      // Eager exploration: reward unique cells, scaled by proximity to exit
      let uniqueVisited = 0;
      let explorationBonus = 0;
      for (const [x, y] of result.path) {
        const key = `${x},${y}`;
        const distToExit = manhattanDistance([x, y], exitPosition);
        // Cells closer to exit get a higher multiplier (max 1.5x, min 1x)
        const proximityMultiplier = 1.5 - 0.5 * (distToExit / (encodedMaze.length + encodedMaze[0].length));
        if (result.path.filter(([px, py]) => px === x && py === y).length === 1) {
          uniqueVisited++;
          explorationBonus += 200 * proximityMultiplier;
        }
      }      
      let fitness = result.fitness + explorationBonus;
      if (result.success) fitness += 5000;
      if (result.success) {
        const optimal = manhattanDistance(startPosition, exitPosition);
        // Calculate path overhead
        const pathOverhead = ((result.path.length - 1) / optimal) * 100 - 100;
        // Higher bonus for more efficient paths
        // Maximum bonus (8000) for paths with 0% overhead, decreasing as overhead increases
        fitness += Math.max(0, 8000 - pathOverhead * 80);
      }
      return fitness;
    }

    const neat = new Neat(inputSize, outputSize, evaluateFitness, {
      popsize: popSize,
      mutation: [
        methods.mutation.FFW,
        methods.mutation.ALL,
        methods.mutation.MOD_ACTIVATION,
        methods.mutation.ADD_NODE,
        methods.mutation.ADD_CONN,
        methods.mutation.SUB_NODE,
        methods.mutation.SUB_CONN,
        methods.mutation.MOD_WEIGHT,
        methods.mutation.MOD_BIAS,
      ],
      mutationRate: 0.5, // Increased from 0.6 to promote more exploration
      mutationAmount: 0.1, // Increased from 10 for more significant mutations
      elitism: Math.max(1, Math.floor(popSize * 0.1)), // Reduced elitism to prevent premature convergence
      provenance: Math.max(1, Math.floor(popSize * 0.1)), // Increased provenance for more diversity
      equal: false, // Changed to false to allow fitness-proportional selection
      allowRecurrent,
    });

    // If initial population is provided, use it
    if (initialPopulation && initialPopulation.length > 0) {
      neat.population = initialPopulation.map(net => net.clone());
    }
    // If initial best network is provided, inject it as elite
    if (initialBestNetwork) {
      neat.population[0] = initialBestNetwork.clone();
    }

    let bestNetwork: Network | undefined;
    let bestFitness = -Infinity;
    let bestResult: any;
    let bestProgress = 0;
    let stagnantGenerations = 0;
    let completedGenerations = 0;
    let successGeneration = 0;

    // --- Main evolution loop: evolve until solved ---
    while (true) {
      const fittest = await neat.evolve();
      const fitness = fittest.score ?? 0;
      completedGenerations++;

      const generationResult = simulateAgent(fittest, encodedMaze, startPosition, exitPosition, maxSteps);

      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestNetwork = fittest;
        bestResult = generationResult;
        bestProgress = generationResult.progress;
        stagnantGenerations = 0;
        
        // Update the dashboard with the new best result
        dashboardManager.update(maze, generationResult, fittest, completedGenerations);
      } else {
        stagnantGenerations++;
        
        // Update the dashboard every few generations even if not better
        if (completedGenerations % logEvery === 0) {
          dashboardManager.update(maze, bestResult, bestNetwork!, completedGenerations);
        }
      }
      
      // Stop only if the maze is actually solved with a reasonable path overhead
      if (bestResult?.success) {
        successGeneration = completedGenerations;
        // Update the dashboard one last time with the solution
        dashboardManager.update(maze, bestResult, bestNetwork!, completedGenerations);
        break;
      }
      
      // If stagnant for too many generations, give up and move on
      if (stagnantGenerations >= maxStagnantGenerations) {
        // Update the dashboard with the best attempt
        dashboardManager.update(maze, bestResult, bestNetwork!, completedGenerations);
        break;
      }
    }

    const finalResult = bestResult;

    // Return best network and population for transfer learning
    return {
      bestNetwork,
      bestResult: finalResult,
      neat,
    };
  }

  // --- TESTS ---
  test('Evolve agent: curriculum learning from small to medium to large to minotaur maze', async () => {
    // Train on tiny
    const tinyResult = await runMazeEvolution({
      maze: tiny,
      allowRecurrent: true,
      popSize: 1000,
      maxSteps: countWalkableTiles(tiny) + 5, // +5 buffer
      maxStagnantGenerations: 500,
      logEvery: 1,
      minProgressToPass: 99,
      label: 'tiny',
    });
    
    // spiralSmall
    const spiralSmallResult = await runMazeEvolution({
      maze: spiralSmall,
      allowRecurrent: true,
      popSize: 1000,
      maxSteps: countWalkableTiles(spiralSmall) + 5,
      maxStagnantGenerations: 500,
      logEvery: 1,
      minProgressToPass: 99,
      initialPopulation: tinyResult?.neat.population,
      initialBestNetwork: tinyResult?.bestNetwork,
      label: 'spiralSmall',
    });

    // Spiral
    const spiralResult = await runMazeEvolution({
      maze: spiral,
      allowRecurrent: true,
      popSize: 1000,
      maxSteps: countWalkableTiles(spiral) + 5,
      maxStagnantGenerations: 500,
      logEvery: 1,
      minProgressToPass: 99,
      initialPopulation: spiralSmallResult?.neat.population,
      initialBestNetwork: spiralSmallResult?.bestNetwork,
      label: 'spiral',
    });

    // Train on small
    const smallResult = await runMazeEvolution({
      maze: small,
      allowRecurrent: true,
      popSize: 1000,
      maxSteps: countWalkableTiles(small) + 5,
      maxStagnantGenerations: 2000,
      logEvery: 1,
      minProgressToPass: 99,
      initialPopulation: spiralResult?.neat.population,
      initialBestNetwork: spiralResult?.bestNetwork,
      label: 'small',
    });

    // Medium
    const mediumResult = await runMazeEvolution({
      maze: medium,
      allowRecurrent: true,
      popSize: Math.min(500, countWalkableTiles(medium)),
      maxSteps: countWalkableTiles(medium) + 5,
      maxStagnantGenerations: 2000,
      logEvery: 1,
      minProgressToPass: 99,
      initialPopulation: smallResult?.neat.population,
      initialBestNetwork: smallResult?.bestNetwork,
      label: 'medium',
    });

    // Large
    const largeResult = await runMazeEvolution({
      maze: large,
      allowRecurrent: true,
      popSize: Math.min(50, countWalkableTiles(large)),
      maxSteps: countWalkableTiles(large) + 5,
      maxStagnantGenerations: 2000,
      logEvery: 1,
      minProgressToPass: 95,
      initialPopulation: mediumResult?.neat.population,
      initialBestNetwork: mediumResult?.bestNetwork,
      label: 'large',
    });

    // Minotaur
    /*// WIP
    await runMazeEvolution({
      maze: minotaur,
      allowRecurrent: true,
      popSize: Math.min(50, countWalkableTiles(minotaur)),
      maxSteps: countWalkableTiles(minotaur) + 5,
      maxStagnantGenerations: 2000,
      logEvery: 1,
      minProgressToPass: 95,
      initialPopulation: largeResult?.neat.population,
      initialBestNetwork: largeResult?.bestNetwork,
      label: 'minotaur',
    });
    //*/
  }, 0);
});
