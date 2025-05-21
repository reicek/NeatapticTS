// Handles the main NEAT evolution loop for maze solving
// Exports: EvolutionEngine class with static methods

import { Neat, Network, methods } from '../../../src/neataptic';
import seedrandom from 'seedrandom';
import { MazeUtils } from './mazeUtils';
import { MazeMovement } from './mazeMovement';
import { FitnessEvaluator } from './fitness';
import { INetwork, IFitnessEvaluationContext, IRunMazeEvolutionOptions } from './interfaces';

/**
 * EvolutionEngine class encapsulates the NEAT evolution process for maze-solving agents.
 *
 * This class provides static methods to run the evolution (`runMazeEvolution`) and to print
 * the structure of neural networks (`printNetworkStructure`).
 */
export class EvolutionEngine {
  /**
   * Runs the NEAT neuro-evolution process for an agent to solve a given ASCII maze.
   *
   * This function orchestrates the evolutionary algorithm (NEAT) to train a population
   * of neural networks. Each network controls an agent attempting to navigate the maze
   * from a starting point ('S') to an exit ('E').
   *
   * The function manages generations of agents, evaluates their fitness based on maze-solving
   * performance (e.g., reaching the exit, progress made, steps taken), and applies genetic
   * operators (selection, crossover, mutation) to evolve better solutions over time.
   *
   * The evolution loop supports a hybrid of Lamarckian and Baldwinian evolution:
   *   - After each generation, every individual in the population is refined with a small number
   *     of supervised backpropagation steps on a set of idealized sensory-action pairs (Lamarckian).
   *   - Additionally, the fittest individual receives extra backpropagation (Baldwinian refinement),
   *     further improving its performance for evaluation, but only the Lamarckian-trained weights are
   *     inherited by the next generation.
   *
   * This hybrid approach can accelerate learning by combining global search (evolution) with local search (backpropagation),
   * while balancing diversity and exploitation.
   *
   * @param options - Grouped configuration options for the maze evolution process.
   * @returns A Promise that resolves with the best network, its simulation result, and the NEAT instance.
   */
  static async runMazeEvolution(options: IRunMazeEvolutionOptions) {
    // Destructure configuration options for clarity
    const { mazeConfig, agentSimConfig, evolutionAlgorithmConfig, reportingConfig, fitnessEvaluator } = options;
    const { maze } = mazeConfig;

    // Evolution algorithm parameters with defaults
    const {
      allowRecurrent = true, // Whether to allow recurrent (looping) connections in the networks
      popSize = 500, // Number of individuals in the population
      maxStagnantGenerations = 500, // Stop if no improvement for this many generations
      minProgressToPass = 95, // Minimum percent progress to consider the maze solved
      randomSeed, // Optional: seed for reproducibility
      initialPopulation, // Optional: start from a given population
      initialBestNetwork, // Optional: seed with a known good network
    } = evolutionAlgorithmConfig;
    const { logEvery = 1, dashboardManager } = reportingConfig;

    // Use provided fitness evaluator or default
    const currentFitnessEvaluator = fitnessEvaluator || FitnessEvaluator.defaultFitnessEvaluator;

    // Seed the random number generator for reproducibility if specified
    if (randomSeed !== undefined) {
      seedrandom(randomSeed.toString(), { global: true });
    }
    // Define neural network input/output sizes
    // Inputs: [best direction encoding, open N, open E, open S, open W]
    // Outputs: [N, E, S, W] (one-hot or softmax)
    const inputSize = 1 + 4; // 5 inputs
    const outputSize = 4; // 4 outputs
    // Encode the maze and find start/exit positions
    const encodedMaze = MazeUtils.encodeMaze(maze);
    const startPosition = MazeUtils.findPosition(maze, 'S');
    const exitPosition = MazeUtils.findPosition(maze, 'E');

    // Prepare the fitness evaluation context (passed to the fitness function)
    const fitnessContext: IFitnessEvaluationContext = {
      encodedMaze,
      startPosition,
      exitPosition,
      agentSimConfig,
    };

    // Fitness callback for NEAT (returns a fitness score for a given network)
    const neatFitnessCallback = (network: Network): number => {
      return currentFitnessEvaluator(network, fitnessContext);
    };

    // Initialize NEAT with configuration
    const neat = new Neat(inputSize, outputSize, neatFitnessCallback, {
      popsize: popSize,
      mutation: [
        // methods.mutation.ADD_NODE,
        methods.mutation.SUB_NODE,
        methods.mutation.ADD_CONN,
        methods.mutation.SUB_CONN,
        methods.mutation.MOD_BIAS,
        methods.mutation.MOD_ACTIVATION,
        methods.mutation.MOD_CONNECTION,
        // methods.mutation.ADD_GRU_NODE,
        methods.mutation.ADD_LSTM_NODE,
      ],
      mutationRate: 0.2, // Probability of mutation per individual
      mutationAmount: 0.3, // Proportion of genome to mutate
      elitism: Math.max(1, Math.floor(popSize * 0.1)), // Top 10% survive unchanged
      provenance: Math.max(1, Math.floor(popSize * 0.3)), // New random individuals per generation
      equal: false, // Allow variable network sizes
      allowRecurrent: true, // Allow recurrent connections
      minHidden: 30, // Minimum number of hidden nodes
    });

    // Optionally set initial population or best network
    if (initialPopulation && initialPopulation.length > 0) {
      // Assuming initialPopulation is INetwork[], and Neat expects Network[]
      // This might require a more robust conversion if INetwork isn't directly compatible
      // or if clone() is essential and only on Network.
      // For now, using type assertion, but this is a potential runtime issue if clone() is missing.
      neat.population = initialPopulation.map(net => (net as Network).clone());
    }
    if (initialBestNetwork) {
      // Similar to above, assuming initialBestNetwork is INetwork and needs to be Network
      neat.population[0] = (initialBestNetwork as Network).clone();
    }

    // Track best network and progress
    let bestNetwork: INetwork | undefined = evolutionAlgorithmConfig.initialBestNetwork; // Corrected access
    let bestFitness = -Infinity;
    let bestResult: any;
    let stagnantGenerations = 0;
    let completedGenerations = 0;

    // Main evolution loop
    // Define the supervised training set for Lamarckian refinement
    // Each entry is a sensory input and the optimal action (one-hot)
    /**
     * The lamarckianTrainingSet encodes idealized agent perceptions and the optimal action for each case.
     * This is used for local search (backpropagation) to refine networks between generations.
     *
     * Each input is an array: [best direction encoding, open N, open E, open S, open W]
     * Each output is a one-hot array: [N, E, S, W]
     */
    const lamarckianTrainingSet = [
      // Idealized sensory states to optimal actions
      { input: [0, 1, 0, 0, 0], output: [1, 0, 0, 0] },       // North
      { input: [0.25, 0, 1, 0, 0], output: [0, 1, 0, 0] },    // East
      { input: [0.5, 0, 0, 1, 0], output: [0, 0, 1, 0] },     // South
      { input: [0.75, 0, 0, 0, 1], output: [0, 0, 0, 1] },    // West
        
      { input: [0, 0.9, 0, 0, 0], output: [1, 0, 0, 0] },       // North
      { input: [0.25, 0, 0.9, 0, 0], output: [0, 1, 0, 0] },    // East
      { input: [0.5, 0, 0, 0.9, 0], output: [0, 0, 1, 0] },     // South
      { input: [0.75, 0, 0, 0, 0.9], output: [0, 0, 0, 1] },    // West
      // Ambiguous cases: two directions open, best direction encoding points to one
      // North scenarios
      { input: [0, 1, 1, 0, 0], output: [1, 0, 0, 0] },       // N and E open, best is N
      { input: [0, 1, 0, 1, 0], output: [1, 0, 0, 0] },
      { input: [0, 1, 0, 0, 1], output: [1, 0, 0, 0] },
      { input: [0, 1, 0, 1, 1], output: [1, 0, 0, 0] },       // North
      { input: [0, 1, 1, 0, 1], output: [1, 0, 0, 0] },       // North
      { input: [0, 1, 1, 1, 0], output: [1, 0, 0, 0] },       // North
      { input: [0, 1, 1, 1, 1], output: [1, 0, 0, 0] },       // North
      // East scenarios
      { input: [0.25, 1, 1, 0, 0], output: [0, 1, 0, 0] },    // N and E open, best is E
      { input: [0.25, 0, 1, 1, 0], output: [0, 1, 0, 0] },
      { input: [0.25, 0, 1, 0, 1], output: [0, 1, 0, 0] }, 
      { input: [0.25, 0, 1, 1, 1], output: [0, 1, 0, 0] },    // East
      { input: [0.25, 1, 1, 0, 1], output: [0, 1, 0, 0] },    // East
      { input: [0.25, 1, 1, 1, 0], output: [0, 1, 0, 0] },    // East
      { input: [0.25, 1, 1, 1, 1], output: [0, 1, 0, 0] },    // East
      // South scenarios
      { input: [0.5, 1, 0, 1, 0], output: [0, 0, 1, 0] },     // E and S open, best is S
      { input: [0.5, 0, 1, 1, 0], output: [0, 0, 1, 0] },  
      { input: [0.5, 0, 0, 1, 1], output: [0, 0, 1, 0] },     
      { input: [0.5, 0, 1, 1, 1], output: [0, 0, 1, 0] },     // South
      { input: [0.5, 1, 0, 1, 1], output: [0, 0, 1, 0] },     // South
      { input: [0.5, 1, 1, 1, 0], output: [0, 0, 1, 0] },     // South
      { input: [0.5, 1, 1, 1, 1], output: [0, 0, 1, 0] },     // South
      // West scenarios
      { input: [0.75, 1, 0, 0, 1], output: [0, 0, 0, 1] },    // S and W open, best is W
      { input: [0.75, 0, 1, 0, 1], output: [0, 0, 0, 1] },    
      { input: [0.75, 0, 0, 1, 1], output: [0, 0, 0, 1] }, 
      { input: [0.75, 0, 1, 1, 1], output: [0, 0, 0, 1] },    // West
      { input: [0.75, 1, 0, 1, 1], output: [0, 0, 0, 1] },    // West
      { input: [0.75, 1, 1, 0, 1], output: [0, 0, 0, 1] },    // West
      { input: [0.75, 1, 1, 1, 1], output: [0, 0, 0, 1] },    // West
    ];

    while (true) {
      // === Evolutionary Loop ===
      // 1. Darwinian evolution: evolve the population (shuffle genomes)
      //    Evolve one generation and get the fittest network.
      //    This applies selection, crossover, and mutation to produce the next population.
      const fittest = await neat.evolve();

      // 2. Lamarckian evolution: backprop refinement for each individual (everyone goes to school)
      //    Each network is trained with a small number of supervised learning steps on the idealized set.
      //    This directly modifies the weights that will be inherited by the next generation (Lamarckian).
      neat.population.forEach(network => {
        network.train(lamarckianTrainingSet, {
          iterations: 100, // Small to preserve diversity
          error: 0.01,
          rate: 0.001,
          momentum: 0.2,
          batchSize: 2,
          allowRecurrent: true, // allow recurrent connections
        });
      });

      // 3. Baldwinian refinement: further train the fittest individual for evaluation only.
      //    This improves its performance for this generation's evaluation, but only the Lamarckian-trained
      //    weights are inherited by offspring. (If you want pure Lamarckian, remove this step.)
      /*
      fittest.train(lamarckianTrainingSet, {
        iterations: 1000, // More steps for the fittest
        error: 0.01,
        rate: 0.001,
        momentum: 0.2,
        batchSize: 20,
        allowRecurrent: true, // allow recurrent connections
      });
      */

      // 4. Evaluate and track progress
      const fitness = fittest.score ?? 0;
      completedGenerations++;

      // Simulate the agent using the fittest network
      // This provides a detailed result (success, progress, steps, etc.)
      const generationResult = MazeMovement.simulateAgent(fittest, encodedMaze, startPosition, exitPosition, agentSimConfig.maxSteps);

      // If new best, update tracking and dashboard
      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestNetwork = fittest;
        bestResult = generationResult;
        stagnantGenerations = 0;
        dashboardManager.update(maze, generationResult, fittest, completedGenerations);
      } else {
        stagnantGenerations++;
        // Periodically update dashboard with current best
        if (completedGenerations % logEvery === 0) {
          if (bestNetwork && bestResult) {
            dashboardManager.update(maze, bestResult, bestNetwork, completedGenerations);
          }
        }
      }

      // Stop if solved or sufficient progress
      if (bestResult?.success && bestResult.progress >= minProgressToPass) {
        if (bestNetwork && bestResult) {
          dashboardManager.update(maze, bestResult, bestNetwork, completedGenerations);
        }
        break;
      }

      // Stop if stagnation limit reached
      if (stagnantGenerations >= maxStagnantGenerations) {
        if (bestNetwork && bestResult) {
          dashboardManager.update(maze, bestResult, bestNetwork, completedGenerations);
        }
        break;
      }
    }

    // Return the best network, its result, and the NEAT instance
    return {
      bestNetwork,
      bestResult,
      neat,
    };
  }

  /**
   * Prints the structure of a given neural network to the console.
   *
   * This is useful for debugging and understanding the evolved architectures.
   * It prints the number of nodes, their types, activation functions, and connection details.
   *
   * @param network - The neural network to inspect.
   */
  static printNetworkStructure(network: INetwork) {
    // Print high-level network structure and statistics
    console.log('Network Structure:');
    console.log('Nodes: ', network.nodes?.length); // Total number of nodes
    const inputNodes = network.nodes?.filter(n => n.type === 'input');
    const outputNodes = network.nodes?.filter(n => n.type === 'output');
    const hiddenNodes = network.nodes?.filter(n => n.type === 'hidden');
    console.log('Input nodes: ', inputNodes?.length); // Number of input nodes
    console.log('Hidden nodes: ', hiddenNodes?.length); // Number of hidden nodes
    console.log('Output nodes: ', outputNodes?.length); // Number of output nodes
    console.log('Activation functions: ', network.nodes?.map(n => n.squash?.name || n.squash)); // List of activation functions
    console.log('Connections: ', network.connections?.length); // Number of connections
    const recurrent = network.connections?.some(c => c.gater || c.from === c.to); // Whether there are recurrent/gated connections
    console.log('Has recurrent/gated connections: ', recurrent);
    // if (network.layers) { // Property 'layers' does not exist on type 'INetwork'.
    //   Object.entries(network.layers).forEach(([name, layer]) => {
    //     if (Array.isArray(layer)) {
    //       console.log(`Layer ${name}:`, layer.length, ' nodes');
    //     } else if (layer && typeof layer === 'object' && 'nodes' in layer) {
    //       // For Neataptic Layer objects
    //       // @ts-ignore
    //       console.log(`Layer ${name}: `, layer.nodes.length, ' nodes');
    //     }
    //   });
    // }
  }
}
