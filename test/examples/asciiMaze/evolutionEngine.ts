// Handles the main NEAT evolution loop for maze solving
// Exports: EvolutionEngine class with static methods

import { Neat, Network, methods } from '../../../src/neataptic';
import seedrandom from 'seedrandom';
import { MazeUtils } from './mazeUtils';
import { MazeMovement } from './mazeMovement';
import { FitnessEvaluator } from './fitness';
import {
  INetwork,
  IFitnessEvaluationContext,
  IRunMazeEvolutionOptions,
} from './interfaces';

/**
 * The `EvolutionEngine` class encapsulates the entire neuro-evolution process for training agents to solve mazes.
 * It leverages the NEAT (Neuro-Evolution of Augmenting Topologies) algorithm to evolve neural networks.
 * This class is designed as a static utility, meaning you don't need to instantiate it to use its methods.
 *
 * Key Responsibilities:
 * - Orchestrating the main evolution loop (generations, evaluation, selection, reproduction).
 * - Configuring and initializing the NEAT algorithm with appropriate parameters.
 * - Managing a hybrid evolution strategy that combines genetic exploration (NEAT) with local optimization (backpropagation).
 * - Handling curriculum learning, where agents can be trained on a sequence of increasingly difficult mazes.
 * - Providing utilities for logging, visualization, and debugging the evolutionary process.
 */
export class EvolutionEngine {
  /**
   * Runs the NEAT neuro-evolution process for an agent to solve a given ASCII maze.
   *
   * This is the core function of the `EvolutionEngine`. It sets up and runs the evolutionary
   * algorithm to train a population of neural networks. Each network acts as the "brain" for an
   * agent, controlling its movement through the maze from a start point 'S' to an exit 'E'.
   *
   * The process involves several key steps:
   * 1.  **Initialization**: Sets up the maze, NEAT parameters, and the initial population of networks.
   * 2.  **Generational Loop**: Iterates through generations, performing the following for each:
   *     a. **Evaluation**: Each network's performance (fitness) is measured by how well its agent navigates the maze.
   *        Fitness is typically based on progress towards the exit, speed, and efficiency.
   *     b. **Lamarckian Refinement**: Each individual in the population undergoes a brief period of supervised training
   *        (backpropagation) on a set of ideal sensory-action pairs. This helps to fine-tune promising behaviors.
   *     c. **Selection & Reproduction**: The NEAT algorithm selects the fittest individuals to become parents for the
   *        next generation. It uses genetic operators (crossover and mutation) to create offspring.
   * 3.  **Termination**: The loop continues until a solution is found (an agent successfully reaches the exit) or other
   *     stopping criteria are met (e.g., maximum generations, stagnation).
   *
   * This hybrid approach, combining the global search of evolution with the local search of backpropagation,
   * can significantly accelerate learning and lead to more robust solutions.
   *
   * @param options - A comprehensive configuration object for the maze evolution process.
   * @returns A Promise that resolves with an object containing the best network found, its simulation result, and the final NEAT instance.
   */
  static async runMazeEvolution(options: IRunMazeEvolutionOptions) {
    // --- Step 1: Destructure and Default Configuration ---
    // Extract all the necessary configuration objects from the main options parameter.
    const {
      mazeConfig,
      agentSimConfig,
      evolutionAlgorithmConfig,
      reportingConfig,
      fitnessEvaluator,
    } = options;
    const { maze } = mazeConfig;
    const { logEvery = 10, dashboardManager } = reportingConfig;

    // Extract evolution parameters, providing sensible defaults for any that are not specified.
    const {
      allowRecurrent = true, // Allow networks to have connections that loop back, enabling memory.
      popSize = 500, // The number of neural networks in each generation.
      maxStagnantGenerations = 500, // Stop evolution if the best fitness doesn't improve for this many generations.
      minProgressToPass = 95, // The percentage of progress required to consider the maze "solved".
      maxGenerations = Infinity, // A safety cap on the total number of generations to prevent infinite loops.
      randomSeed, // An optional seed for the random number generator to ensure reproducible results.
      initialPopulation, // An optional population of networks to start with.
      initialBestNetwork, // An optional pre-trained network to seed the population.
      lamarckianIterations = 10, // The number of backpropagation steps for each individual per generation.
      lamarckianSampleSize, // If set, use a random subset of the training data for Lamarckian learning.
      plateauGenerations = 40, // Number of generations to wait for improvement before considering the population to be on a plateau.
      plateauImprovementThreshold = 1e-6, // The minimum fitness improvement required to reset the plateau counter.
      simplifyDuration = 30, // The number of generations to run the network simplification process.
      simplifyPruneFraction = 0.05, // The fraction of weak connections to prune during simplification.
      simplifyStrategy = 'weakWeight', // The strategy for choosing which connections to prune.
      persistEvery = 25, // Save a snapshot of the best networks every N generations.
      persistDir = './ascii_maze_snapshots', // The directory to save snapshots in.
      persistTopK = 3, // The number of top-performing networks to save in each snapshot.
      dynamicPopEnabled = true, // Enable dynamic adjustment of the population size.
      dynamicPopMax: dynamicPopMaxCfg, // The maximum population size for dynamic adjustments.
      dynamicPopExpandInterval = 25, // The number of generations between population size expansions.
      dynamicPopExpandFactor = 0.15, // The factor by which to expand the population size.
      dynamicPopPlateauSlack = 0.6, // A slack factor for plateau detection when dynamic population is enabled.
    } = evolutionAlgorithmConfig;

    // Determine the maximum population size, with a fallback if not explicitly configured.
    const dynamicPopMax =
      typeof dynamicPopMaxCfg === 'number'
        ? dynamicPopMaxCfg
        : Math.max(popSize, 120);

    // --- Step 2: Maze and Environment Setup ---
    // Encode the maze into a numerical format (0 for walls, 1 for paths) for efficient processing.
    const encodedMaze = MazeUtils.encodeMaze(maze);
    // Locate the starting 'S' and exit 'E' positions within the maze.
    const startPosition = MazeUtils.findPosition(maze, 'S');
    const exitPosition = MazeUtils.findPosition(maze, 'E');
    // Pre-calculate the distance from every point in the maze to the exit. This is a crucial
    // optimization and provides a rich source of information for the fitness function.
    const distanceMap = MazeUtils.buildDistanceMap(encodedMaze, exitPosition);

    // Define the structure of the neural network: 6 inputs and 4 outputs.
    // Inputs: [compassScalar, openN, openE, openS, openW, progressDelta]
    // Outputs: [moveN, moveE, moveS, moveW]
    const inputSize = 6;
    const outputSize = 4;

    // Select the fitness evaluator function. Use the provided one or a default.
    const currentFitnessEvaluator =
      fitnessEvaluator || FitnessEvaluator.defaultFitnessEvaluator;

    // --- Step 3: Fitness Evaluation Context ---
    // Bundle all the necessary environmental data into a context object. This object will be
    // passed to the fitness function, so it has all the information it needs to evaluate a network.
    const fitnessContext: IFitnessEvaluationContext = {
      encodedMaze,
      startPosition,
      exitPosition,
      agentSimConfig,
      distanceMap,
    };

    // Create the fitness callback function that NEAT will use. This function takes a network,
    // runs the simulation, and returns a single numerical score representing its fitness.
    const neatFitnessCallback = (network: Network): number => {
      return currentFitnessEvaluator(network, fitnessContext);
    };

    // --- Step 4: NEAT Algorithm Initialization ---
    // Create a new instance of the Neat algorithm with a detailed configuration.
    const neat = new Neat(inputSize, outputSize, neatFitnessCallback, {
      popsize: popSize,
      // Define the types of mutations that can occur, allowing for structural evolution.
      mutation: [
        methods.mutation.ADD_NODE,
        methods.mutation.SUB_NODE,
        methods.mutation.ADD_CONN,
        methods.mutation.SUB_CONN,
        methods.mutation.MOD_BIAS,
        methods.mutation.MOD_ACTIVATION,
        methods.mutation.MOD_CONNECTION,
        methods.mutation.ADD_LSTM_NODE, // Allow adding LSTM nodes for more complex memory.
      ],
      mutationRate: 0.2,
      mutationAmount: 0.3,
      elitism: Math.max(1, Math.floor(popSize * 0.1)), // Preserve the top 10% of the population.
      provenance: Math.max(1, Math.floor(popSize * 0.2)), // Keep a portion of the population from previous species.
      allowRecurrent: allowRecurrent,
      minHidden: 6, // Start with a minimum number of hidden nodes.
      // Enable advanced features for more sophisticated evolution.
      adaptiveMutation: { enabled: true, strategy: 'twoTier' },
      multiObjective: {
        enabled: true,
        complexityMetric: 'nodes',
        autoEntropy: true,
      },
      telemetry: {
        enabled: true,
        performance: true,
        complexity: true,
        hypervolume: true,
      },
      lineageTracking: true,
      novelty: {
        enabled: true,
        descriptor: (g: any) => [g.nodes.length, g.connections.length],
        blendFactor: 0.15,
      },
      targetSpecies: 10, // Aim for a target number of species to maintain diversity.
      adaptiveTargetSpecies: {
        enabled: true,
        entropyRange: [0.3, 0.8],
        speciesRange: [6, 14],
        smooth: 0.5,
      },
    });

    // If an initial population is provided, use it to seed the NEAT instance.
    if (initialPopulation && initialPopulation.length > 0) {
      neat.population = initialPopulation.map((net) =>
        (net as Network).clone()
      );
    }
    // If an initial best network is provided, inject it into the population.
    if (initialBestNetwork) {
      neat.population[0] = (initialBestNetwork as Network).clone();
    }

    // --- Step 5: Evolution State Tracking ---
    // Initialize variables to track the progress of the evolution.
    let bestNetwork: INetwork | undefined =
      evolutionAlgorithmConfig.initialBestNetwork;
    let bestFitness = -Infinity;
    let bestResult: any;
    let stagnantGenerations = 0;
    let completedGenerations = 0;
    let plateauCounter = 0;
    let simplifyMode = false;
    let simplifyRemaining = 0;
    let lastBestFitnessForPlateau = -Infinity;

    // --- Step 6: Filesystem and Persistence Setup ---
    // Ensure the directory for saving snapshots exists.
    const fs = require('fs');
    const path = require('path');
    if (persistDir && !fs.existsSync(persistDir)) {
      try {
        fs.mkdirSync(persistDir, { recursive: true });
      } catch (e) {
        console.error(
          `Could not create persistence directory: ${persistDir}`,
          e
        );
      }
    }

    // --- Step 7: Lamarckian Learning Setup ---
    // Define the supervised training set for the Lamarckian refinement process.
    // This dataset consists of idealized sensory inputs and the corresponding optimal actions.
    // It helps to quickly teach the networks basic, correct behaviors.
    /**
     * @const {Array<Object>} lamarckianTrainingSet
     * Encodes idealized agent perceptions and the optimal action for each case.
     * This is used for local search (backpropagation) to refine networks between generations.
     *
     * Input format: `[compassScalar, openN, openE, openS, openW, progressDelta]`
     * - `compassScalar`: Direction to the exit (0=N, 0.25=E, 0.5=S, 0.75=W).
     * - `openN/E/S/W`: Whether the path is open in that direction (1=open, 0=wall).
     * - `progressDelta`: Change in distance to the exit ( >0.5 is good, <0.5 is bad).
     *
     * Output format: A one-hot encoded array representing the desired move `[N, E, S, W]`.
     */
    const lamarckianTrainingSet: {
      input: number[];
      output: number[];
    }[] = (() => {
      const ds: { input: number[]; output: number[] }[] = [];
      // Helper to create a smoothed one-hot output vector.
      const OUT = (d: number) =>
        [0, 1, 2, 3].map((i) => (i === d ? 0.92 : 0.02));
      // Helper to add a new training case.
      const add = (inp: number[], dir: number) =>
        ds.push({ input: inp, output: OUT(dir) });

      // Cases: Single open path with good progress.
      add([0, 1, 0, 0, 0, 0.7], 0); // Go North
      add([0.25, 0, 1, 0, 0, 0.7], 1); // Go East
      add([0.5, 0, 0, 1, 0, 0.7], 2); // Go South
      add([0.75, 0, 0, 0, 1, 0.7], 3); // Go West

      // Cases: Single open path with very strong progress.
      add([0, 1, 0, 0, 0, 0.9], 0);
      add([0.25, 0, 1, 0, 0, 0.9], 1);

      // Cases: Two-way junctions, should follow the compass.
      add([0, 1, 0.6, 0, 0, 0.6], 0);
      add([0, 1, 0, 0.6, 0, 0.6], 0);
      add([0.25, 0.6, 1, 0, 0, 0.6], 1);
      add([0.25, 0, 1, 0.6, 0, 0.6], 1);
      add([0.5, 0, 0.6, 1, 0, 0.6], 2);
      add([0.5, 0, 0, 1, 0.6, 0.6], 2);
      add([0.75, 0, 0, 0.6, 1, 0.6], 3);
      add([0.75, 0.6, 0, 0, 1, 0.6], 3);

      // Cases: Four-way junctions with slight progress, follow compass.
      add([0, 1, 0.8, 0.5, 0.4, 0.55], 0);
      add([0.25, 0.7, 1, 0.6, 0.5, 0.55], 1);
      add([0.5, 0.6, 0.55, 1, 0.65, 0.55], 2);
      add([0.75, 0.5, 0.45, 0.7, 1, 0.55], 3);

      // Cases: Regressing (moving away from exit), should still follow compass to reorient.
      add([0, 1, 0.3, 0, 0, 0.4], 0);
      add([0.25, 0.5, 1, 0.4, 0, 0.4], 1);
      add([0.5, 0, 0.3, 1, 0.2, 0.4], 2);
      add([0.75, 0, 0.5, 0.4, 1, 0.4], 3);
      // Back-only retreat pattern (only opposite available)
      add([0, 0, 0, 0.001, 0, 0.45], 2);
      // Mild augmentation (jitter openness & progress)
      ds.forEach((p) => {
        for (let i = 1; i <= 4; i++)
          if (p.input[i] === 1 && Math.random() < 0.25)
            p.input[i] = 0.95 + Math.random() * 0.05;
        if (Math.random() < 0.35)
          p.input[5] = Math.min(
            1,
            Math.max(0, p.input[5] + (Math.random() * 0.1 - 0.05))
          );
      });
      return ds;
    })();

    // --- Pre-train generation 0 population on supervised compass dataset (Lamarckian warm start) ---
    if (lamarckianTrainingSet.length) {
      // Helper: recenters output node biases to avoid all outputs saturating high simultaneously.
      const centerOutputBiases = (net: any) => {
        try {
          const outs = net.nodes?.filter((n: any) => n.type === 'output');
          if (!outs?.length) return;
          const mean =
            outs.reduce((a: number, n: any) => a + n.bias, 0) / outs.length;
          let varc = 0;
          outs.forEach((n: any) => {
            varc += Math.pow(n.bias - mean, 2);
          });
          varc /= outs.length;
          const std = Math.sqrt(varc);
          outs.forEach((n: any) => {
            n.bias = Math.max(-5, Math.min(5, n.bias - mean)); // subtract mean & clamp
          });
          (net as any)._outputBiasStats = { mean, std };
        } catch {
          /* ignore */
        }
      };
      neat.population.forEach((net: any, idx: number) => {
        try {
          net.train(lamarckianTrainingSet, {
            iterations: Math.min(
              60,
              8 + Math.floor(lamarckianTrainingSet.length / 2)
            ),
            error: 0.01,
            rate: 0.002,
            momentum: 0.1,
            batchSize: 4,
            allowRecurrent: true,
            cost: methods.Cost.softmaxCrossEntropy,
          });
          // Strengthen openness bits -> outputs mapping (inputs 1..4 correspond to N,E,S,W open flags)
          try {
            const outputNodes = net.nodes.filter(
              (n: any) => n.type === 'output'
            );
            const inputNodes = net.nodes.filter((n: any) => n.type === 'input');
            for (let d = 0; d < 4; d++) {
              const inNode = inputNodes[d + 1]; // skip compass scalar at index 0
              const outNode = outputNodes[d];
              if (!inNode || !outNode) continue;
              let conn = net.connections.find(
                (c: any) => c.from === inNode && c.to === outNode
              );
              const w = Math.random() * 0.25 + 0.55; // 0.55..0.8
              if (!conn) net.connect(inNode, outNode, w);
              else conn.weight = w;
            }
            // Light compass scalar fan-out with small weights to allow direction discrimination learning
            const compassNode = inputNodes[0];
            if (compassNode) {
              outputNodes.forEach((out: any, d: number) => {
                let conn = net.connections.find(
                  (c: any) => c.from === compassNode && c.to === out
                );
                const base = 0.05 + d * 0.01; // slight differentiation
                if (!conn) net.connect(compassNode, out, base);
                else conn.weight = base;
              });
            }
          } catch {
            /* ignore */
          }
          centerOutputBiases(net);
        } catch {
          /* ignore training errors */
        }
      });
    }

    // Lightweight profiling (opt-in): set env ASCII_MAZE_PROFILE=1 to enable
    const doProfile = process.env.ASCII_MAZE_PROFILE === '1';
    let tEvolveTotal = 0;
    let tLamarckTotal = 0;
    let tSimTotal = 0;

    while (true) {
      // === Evolutionary Loop ===
      // 1. Darwinian evolution: evolve the population (shuffle genomes)
      //    Evolve one generation and get the fittest network.
      //    This applies selection, crossover, and mutation to produce the next population.
      const t0 = doProfile ? Date.now() : 0;
      const fittest = await neat.evolve();
      if (doProfile) tEvolveTotal += Date.now() - t0;
      // Force identity activation on output nodes; we apply softmax externally (improves gradient richness & avoids early saturation)
      (neat.population || []).forEach((g: any) => {
        g.nodes?.forEach((n: any) => {
          if (n.type === 'output') n.squash = methods.Activation.identity;
        });
      });

      // --- Diversity guardrail: if species collapsed to 1 for 20+ generations, temporarily boost mutation + novelty
      (EvolutionEngine as any)._speciesHistory =
        (EvolutionEngine as any)._speciesHistory || [];
      const speciesCount =
        (neat as any).population?.reduce((set: Set<any>, g: any) => {
          if (g.species) set.add(g.species);
          return set;
        }, new Set()).size || 1;
      (EvolutionEngine as any)._speciesHistory.push(speciesCount);
      if ((EvolutionEngine as any)._speciesHistory.length > 50)
        (EvolutionEngine as any)._speciesHistory.shift();
      const recent = (EvolutionEngine as any)._speciesHistory.slice(-20);
      const collapsed =
        recent.length === 20 && recent.every((c: number) => c === 1);
      if (collapsed) {
        // Temporarily escalate mutation params and novelty blend to force exploration
        const neatAny: any = neat as any;
        if (typeof neatAny.mutationRate === 'number')
          neatAny.mutationRate = Math.min(0.6, neatAny.mutationRate * 1.5);
        if (typeof neatAny.mutationAmount === 'number')
          neatAny.mutationAmount = Math.min(0.8, neatAny.mutationAmount * 1.3);
        if (neatAny.config && neatAny.config.novelty) {
          neatAny.config.novelty.blendFactor = Math.min(
            0.4,
            neatAny.config.novelty.blendFactor * 1.2
          );
        }
      }

      // --- Dynamic population expansion (only grows; does not shrink) ---
      // Rationale: Start with smaller population for faster iterations; expand search space when stagnating.
      if (
        dynamicPopEnabled &&
        completedGenerations > 0 &&
        neat.population?.length &&
        neat.population.length < dynamicPopMax
      ) {
        const plateauRatio =
          plateauGenerations > 0 ? plateauCounter / plateauGenerations : 0;
        const genTrigger =
          completedGenerations % dynamicPopExpandInterval === 0;
        if (genTrigger && plateauRatio >= dynamicPopPlateauSlack) {
          const currentSize = neat.population.length;
          const targetAdd = Math.min(
            Math.max(1, Math.floor(currentSize * dynamicPopExpandFactor)),
            dynamicPopMax - currentSize
          );
          if (targetAdd > 0) {
            // Sort by score descending; use top quarter as parents
            const sorted = neat.population
              .slice()
              .sort(
                (a: any, b: any) =>
                  (b.score || -Infinity) - (a.score || -Infinity)
              );
            const parentPool = sorted.slice(
              0,
              Math.max(2, Math.ceil(sorted.length * 0.25))
            );
            for (let i = 0; i < targetAdd; i++) {
              const parent =
                parentPool[Math.floor(Math.random() * parentPool.length)];
              const clone = parent.clone ? parent.clone() : parent; // defensive
              // Apply a few random mutations to diversify
              const mutateCount = 1 + (Math.random() < 0.5 ? 1 : 0);
              for (let m = 0; m < mutateCount; m++) {
                try {
                  const mutOps = neat.options.mutation || [];
                  if (mutOps.length) {
                    const op =
                      mutOps[Math.floor(Math.random() * mutOps.length)];
                    clone.mutate(op);
                  }
                } catch {
                  /* ignore */
                }
              }
              // Reset score so it will be evaluated newly
              clone.score = undefined;
              neat.population.push(clone);
            }
            neat.options.popsize = neat.population.length; // keep config consistent
            process.stdout.write(
              `[DYNAMIC_POP] Expanded population to ${neat.population.length} at gen ${completedGenerations}\n`
            );
          }
        }
      }

      // 2. Lamarckian evolution: backprop refinement for each individual (everyone goes to school)
      //    Each network is trained with a small number of supervised learning steps on the idealized set.
      //    This directly modifies the weights that will be inherited by the next generation (Lamarckian).
      if (lamarckianIterations > 0 && lamarckianTrainingSet.length) {
        const t1 = doProfile ? Date.now() : 0;
        // Optional sampling to cut cost
        let trainingSetRef = lamarckianTrainingSet;
        if (
          lamarckianSampleSize &&
          lamarckianSampleSize < lamarckianTrainingSet.length
        ) {
          // Reservoir sample simple approach
          const picked: typeof lamarckianTrainingSet = [];
          for (let i = 0; i < lamarckianSampleSize; i++) {
            picked.push(
              lamarckianTrainingSet[
                (Math.random() * lamarckianTrainingSet.length) | 0
              ]
            );
          }
          trainingSetRef = picked;
        }
        let gradNormSum = 0;
        let gradSamples = 0;
        neat.population.forEach((network) => {
          network.train(trainingSetRef, {
            iterations: lamarckianIterations, // Small to preserve diversity
            error: 0.01,
            rate: 0.001,
            momentum: 0.2,
            batchSize: 2,
            allowRecurrent: true, // allow recurrent connections
            cost: methods.Cost.softmaxCrossEntropy,
          });
          // Re-center output biases after local refinement
          try {
            const outs = (network as any).nodes?.filter(
              (n: any) => n.type === 'output'
            );
            if (outs?.length) {
              const mean =
                outs.reduce((a: number, n: any) => a + n.bias, 0) / outs.length;
              let varc = 0;
              outs.forEach((n: any) => {
                varc += Math.pow(n.bias - mean, 2);
              });
              varc /= outs.length;
              const std = Math.sqrt(varc);
              outs.forEach((n: any) => {
                let adjusted = n.bias - mean;
                if (std < 0.25) adjusted *= 0.7; // compress if low variance cluster
                n.bias = Math.max(-5, Math.min(5, adjusted));
              });
            }
          } catch {
            /* ignore */
          }
          // Capture gradient norm stats if available
          try {
            if (typeof (network as any).getTrainingStats === 'function') {
              const ts = (network as any).getTrainingStats();
              if (ts && Number.isFinite(ts.gradNorm)) {
                gradNormSum += ts.gradNorm;
                gradSamples++;
              }
            }
          } catch {
            /* ignore */
          }
        });
        if (gradSamples > 0) {
          process.stdout.write(
            `[GRAD] gen=${completedGenerations} meanGradNorm=${(
              gradNormSum / gradSamples
            ).toFixed(4)} samples=${gradSamples}\n`
          );
        }
        if (doProfile) tLamarckTotal += Date.now() - t1;
      }

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

      // Plateau detection logic
      if (fitness > lastBestFitnessForPlateau + plateauImprovementThreshold) {
        plateauCounter = 0;
        lastBestFitnessForPlateau = fitness;
      } else {
        plateauCounter++;
      }
      // Enter simplify mode
      if (!simplifyMode && plateauCounter >= plateauGenerations) {
        simplifyMode = true;
        simplifyRemaining = simplifyDuration;
        plateauCounter = 0; // reset
      }
      // Apply simplify pruning if active
      if (simplifyMode) {
        // Disable weakest fraction of enabled connections in each genome
        neat.population.forEach((g: any) => {
          const enabledConns = g.connections.filter(
            (c: any) => c.enabled !== false
          );
          if (!enabledConns.length) return;
          const pruneCount = Math.max(
            1,
            Math.floor(enabledConns.length * simplifyPruneFraction)
          );
          let candidates = enabledConns.slice();
          if (simplifyStrategy === 'weakRecurrentPreferred') {
            // Identify recurrent (self-loop or cycle gating) connections first; heuristic: from===to or gater present
            const recurrent = candidates.filter(
              (c: any) => c.from === c.to || c.gater
            );
            const nonRecurrent = candidates.filter(
              (c: any) => !(c.from === c.to || c.gater)
            );
            // Sort each group by absolute weight ascending
            recurrent.sort(
              (a: any, b: any) => Math.abs(a.weight) - Math.abs(b.weight)
            );
            nonRecurrent.sort(
              (a: any, b: any) => Math.abs(a.weight) - Math.abs(b.weight)
            );
            // Prefer pruning weak recurrent connections first, then remaining weak weights
            candidates = [...recurrent, ...nonRecurrent];
          } else {
            candidates.sort(
              (a: any, b: any) => Math.abs(a.weight) - Math.abs(b.weight)
            );
          }
          candidates
            .slice(0, pruneCount)
            .forEach((c: any) => (c.enabled = false));
        });
        simplifyRemaining--;
        if (simplifyRemaining <= 0) simplifyMode = false;
      }

      // Simulate the agent using the fittest network
      // This provides a detailed result (success, progress, steps, etc.)
      const t2 = doProfile ? Date.now() : 0;
      const generationResult = MazeMovement.simulateAgent(
        fittest,
        encodedMaze,
        startPosition,
        exitPosition,
        distanceMap,
        agentSimConfig.maxSteps
      );
      // Capture output history from simulation for telemetry (mazeMovement stores on network)
      try {
        (fittest as any)._lastStepOutputs =
          (fittest as any)._lastStepOutputs ||
          (fittest as any)._lastStepOutputs;
      } catch {}
      // Attach auxiliary metrics to fittest genome for potential external analysis
      (fittest as any)._saturationFraction =
        generationResult.saturationFraction;
      (fittest as any)._actionEntropy = generationResult.actionEntropy;
      // Saturation-based pruning heuristic: if outputs are chronically saturated reduce weak outgoing weights
      if (
        generationResult.saturationFraction &&
        generationResult.saturationFraction > 0.5
      ) {
        try {
          const outNodes = fittest.nodes.filter(
            (n: any) => n.type === 'output'
          );
          // Identify hidden nodes whose outgoing weights converge to all outputs with similar large activations
          const hidden = fittest.nodes.filter((n: any) => n.type === 'hidden');
          hidden.forEach((h: any) => {
            // Gather outgoing connections to outputs
            const outs = h.connections.out.filter(
              (c: any) => outNodes.includes(c.to) && c.enabled !== false
            );
            if (outs.length >= 2) {
              // Compute absolute weight mean & variance
              const weights = outs.map((c: any) => Math.abs(c.weight));
              const mean =
                weights.reduce((a: number, b: number) => a + b, 0) /
                weights.length;
              const varc =
                weights.reduce(
                  (a: number, b: number) => a + Math.pow(b - mean, 2),
                  0
                ) / weights.length;
              if (mean < 0.5 && varc < 0.01) {
                // Likely low-signal uniform fan-out: disable weakest half to force differentiation
                outs.sort(
                  (a: any, b: any) => Math.abs(a.weight) - Math.abs(b.weight)
                );
                const disableCount = Math.max(1, Math.floor(outs.length / 2));
                for (let i = 0; i < disableCount; i++) outs[i].enabled = false;
              }
            }
          });
        } catch {
          /* soft-fail */
        }
      }
      // Instrumentation: log approximate action entropy (based on path move variety)
      if (completedGenerations % logEvery === 0) {
        try {
          const movesRaw = generationResult.path.map(
            (p: [number, number], idx: number, arr: any[]) => {
              if (idx === 0) return null;
              const prev = arr[idx - 1];
              const dx = p[0] - prev[0];
              const dy = p[1] - prev[1];
              if (dx === 0 && dy === -1) return 0;
              if (dx === 1 && dy === 0) return 1;
              if (dx === 0 && dy === 1) return 2;
              if (dx === -1 && dy === 0) return 3;
              return null;
            }
          );
          const moves: number[] = [];
          for (const mv of movesRaw) {
            if (mv !== null) moves.push(mv as number);
          }
          const counts = [0, 0, 0, 0];
          moves.forEach((m: number) => counts[m]++);
          const totalMoves = moves.length || 1;
          const probs = counts.map((c) => c / totalMoves);
          let entropy = 0;
          probs.forEach((p) => {
            if (p > 0) entropy += -p * Math.log(p);
          });
          // natural log entropy max ln(4)=1.386; normalize
          const entropyNorm = entropy / Math.log(4);
          process.stdout.write(
            `[ACTION_ENTROPY] gen=${completedGenerations} entropyNorm=${entropyNorm.toFixed(
              3
            )} uniqueMoves=${counts.filter((c) => c > 0).length} pathLen=${
              generationResult.path.length
            }\n`
          );
          // Output bias stats for fittest network
          try {
            const outs = fittest.nodes.filter((n: any) => n.type === 'output');
            if (outs.length) {
              const meanB =
                outs.reduce((a: number, n: any) => a + n.bias, 0) / outs.length;
              let varcB = 0;
              outs.forEach((n: any) => {
                varcB += Math.pow(n.bias - meanB, 2);
              });
              varcB /= outs.length;
              const stdB = Math.sqrt(varcB);
              process.stdout.write(
                `[OUTPUT_BIAS] gen=${completedGenerations} mean=${meanB.toFixed(
                  3
                )} std=${stdB.toFixed(3)} biases=${outs
                  .map((o: any) => o.bias.toFixed(2))
                  .join(',')}\n`
              );
            }
          } catch {}
          // Enhanced output logits / softmax telemetry (if last step outputs captured)
          try {
            const lastHist: number[][] =
              (fittest as any)._lastStepOutputs || [];
            if (lastHist.length) {
              const recent = lastHist.slice(-40);
              // Aggregate per-output mean & std
              const k = 4;
              const means = new Array(k).fill(0);
              recent.forEach((v) => {
                for (let i = 0; i < k; i++) means[i] += v[i];
              });
              for (let i = 0; i < k; i++) means[i] /= recent.length;
              const stds = new Array(k).fill(0);
              recent.forEach((v) => {
                for (let i = 0; i < k; i++)
                  stds[i] += Math.pow(v[i] - means[i], 2);
              });
              for (let i = 0; i < k; i++)
                stds[i] = Math.sqrt(stds[i] / recent.length);
              // Kurtosis (Fisher, subtract 3)
              const kurt = new Array(k).fill(0);
              recent.forEach((v) => {
                for (let i = 0; i < k; i++)
                  kurt[i] += Math.pow(v[i] - means[i], 4);
              });
              for (let i = 0; i < k; i++) {
                const denom = Math.pow(stds[i] || 1e-9, 4) * recent.length;
                kurt[i] = denom > 0 ? kurt[i] / denom - 3 : 0;
              }
              // Softmax distribution mean entropy over recent steps
              let entAgg = 0;
              recent.forEach((v) => {
                const max = Math.max(...v);
                const exps = v.map((x) => Math.exp(x - max));
                const sum = exps.reduce((a, b) => a + b, 0) || 1;
                const probs = exps.map((e) => e / sum);
                let e = 0;
                probs.forEach((p) => {
                  if (p > 0) e += -p * Math.log(p);
                });
                entAgg += e / Math.log(4);
              });
              const entMean = entAgg / recent.length;
              // Decision stability: fraction of consecutive identical argmax
              let stable = 0,
                totalTrans = 0;
              let prevDir = -1;
              recent.forEach((v) => {
                const arg = v.indexOf(Math.max(...v));
                if (prevDir === arg) stable++;
                if (prevDir !== -1) totalTrans++;
                prevDir = arg;
              });
              const stability = totalTrans ? stable / totalTrans : 0;
              process.stdout.write(
                `[LOGITS] gen=${completedGenerations} means=${means
                  .map((m) => m.toFixed(3))
                  .join(',')} stds=${stds
                  .map((s) => s.toFixed(3))
                  .join(',')} kurt=${kurt
                  .map((kv) => kv.toFixed(2))
                  .join(',')} entMean=${entMean.toFixed(
                  3
                )} stability=${stability.toFixed(3)} steps=${recent.length}\n`
              );
              // Anti-collapse trigger: if all std below threshold & entropy low OR stability extremely high
              (EvolutionEngine as any)._collapseStreak =
                (EvolutionEngine as any)._collapseStreak || 0;
              const collapsed =
                stds.every((s) => s < 0.005) &&
                (entMean < 0.35 || stability > 0.97);
              if (collapsed) (EvolutionEngine as any)._collapseStreak++;
              else (EvolutionEngine as any)._collapseStreak = 0;
              if ((EvolutionEngine as any)._collapseStreak === 6) {
                // Reinitialize a fraction of non-elite population's output weights to break collapse
                try {
                  const eliteCount = neat.options.elitism || 0;
                  const pop = neat.population || [];
                  const reinitTargets = pop
                    .slice(eliteCount)
                    .filter(() => Math.random() < 0.3);
                  let connReset = 0,
                    biasReset = 0;
                  reinitTargets.forEach((g: any) => {
                    const outs = g.nodes.filter(
                      (n: any) => n.type === 'output'
                    );
                    // Reset biases
                    outs.forEach((o: any) => {
                      o.bias = Math.random() * 0.2 - 0.1;
                      biasReset++;
                    });
                    // Reset incoming connection weights to outputs
                    g.connections.forEach((c: any) => {
                      if (outs.includes(c.to)) {
                        c.weight = Math.random() * 0.4 - 0.2;
                        connReset++;
                      }
                    });
                  });
                  process.stdout.write(
                    `[ANTICOLLAPSE] gen=${completedGenerations} reinitGenomes=${reinitTargets.length} connReset=${connReset} biasReset=${biasReset}\n`
                  );
                } catch {
                  /* ignore */
                }
              }
            }
          } catch {}
          // Exploration ratio (unique / path length)
          try {
            const unique = generationResult.path.length
              ? new Set(generationResult.path.map((p: any) => p.join(','))).size
              : 0;
            const ratio = generationResult.path.length
              ? unique / generationResult.path.length
              : 0;
            process.stdout.write(
              `[EXPLORE] gen=${completedGenerations} unique=${unique} pathLen=${
                generationResult.path.length
              } ratio=${ratio.toFixed(
                3
              )} progress=${generationResult.progress.toFixed(
                1
              )} satFrac=${(generationResult as any).saturationFraction?.toFixed(
                3
              )}\n`
            );
          } catch {}
          // Diversity metrics (species distribution + basic connection weight variance)
          try {
            const pop: any[] = neat.population || [];
            const speciesCounts: Record<string, number> = {};
            pop.forEach((g) => {
              const sid = g.species != null ? String(g.species) : 'none';
              speciesCounts[sid] = (speciesCounts[sid] || 0) + 1;
            });
            const counts = Object.values(speciesCounts);
            const total = counts.reduce((a, b) => a + b, 0) || 1;
            const simpson =
              1 - counts.reduce((a, b) => a + Math.pow(b / total, 2), 0); // Simpson diversity index
            // Weight variance sample (subset for speed)
            let wMean = 0,
              wCount = 0;
            const sample = pop.slice(0, Math.min(pop.length, 40));
            sample.forEach((g) => {
              g.connections.forEach((c: any) => {
                if (c.enabled !== false) {
                  wMean += c.weight;
                  wCount++;
                }
              });
            });
            wMean = wCount ? wMean / wCount : 0;
            let wVar = 0;
            sample.forEach((g) => {
              g.connections.forEach((c: any) => {
                if (c.enabled !== false) wVar += Math.pow(c.weight - wMean, 2);
              });
            });
            const wStd = wCount ? Math.sqrt(wVar / wCount) : 0;
            process.stdout.write(
              `[DIVERSITY] gen=${completedGenerations} species=${
                Object.keys(speciesCounts).length
              } simpson=${simpson.toFixed(3)} weightStd=${wStd.toFixed(3)}\n`
            );
          } catch {}
        } catch {}
      }
      if (doProfile) tSimTotal += Date.now() - t2;

      // If new best, update tracking and dashboard
      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestNetwork = fittest;
        bestResult = generationResult;
        stagnantGenerations = 0;
        dashboardManager.update(
          maze,
          generationResult,
          fittest,
          completedGenerations,
          neat
        );
      } else {
        stagnantGenerations++;
        // Periodically update dashboard with current best
        if (completedGenerations % logEvery === 0) {
          if (bestNetwork && bestResult) {
            dashboardManager.update(
              maze,
              bestResult,
              bestNetwork,
              completedGenerations,
              neat
            );
          }
        }
      }

      // Persistence snapshot
      if (
        persistEvery > 0 &&
        completedGenerations % persistEvery === 0 &&
        bestNetwork
      ) {
        try {
          const snap: any = {
            generation: completedGenerations,
            bestFitness: bestFitness,
            simplifyMode,
            plateauCounter,
            timestamp: Date.now(),
            telemetryTail: neat.getTelemetry
              ? neat.getTelemetry().slice(-5)
              : undefined,
          };
          const popSorted = neat.population
            .slice()
            .sort(
              (a: any, b: any) =>
                (b.score || -Infinity) - (a.score || -Infinity)
            );
          const top = popSorted
            .slice(0, persistTopK)
            .map((g: any, idx: number) => ({
              idx,
              score: g.score,
              nodes: g.nodes.length,
              connections: g.connections.length,
              json: g.toJSON ? g.toJSON() : undefined,
            }));
          snap.top = top;
          const file = path.join(
            persistDir,
            `snapshot_gen${completedGenerations}.json`
          );
          fs.writeFileSync(file, JSON.stringify(snap, null, 2));
        } catch (e) {
          // ignore persistence errors
        }
      }

      // Stop if solved or sufficient progress
      if (bestResult?.success && bestResult.progress >= minProgressToPass) {
        if (bestNetwork && bestResult) {
          dashboardManager.update(
            maze,
            bestResult,
            bestNetwork,
            completedGenerations,
            neat
          );
        }
        break;
      }

      // Stop if stagnation limit reached
      if (stagnantGenerations >= maxStagnantGenerations) {
        if (bestNetwork && bestResult) {
          dashboardManager.update(
            maze,
            bestResult,
            bestNetwork,
            completedGenerations,
            neat
          );
        }
        break;
      }

      // Safety cap on generations
      if (completedGenerations >= maxGenerations) {
        break;
      }
    }

    if (doProfile && completedGenerations > 0) {
      const gen = completedGenerations;
      const avgEvolve = (tEvolveTotal / gen).toFixed(2);
      const avgLamarck = (tLamarckTotal / gen).toFixed(2);
      const avgSim = (tSimTotal / gen).toFixed(2);
      // Direct stdout to avoid jest buffering suppression
      process.stdout.write(
        `\n[PROFILE] Generations=${gen} avg(ms): evolve=${avgEvolve} lamarck=${avgLamarck} sim=${avgSim} totalPerGen=${(
          +avgEvolve +
          +avgLamarck +
          +avgSim
        ).toFixed(2)}\n`
      );
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
    const inputNodes = network.nodes?.filter((n) => n.type === 'input');
    const outputNodes = network.nodes?.filter((n) => n.type === 'output');
    const hiddenNodes = network.nodes?.filter((n) => n.type === 'hidden');
    console.log('Input nodes: ', inputNodes?.length); // Number of input nodes
    console.log('Hidden nodes: ', hiddenNodes?.length); // Number of hidden nodes
    console.log('Output nodes: ', outputNodes?.length); // Number of output nodes
    console.log(
      'Activation functions: ',
      network.nodes?.map((n) => n.squash?.name || n.squash)
    ); // List of activation functions
    console.log('Connections: ', network.connections?.length); // Number of connections
    const recurrent = network.connections?.some(
      (c) => c.gater || c.from === c.to
    ); // Whether there are recurrent/gated connections
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
