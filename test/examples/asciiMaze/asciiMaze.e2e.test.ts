import { methods } from '../../../src/neataptic';
import Network from '../../../src/architecture/network'; // Correct import for Network
import {
  tiny,
  spiralSmall,
  spiral,
  small,
  medium,
  medium2,
  large,
  minotaur,
} from './mazes';
import { colors } from './colors';
import { DashboardManager } from './dashboardManager';
import { TerminalUtility } from './terminalUtility';
import { IDashboardManager } from './interfaces';
import { EvolutionEngine } from './evolutionEngine';

/**
 * Forces console output for this test by writing directly to stdout/stderr.
 * This is necessary to ensure that log messages are visible during test execution,
 * especially in environments where `console.log` might be mocked or redirected.
 * @param args - The arguments to log, which will be joined into a single string.
 */
const forceLog = (...args: any[]): void => {
  // Step 1: Join all arguments into a single string, separated by spaces, and add a newline character.
  const message = args.join(' ') + '\n';
  // Step 2: Write the formatted message directly to the standard output stream.
  process.stdout.write(message);
};

/**
 * Manages the dashboard for visualizing and logging the progress of the neuro-evolution process.
 * It uses a terminal clearer to create animations and the `forceLog` function to output progress.
 * @type {IDashboardManager}
 */
const dashboardManagerInstance: IDashboardManager = new DashboardManager(
  TerminalUtility.createTerminalClearer(),
  forceLog
);

/**
 * Educational Note: This example demonstrates a powerful combination of neuro-evolution and supervised learning.
 *
 * 1.  **Neuro-evolution (NEAT)**: The primary mechanism for discovering a solution. The network's topology (neurons and connections)
 *     and weights are evolved over generations to solve the maze. This is excellent for exploration and finding novel solutions
 *     in complex problem spaces where the ideal network structure is unknown.
 *
 * 2.  **Input Encoding**: The agent's senses are encoded into the network's input neurons.
 *     - The first neuron receives a "compass" value (0=N, 0.25=E, 0.5=S, 0.75=W), indicating the general direction of the exit.
 *     - The next four neurons represent whether the path is open in each of the four cardinal directions.
 *     - The final neuron represents the change in progress towards the exit, rewarding forward movement.
 *
 * 3.  **Output Interpretation**: The network has four output neurons, one for each cardinal direction. A `softmax` activation function
 *     is used on the output layer, which converts the raw outputs into a probability distribution. This means the outputs sum to 1,
 *     representing the network's confidence in each direction. The `argmax` of these probabilities is chosen as the agent's action.
 *
 * 4.  **Supervised Refinement**: After a successful network (a "winner") is found via evolution, it is further refined using
 *     backpropagation (a supervised learning algorithm). This "fine-tuning" process uses a small, high-quality dataset of
 *     ideal inputs and outputs to solidify the learned rules (e.g., "if the path is only open to the North, move North").
 *
 * This synergistic approach allows the agent to first discover a working strategy through evolution and then perfect it through
 * targeted training, leading to a more robust and efficient solution.
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
  // Step 1: Validate that a winning network was provided.
  if (!winner) {
    throw new Error('No winning network provided for refinement.');
  }

  // Step 2: Ensure all nodes in the network have a valid activation function.
  // Backpropagation requires a differentiable activation function to work correctly.
  // The logistic function is a common and suitable choice.
  winner.nodes.forEach((n) => {
    if (typeof n.squash !== 'function') {
      // Assign the logistic activation function if one is not already set.
      n.squash = methods.Activation.logistic;
    }
  });

  /**
   * @const {Array<Object>} trainingSet - A supervised training dataset for refining the network.
   * This set contains idealized scenarios where there is only one clear path to the exit.
   * The goal is to teach the network the fundamental rule: "If there is only one open direction that leads to progress, take it."
   * Each entry has an `input` array and a corresponding `output` array.
   *
   * Inputs: `[compassScalar, openN, openE, openS, openW, progressDelta]`
   * Outputs: An array of probabilities, with the target direction having a high value (0.92) and others a low value (0.02).
   */
  const trainingSet: { input: number[]; output: number[] }[] = [];

  /**
   * A helper function to create the target output array for the training set.
   * @param {number} d - The index of the correct direction (0:N, 1:E, 2:S, 3:W).
   * @returns {number[]} An array of probabilities for the output layer.
   */
  const OUT = (d: number) => [0, 1, 2, 3].map((i) => (i === d ? 0.92 : 0.02));

  /**
   * A helper function to add a new entry to the training set.
   * @param {number[]} i - The input vector.
   * @param {number} d - The index of the target output direction.
   */
  const add = (i: number[], d: number) =>
    trainingSet.push({ input: i, output: OUT(d) });

  // Step 3: Populate the training set with clear, unambiguous examples.
  // For each cardinal direction, add cases where it's the only open path.
  // Vary the `progressDelta` to ensure the network doesn't overfit to a specific progress value.

  // North-only open scenarios
  add([0, 1, 0, 0, 0, 0.85], 0);
  add([0, 1, 0, 0, 0, 0.65], 0);
  add([0, 1, 0, 0, 0, 0.55], 0); // Case with milder progress

  // East-only open scenarios
  add([0.25, 0, 1, 0, 0, 0.85], 1);
  add([0.25, 0, 1, 0, 0, 0.65], 1);
  add([0.25, 0, 1, 0, 0, 0.55], 1);

  // South-only open scenarios
  add([0.5, 0, 0, 1, 0, 0.85], 2);
  add([0.5, 0, 0, 1, 0, 0.65], 2);
  add([0.5, 0, 0, 1, 0, 0.55], 2);

  // West-only open scenarios
  add([0.75, 0, 0, 0, 1, 0.85], 3);
  add([0.75, 0, 0, 0, 1, 0.65], 3);
  add([0.75, 0, 0, 0, 1, 0.55], 3);

  // Add examples with near-neutral progress to cover more situations.
  add([0, 1, 0, 0, 0, 0.5], 0);
  add([0.25, 0, 1, 0, 0, 0.5], 1);
  add([0.5, 0, 0, 1, 0, 0.5], 2);
  add([0.75, 0, 0, 0, 1, 0.5], 3);

  // This line can be uncommented for deep debugging to analyze the network's state before training.
  // analizeWinner(winner.clone(), trainingSet);

  // Step 4: Train the network using the prepared training set.
  // The `train` method uses backpropagation to adjust the network's weights
  // to minimize the difference between its output and the target output.
  winner.train(trainingSet, {
    iterations: 220, // The maximum number of training iterations.
    error: 0.005, // The target error threshold to stop training.
    rate: 0.001, // The learning rate, controlling the step size of weight adjustments.
    momentum: 0.1, // Momentum helps to avoid local minima and speed up convergence.
    batchSize: 8, // The number of samples to process before updating weights.
    cost: methods.Cost.softmaxCrossEntropy, // A cost function suitable for classification with softmax output.
  });

  // Step 5: Return a clone of the refined network.
  // Cloning ensures that the original object from the evolution process is not mutated,
  // which is good practice for maintaining a clear data flow.
  return winner.clone();
}

/**
 * Analyzes the weights and outputs of a network before and after training.
 * This function is a powerful debugging tool to understand how a network's internal state
 * changes during the supervised training process. It logs connection weights and eligibility
 * traces at each iteration, providing a detailed view of the learning dynamics.
 *
 * @param winner - The network to analyze.
 * @param trainingSet - The training data to use for supervised learning.
 */
function analizeWinner(
  winner: Network,
  trainingSet: { input: number[]; output: number[] }[]
) {
  // Step 1: Log the network's connection weights before training to establish a baseline.
  console.log(
    'Winner weights before training:',
    winner.connections.map((c) => c.weight)
  );

  // Step 2: Train the network with a highly detailed logging schedule.
  // The `schedule` option allows executing a function at specified iterations.
  winner.train(trainingSet, {
    iterations: 2000,
    recurrent: true,
    error: 0.001,
    log: 100, // Log training progress every 100 iterations.
    rate: 0.0001,
    batchSize: 10,
    schedule: {
      iterations: 2000, // The schedule will run for all iterations.
      function: ({ iteration }: { iteration: number }) => {
        // For each node in the network, log the details of its incoming and outgoing connections.
        // This includes the weight, eligibility trace, and total delta weight, which are key
        // metrics in understanding how the network is learning.
        winner.nodes.forEach((n, idx) => {
          n.connections.in.forEach((c, cidx) => {
            console.log(
              `Iter ${iteration}: Node[${idx}] InConn[${cidx}] from Node ${c.from.index} weight=${c.weight} elig=${c.eligibility} tDeltaW=${c.totalDeltaWeight}`
            );
          });
          n.connections.out.forEach((c, cidx) => {
            console.log(
              `Iter ${iteration}: Node[${idx}] OutConn[${cidx}] to Node ${c.to.index} weight=${c.weight} elig=${c.eligibility} tDeltaW=${c.totalDeltaWeight}`
            );
          });
        });
      },
    },
  });

  // Step 3: Log the network's connection weights after training to see the changes.
  console.log(
    'Winner weights after training:',
    winner.connections.map((c) => c.weight)
  );

  // Step 4: Test the network's response to the primary input patterns after training.
  // This helps to verify that the network has learned the correct associations.
  console.log(
    'Test output for North-only scenario [0,1,0,0,0,0.7]:',
    winner.activate([0, 1, 0, 0, 0, 0.7])
  );
  console.log(
    'Test output for East-only scenario [0.25,0,1,0,0,0.7]:',
    winner.activate([0.25, 0, 1, 0, 0, 0.7])
  );
  console.log(
    'Test output for South-only scenario [0.5,0,0,1,0,0.7]:',
    winner.activate([0.5, 0, 0, 1, 0, 0.7])
  );
  console.log(
    'Test output for West-only scenario [0.75,0,0,0,1,0.7]:',
    winner.activate([0.75, 0, 0, 0, 1, 0.7])
  );

  // Step 5: Pause execution for manual inspection in a debugging environment.
  debugger;
}

describe('ASCII Maze Solver using Neuro-Evolution', () => {
  /**
   * This `beforeAll` block runs once before any of the tests in this suite.
   * It sets up the environment for clear and informative logging.
   */
  beforeAll(() => {
    // Override the global console.log function to ensure output is always visible,
    // even in test environments that might otherwise suppress it.
    console.log = jest.fn((...args) => {
      // A small filter to avoid logging specific debug messages from the dashboard.
      if (args.some((a) => typeof a === 'string' && a.includes('[DPDBG-')))
        return;
      // Write directly to standard output.
      process.stdout.write(args.join(' ') + '\n');
    });

    // Store the original console.debug implementation.
    const origDebug = console.debug;
    // Override console.debug to filter out specific dashboard messages.
    console.debug = (...args: any[]) => {
      if (args.some((a) => typeof a === 'string' && a.includes('[DPDBG-')))
        return;
      // Call the original debug function if it exists.
      origDebug?.(...args);
    };

    // Print a header with instructions for the user on how to see detailed logs.
    console.log('\n');
    console.log('═══════════════════════════════════════════════════');
    console.log(' To see all ASCII maze visualization and logs, run');
    console.log(`  ${colors.cyan}npm run test:e2e:logs${colors.reset}`);
    console.log('═══════════════════════════════════════════════════');
    console.log('\n');
  });

  // --- Main Test Case ---
  it('Evolve agent: curriculum learning in multiple steps from a tiny maze to the minotaur maze', async () => {
    // Educational Note on Curriculum Learning:
    // This test implements a strategy called "curriculum learning," where the agent is trained
    // on a sequence of increasingly difficult tasks (mazes). It starts with a very simple maze
    // and gradually moves to more complex ones. The winning network from one maze is used as the
    // starting point for the next, allowing the agent to build upon its knowledge. This is often
    // more effective than trying to solve the most difficult task from scratch.

    // --- Phase 1: Tiny Maze ---
    // The simplest maze to establish a baseline behavior.
    const tinyResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: tiny },
      agentSimConfig: { maxSteps: 100 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 40,
        maxStagnantGenerations: 200,
        minProgressToPass: 99,
        maxGenerations: 200,
        lamarckianIterations: 5,
        lamarckianSampleSize: 16,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'tiny',
      },
    });

    // Refine the winner from the tiny maze using backpropagation.
    const tinyRefined = refineWinnerWithBackprop(
      tinyResult?.bestNetwork as Network
    );

    // --- Phase 2: Small Spiral Maze ---
    // Introduces the challenge of a spiral, requiring the agent to make a long sequence of turns.
    const spiralSmallResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: spiralSmall },
      agentSimConfig: { maxSteps: 100 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 40,
        maxStagnantGenerations: 200,
        minProgressToPass: 99,
        maxGenerations: 200,
        lamarckianIterations: 5,
        lamarckianSampleSize: 16,
        initialBestNetwork: tinyRefined, // Start with the refined network from the previous phase.
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'spiralSmall',
      },
    });
    const spiralSmallRefined = refineWinnerWithBackprop(
      spiralSmallResult?.bestNetwork as Network
    );

    // --- Phase 3: Spiral Maze ---
    // A larger spiral to test the agent's ability to generalize its turning behavior.
    const spiralResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: spiral },
      agentSimConfig: { maxSteps: 150 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 40,
        maxStagnantGenerations: 200,
        minProgressToPass: 99,
        maxGenerations: 300,
        lamarckianIterations: 5,
        lamarckianSampleSize: 16,
        initialBestNetwork: spiralSmallRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'spiral',
      },
    });
    const spiralRefined = refineWinnerWithBackprop(
      spiralResult.bestNetwork as Network
    );

    // --- Phase 4: Small Maze ---
    // A more traditional maze with branches and dead ends.
    const smallResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: small },
      agentSimConfig: { maxSteps: 50 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 40,
        maxStagnantGenerations: 200,
        minProgressToPass: 99,
        maxGenerations: 300,
        lamarckianIterations: 5,
        lamarckianSampleSize: 16,
        initialBestNetwork: spiralRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'small',
      },
    });
    const smallRefined = refineWinnerWithBackprop(
      smallResult?.bestNetwork as Network
    );

    // --- Phase 5: Medium Maze ---
    // Increases the size and complexity, requiring more steps and better navigation.
    const mediumResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: medium },
      agentSimConfig: { maxSteps: 250 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 40,
        maxStagnantGenerations: 200,
        minProgressToPass: 99,
        maxGenerations: 400,
        lamarckianIterations: 5,
        lamarckianSampleSize: 16,
        initialBestNetwork: smallRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'medium',
      },
    });
    const mediumRefined = refineWinnerWithBackprop(
      mediumResult?.bestNetwork as Network
    );

    // --- Phase 6: Medium 2 Maze ---
    // A different medium-sized maze to test generalization.
    const medium2Result = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: medium2 },
      agentSimConfig: { maxSteps: 300 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 40,
        maxStagnantGenerations: 200,
        minProgressToPass: 99,
        maxGenerations: 400,
        lamarckianIterations: 5,
        lamarckianSampleSize: 16,
        initialBestNetwork: mediumRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'medium2',
      },
    });
    const medium2Refined = refineWinnerWithBackprop(
      medium2Result?.bestNetwork as Network
    );

    // --- Phase 7: Large Maze ---
    // A significant step up in difficulty, with a much larger search space.
    const largeResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: large },
      agentSimConfig: { maxSteps: 400 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 40,
        maxStagnantGenerations: 200,
        minProgressToPass: 99,
        maxGenerations: 500,
        lamarckianIterations: 5,
        lamarckianSampleSize: 16,
        initialBestNetwork: medium2Refined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'large',
      },
    });
    const largeRefined = refineWinnerWithBackprop(
      largeResult?.bestNetwork as Network
    );

    // --- Final Phase: Minotaur Maze ---
    // The most complex maze, requiring the agent to have learned a robust and general navigation strategy.
    await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: minotaur },
      agentSimConfig: { maxSteps: 700 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 40,
        maxStagnantGenerations: 200,
        minProgressToPass: 99,
        maxGenerations: 600,
        lamarckianIterations: 5,
        lamarckianSampleSize: 16,
        initialBestNetwork: largeRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'minotaur',
      },
    });
  }, 0);
});
