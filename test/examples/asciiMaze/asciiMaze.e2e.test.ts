import Network from '../../../src/architecture/network'; // Correct import for Network
import { MazeGenerator } from './mazes';
import { colors } from './colors';
import { DashboardManager } from './dashboardManager';
import { TerminalUtility } from './terminalUtility';
import { IDashboardManager } from './interfaces';
import { EvolutionEngine } from './evolutionEngine';
import { NetworkRefinement } from './networkRefinement';

/**
 * Forces console output for this test by writing directly to stdout/stderr.
 * This is necessary to ensure that log messages are visible during test execution,
 * especially in environments where `console.log` might be mocked or redirected.
 * @param args - The arguments to log, which will be joined into a single string.
 */
const forceLog = (...args: unknown[]): void => {
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
  forceLog,
);

jest.setTimeout(3600000); //

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
    console.debug = (...args: unknown[]) => {
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

  // Curriculum-style procedural maze evolution (mirrors browser demo):
  // Progressive dimensions 8 -> 40 (step 4). Each phase seeds next with prior best network.
  /**
   * Generate a curriculum of square maze dimensions.
   * Starts at 8 and increments by `increment` up to `max` (inclusive).
   * @param increment step between consecutive dimensions
   * @param max maximum dimension (inclusive)
   */
  const curriculumDimensions = (increment: number, max: number): number[] => {
    const seq: number[] = [];
    for (let size = 8; size <= max; size += increment) {
      seq.push(size);
    }
    return seq;
  };

  let proceduralPrevBest: Network | undefined;

  for (const dim of curriculumDimensions(4, 200)) {
    it(`Procedural maze ${dim}x${dim}`, async () => {
      const result = await EvolutionEngine.runMazeEvolution({
        mazeConfig: { maze: new MazeGenerator(dim, dim).generate() },
        agentSimConfig: { maxSteps: 2000 }, // matches browser AGENT_MAX_STEPS
        evolutionAlgorithmConfig: {
          allowRecurrent: true,
          popSize: 40,
          autoPauseOnSolve: false,
          maxStagnantGenerations: 50,
          minProgressToPass: 95,
          // hard cap per phase (browser DEFAULT_MAX_GENERATIONS)
          maxGenerations: 100,
          lamarckianIterations: 4,
          lamarckianSampleSize: 12,
          initialBestNetwork: proceduralPrevBest, // seed from previous phase (undefined for first)
        },
        reportingConfig: {
          dashboardManager: dashboardManagerInstance,
          logEvery: 1, // per-generation logging like browser PER_GENERATION_LOG_FREQUENCY
          label: `procedural-curriculum-${dim}x${dim}`,
        },
      });
      proceduralPrevBest = result
        ? NetworkRefinement.refineWinnerWithBackprop(
            result.bestNetwork as Network,
          )
        : undefined;

      expect(!!result?.bestNetwork).toBe(true);
    });
  }
});
