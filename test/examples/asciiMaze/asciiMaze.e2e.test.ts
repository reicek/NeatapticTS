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
  minotaur
} from './mazes';
import { colors } from './colors';
import {
  DashboardManager
} from './dashboardManager';
import { TerminalUtility } from './terminalUtility';
import {
  IDashboardManager,
} from './interfaces';
import { EvolutionEngine } from './evolutionEngine';

// Force console output for this test by writing directly to stdout/stderr
const forceLog = (...args: any[]): void => {
  // Write log messages directly to stdout to ensure visibility during test runs
  const message = args.join(' ') + '\n';
  process.stdout.write(message);
};

// Create the dashboard manager for visualizing and logging progress
const dashboardManagerInstance: IDashboardManager = new DashboardManager(TerminalUtility.createTerminalClearer(), forceLog);

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

  // Ensure all nodes have a valid squash (activation) function before training.
  // This is necessary for proper backpropagation.
  winner.nodes.forEach(n => {
    if (typeof n.squash !== 'function') {
      n.squash = methods.Activation.LOGISTIC;
    }
  });

  // Expanded training set for better learning.
  // Each entry represents a possible sensory input and the ideal output action.
  const trainingSet = [
    // Idealized inputs and outputs for the agent's actions
    { input: [0, 1, 0, 0, 0], output: [1, 0, 0, 0] },       // North
    { input: [0.25, 0, 1, 0, 0], output: [0, 1, 0, 0] },    // East
    { input: [0.5, 0, 0, 1, 0], output: [0, 0, 1, 0] },     // South
    { input: [0.75, 0, 0, 0, 1], output: [0, 0, 0, 1] },    // West

    // Ambiguous cases where multiple directions are open; the best direction is chosen.
    { input: [0, 1, 1, 0, 0], output: [1, 0, 0, 0] },       // N and E open, best is N
    { input: [0, 1, 0, 1, 0], output: [1, 0, 0, 0] },
    { input: [0, 1, 0, 0, 1], output: [1, 0, 0, 0] },
    { input: [0, 1, 0, 1, 1], output: [1, 0, 0, 0] },       // North
    { input: [0, 1, 1, 0, 1], output: [1, 0, 0, 0] },       // North
    { input: [0, 1, 1, 1, 0], output: [1, 0, 0, 0] },       // North
    { input: [0, 1, 1, 1, 1], output: [1, 0, 0, 0] },       // North

    { input: [0.25, 1, 1, 0, 0], output: [0, 1, 0, 0] },    // N and E open, best is E
    { input: [0.25, 0, 1, 1, 0], output: [0, 1, 0, 0] },
    { input: [0.25, 0, 1, 0, 1], output: [0, 1, 0, 0] }, 
    { input: [0.25, 0, 1, 1, 1], output: [0, 1, 0, 0] },    // East
    { input: [0.25, 1, 1, 0, 1], output: [0, 1, 0, 0] },    // East
    { input: [0.25, 1, 1, 1, 0], output: [0, 1, 0, 0] },    // East
    { input: [0.25, 1, 1, 1, 1], output: [0, 1, 0, 0] },    // East

    { input: [0.5, 1, 0, 1, 0], output: [0, 0, 1, 0] },     // E and S open, best is S
    { input: [0.5, 0, 1, 1, 0], output: [0, 0, 1, 0] },  
    { input: [0.5, 0, 0, 1, 1], output: [0, 0, 1, 0] },     
    { input: [0.5, 0, 1, 1, 1], output: [0, 0, 1, 0] },     // South
    { input: [0.5, 1, 0, 1, 1], output: [0, 0, 1, 0] },     // South
    { input: [0.5, 1, 1, 1, 0], output: [0, 0, 1, 0] },     // South
    { input: [0.5, 1, 1, 1, 1], output: [0, 0, 1, 0] },     // South


    { input: [0.75, 1, 0, 0, 1], output: [0, 0, 0, 1] },    // S and W open, best is W
    { input: [0.75, 0, 1, 0, 1], output: [0, 0, 0, 1] },    
    { input: [0.75, 0, 0, 1, 1], output: [0, 0, 0, 1] }, 
    { input: [0.75, 0, 1, 1, 1], output: [0, 0, 0, 1] },    // West
    { input: [0.75, 1, 0, 1, 1], output: [0, 0, 0, 1] },    // West
    { input: [0.75, 1, 1, 0, 1], output: [0, 0, 0, 1] },    // West
    { input: [0.75, 1, 1, 1, 1], output: [0, 0, 0, 1] },    // West
  ];

  // Uncomment to analyze the winner before training (for debugging)
  // analizeWinner(winner.clone(), trainingSet);

  // Train the winner with backpropagation (champion gets the ultimate training)
  winner.train(trainingSet, {
    iterations: 30000,
    error: 0.001,
    rate: 0.0001,
    momentum: 0.2,
    batchSize: 10,
  });

  // Return a clone of the trained winner (for compatibility with rest of pipeline)
  return winner.clone();
}

/**
 * Analyzes the weights and outputs of a network before and after training.
 * Used for debugging and understanding how the network changes during training.
 *
 * @param winner - The network to analyze.
 * @param trainingSet - The training data to use for supervised learning.
 */
function analizeWinner(winner: Network, trainingSet: {input: number[], output: number[]}[]) {
  // Print weights before training
  console.log('Winner weights before training:', winner.connections.map(c => c.weight));

  winner.train(trainingSet, {
    iterations: 20000,
    recurrent: true,
    error: 0.001,
    log: 100,
    rate: 0.0001,
    batchSize: 10,
    schedule: {
      iterations: 2000,
      function: ({ iteration }: { iteration: number }) => {
        // Log connection weights and eligibility traces for each node
        winner.nodes.forEach((n, idx) => {
          n.connections.in.forEach((c, cidx) => {
            console.log(`Iter ${iteration}: Node[${idx}] InConn[${cidx}] from Node ${c.from.index} weight=${c.weight} elig=${c.eligibility} tDeltaW=${c.totalDeltaWeight}`);
          });
          n.connections.out.forEach((c, cidx) => {
            console.log(`Iter ${iteration}: Node[${idx}] OutConn[${cidx}] to Node ${c.to.index} weight=${c.weight} elig=${c.eligibility} tDeltaW=${c.totalDeltaWeight}`);
          });
        });
      }
    }
  });

  // Print weights after training
  console.log('Winner weights after training:', winner.connections.map(c => c.weight));
  // Print test outputs for key input patterns
  console.log('Test output for [0,1,0,0,0]:', winner.activate([0,1,0,0,0]));
  console.log('Test output for [0.25,0,1,0,0]:', winner.activate([0.25,0,1,0,0]));
  console.log('Test output for [0.5,0,0,1,0]:', winner.activate([0.5,0,0,1,0]));
  console.log('Test output for [0.75,0,0,0,1]:', winner.activate([0.75,0,0,0,1]));
  
  debugger;
}

describe('ASCII Maze Solver using Neuro-Evolution', () => {
  beforeAll(() => {
    // Print instructions for running with visible logs
    // Override console.log to ensure output is visible in test environments
    console.log = jest.fn((...args) => process.stdout.write(args.join(' ') + '\n'));
    console.log('\n');
    console.log('═══════════════════════════════════════════════════');
    console.log(' To see all ASCII maze visualization and logs, run');
    console.log(`  ${colors.cyan}npm run test:e2e:logs${colors.reset}`);
    console.log('═══════════════════════════════════════════════════');
    console.log('\n');
  });

  // --- TESTS ---
  it('Evolve agent: curriculum learning in multiple steps from a tiny mazy to the minotaur maze', async () => {
    // Train on tiny maze
    const tinyResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: tiny },
      agentSimConfig: { maxSteps: 100 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 10, // Lamarckian evolution allows for smaller populations
        maxStagnantGenerations: 100,
        minProgressToPass: 99,
        // initialBestNetwork: Architect.perceptron(5, 10, 10, 10, 10, 4),
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'tiny',
      }
    });
    
    // Refine the best network from tiny maze with supervised backpropagation
    const tinyRefined = refineWinnerWithBackprop(tinyResult?.bestNetwork as Network);

    // spiralSmall maze
    const spiralSmallResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: spiralSmall },
      agentSimConfig: { maxSteps: 100 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 10,
        maxStagnantGenerations: 100,
        minProgressToPass: 99,
        initialBestNetwork: tinyRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'spiralSmall',
      }
    });
    const spiralSmallRefined = refineWinnerWithBackprop(spiralSmallResult?.bestNetwork as Network);

    // Spiral maze
    const spiralResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: spiral },
      agentSimConfig: { maxSteps: 150 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 10,
        maxStagnantGenerations: 100,
        minProgressToPass: 99,
        initialBestNetwork: spiralSmallRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'spiral',
      }
    });
    const spiralRefined = refineWinnerWithBackprop(spiralResult.bestNetwork as Network);

    // Train on small maze
    const smallResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: small },
      agentSimConfig: { maxSteps: 50 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 10,
        maxStagnantGenerations: 100,
        minProgressToPass: 99,
        initialBestNetwork: spiralRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'small',
      }
    });
    const smallRefined = refineWinnerWithBackprop(smallResult?.bestNetwork as Network);

    // Medium maze
    const mediumResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: medium },
      agentSimConfig: { maxSteps: 250 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 10,
        maxStagnantGenerations: 100,
        minProgressToPass: 99,
        initialBestNetwork: smallRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'medium',
      }
    });
    const mediumRefined = refineWinnerWithBackprop(mediumResult?.bestNetwork as Network);

    // Medium 2 maze
    const medium2Result = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: medium2 },
      agentSimConfig: { maxSteps: 300 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 10,
        maxStagnantGenerations: 100,
        minProgressToPass: 99,
        initialBestNetwork: mediumRefined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'medium2',
      }
    });
    const medium2Refined = refineWinnerWithBackprop(medium2Result?.bestNetwork as Network);

    // Large maze
    const largeResult = await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: large },
      agentSimConfig: { maxSteps: 400 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 10,
        maxStagnantGenerations: 100,
        minProgressToPass: 99,
        initialBestNetwork: medium2Refined,
      },
      reportingConfig: {
        dashboardManager: dashboardManagerInstance,
        logEvery: 1,
        label: 'large',
      }
    });
    const largeRefined = refineWinnerWithBackprop(largeResult?.bestNetwork as Network);

    // Minotaur maze (final, most complex)
    await EvolutionEngine.runMazeEvolution({
      mazeConfig: { maze: minotaur },
      agentSimConfig: { maxSteps: 700 },
      evolutionAlgorithmConfig: {
        allowRecurrent: true,
        popSize: 10,
        maxStagnantGenerations: 100,
        minProgressToPass: 99,
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
