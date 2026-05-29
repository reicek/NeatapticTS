import type Network from '../../src/architecture/network/network.ts';
import { Neat, methods } from '../../src/browser-entry.ts';

const XOR_EXAMPLE_GENERATION_BUDGET = 100;
const XOR_SOLVED_SCORE = 3.9;
const XOR_EXAMPLE_ELITISM = 5;
const XOR_EXAMPLE_MUTATION_AMOUNT = 2;
const XOR_EXAMPLE_MUTATION_RATE = 0.8;
const XOR_EXAMPLE_POPULATION_SIZE = 100;
const XOR_EXAMPLE_SEED = 42;
const XOR_EXAMPLE_MUTATION_POOL = methods.mutation.FFW.slice();
const XOR_DATASET = [
  { expectedOutput: 0, inputValues: [0, 0] },
  { expectedOutput: 1, inputValues: [0, 1] },
  { expectedOutput: 1, inputValues: [1, 0] },
  { expectedOutput: 0, inputValues: [1, 1] },
] as const;

/** One prediction row emitted by the Evolve XOR starter example. */
export interface EvolveXorPredictionSummary {
  expectedOutput: number;
  inputValues: number[];
  outputValue: number;
}

/** Public summary returned by the Evolve XOR starter example. */
export interface EvolveXorExampleResult {
  bestScore: number;
  finalGeneration: number;
  predictions: EvolveXorPredictionSummary[];
  solved: boolean;
}

/**
 * Runs the smallest starter evolution example in the learning path.
 *
 * This example keeps the contract narrow on purpose: one seeded NEAT run,
 * one tiny XOR fitness function, one feed-forward mutation shelf, and one
 * final summary that shows how the champion predicts the four XOR inputs.
 *
 * @returns Structured summary of one seeded XOR evolution pass.
 *
 * @example
 * ```ts
 * import { runEvolveXorExample } from './index';
 *
 * const summary = await runEvolveXorExample();
 * console.log(summary.bestScore);
 * ```
 */
export async function runEvolveXorExample(): Promise<EvolveXorExampleResult> {
  const neat = new Neat(2, 1, scoreXorFitness, {
    elitism: XOR_EXAMPLE_ELITISM,
    fastMode: true,
    mutation: XOR_EXAMPLE_MUTATION_POOL.slice(),
    mutationAmount: XOR_EXAMPLE_MUTATION_AMOUNT,
    mutationRate: XOR_EXAMPLE_MUTATION_RATE,
    popsize: XOR_EXAMPLE_POPULATION_SIZE,
    seed: XOR_EXAMPLE_SEED,
  });

  let bestGenome: Network | undefined;

  // Step 1: Advance one small seeded run until the generation budget or solved score stops it.
  for (
    let generationIndex = 0;
    generationIndex < XOR_EXAMPLE_GENERATION_BUDGET;
    generationIndex += 1
  ) {
    await neat.evaluate();
    bestGenome = await neat.evolve();

    if ((bestGenome.score ?? Number.NEGATIVE_INFINITY) >= XOR_SOLVED_SCORE) {
      break;
    }
  }

  if (!bestGenome) {
    throw new Error(
      'Evolve XOR example expected one best genome after evolution.',
    );
  }

  // Step 2: Measure the champion against the four XOR cases.
  const predictions = XOR_DATASET.map((xorSample) => ({
    expectedOutput: xorSample.expectedOutput,
    inputValues: [...xorSample.inputValues],
    outputValue: roundOutputValue(
      activateIndependentOutput(bestGenome, xorSample.inputValues),
    ),
  }));

  // Step 3: Return one compact summary for docs, tests, and the console runner.
  return {
    bestScore: roundScore(bestGenome.score ?? Number.NEGATIVE_INFINITY),
    finalGeneration: neat.generation,
    predictions,
    solved: (bestGenome.score ?? Number.NEGATIVE_INFINITY) >= XOR_SOLVED_SCORE,
  };

  /**
   * Scores one genome against the four XOR truth-table rows.
   *
   * @param candidateNetwork - Genome runtime being scored by the controller.
   * @returns Fitness where larger values mean smaller XOR error.
   */
  function scoreXorFitness(candidateNetwork: Network): number {
    const totalAbsoluteError = XOR_DATASET.reduce(
      (currentErrorTotal, xorSample) => {
        const predictedOutput = activateIndependentOutput(
          candidateNetwork,
          xorSample.inputValues,
        );

        return (
          currentErrorTotal +
          Math.abs(xorSample.expectedOutput - predictedOutput)
        );
      },
      0,
    );

    return XOR_DATASET.length - totalAbsoluteError;
  }
}

/**
 * Activates one XOR sample from a fresh network state.
 *
 * @param network - Runtime network being evaluated.
 * @param inputValues - One XOR input row.
 * @returns Raw output value for that independent sample.
 */
function activateIndependentOutput(
  network: Network,
  inputValues: readonly number[],
): number {
  network.clear();
  return network.activate([...inputValues])[0] ?? 0;
}

/**
 * Formats the Evolve XOR summary as a short console-friendly block.
 *
 * @param exampleResult - Structured summary returned by `runEvolveXorExample`.
 * @returns Readable multi-line text for one seeded XOR pass.
 */
export function formatEvolveXorExampleResult(
  exampleResult: EvolveXorExampleResult,
): string {
  const predictionLines = exampleResult.predictions.map(
    (predictionSummary) =>
      `${predictionSummary.inputValues.join(', ')} -> ${predictionSummary.outputValue} expected ${predictionSummary.expectedOutput}`,
  );

  return [
    'Evolve XOR',
    `Generation budget: ${XOR_EXAMPLE_GENERATION_BUDGET}`,
    `Final generation: ${exampleResult.finalGeneration}`,
    `Best score: ${exampleResult.bestScore}`,
    `Solved: ${exampleResult.solved ? 'yes' : 'no'}`,
    'Predictions:',
    ...predictionLines,
  ].join('\n');
}

/**
 * Rounds one predicted XOR output to a readable precision.
 *
 * @param outputValue - Raw activation output from the champion genome.
 * @returns Rounded output value.
 */
function roundOutputValue(outputValue: number): number {
  return Number(outputValue.toFixed(6));
}

/**
 * Rounds one XOR fitness score to a readable precision.
 *
 * @param scoreValue - Raw fitness score.
 * @returns Rounded score.
 */
function roundScore(scoreValue: number): number {
  return Number(scoreValue.toFixed(6));
}
