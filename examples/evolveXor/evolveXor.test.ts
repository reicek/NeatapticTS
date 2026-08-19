import { formatEvolveXorExampleResult, runEvolveXorExample } from './index';

/**
 * Deterministic Math.random replacement used during the XOR evolution test.
 *
 * The NEAT controller seeds its own xorshift RNG, but several code paths
 * (connection/node constructors) fall back to `Math.random()`.  When the
 * full test suite runs in a shared Jest worker, `Math.random` state is
 * polluted by hundreds of prior test files, shifting those fallback calls
 * and making the evolution trajectory non-deterministic.  Replacing
 * `Math.random` with a seeded LCG for the duration of this test ensures
 * the evolution is fully reproducible regardless of execution context.
 */
const MATH_RANDOM_SEED = 42;
const originalMathRandom = Math.random;

function createSeededRandom(seed: number): () => number {
  let state = seed % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

describe('evolveXor example', () => {
  beforeAll(() => {
    Math.random = createSeededRandom(MATH_RANDOM_SEED);
  });

  afterAll(() => {
    Math.random = originalMathRandom;
  });

  it('runs one seeded XOR evolution walkthrough that reaches a solved starter-example state', async () => {
    // Arrange
    const evolveXorExampleResult = await runEvolveXorExample();

    // Act
    const formattedSummary = formatEvolveXorExampleResult(
      evolveXorExampleResult,
    );

    // Assert
    expect({
      finalGenerationWithinBudget:
        evolveXorExampleResult.finalGeneration >= 1 &&
        evolveXorExampleResult.finalGeneration <= 100,
      hasHeader: formattedSummary.startsWith(
        'Evolve XOR\nGeneration budget: 100\nFinal generation: ',
      ),
      hasPredictionSection: formattedSummary.includes('\nPredictions:\n'),
      predictionCount: evolveXorExampleResult.predictions.length,
      predictionExpectedOutputs: evolveXorExampleResult.predictions.map(
        (predictionSummary) => predictionSummary.expectedOutput,
      ),
      predictionInputs: evolveXorExampleResult.predictions.map(
        (predictionSummary) => predictionSummary.inputValues.join(','),
      ),
      predictionsMatchSolvedThresholds:
        evolveXorExampleResult.predictions.every((predictionSummary) =>
          predictionSummary.expectedOutput === 1
            ? predictionSummary.outputValue >= 0.75
            : predictionSummary.outputValue <= 0.25,
        ),
      scoreInSolvedRange:
        evolveXorExampleResult.bestScore >= 3.9 &&
        evolveXorExampleResult.bestScore <= 4.1,
      solved: evolveXorExampleResult.solved,
    }).toEqual({
      finalGenerationWithinBudget: true,
      hasHeader: true,
      hasPredictionSection: true,
      predictionCount: 4,
      predictionExpectedOutputs: [0, 1, 1, 0],
      predictionInputs: ['0,0', '0,1', '1,0', '1,1'],
      predictionsMatchSolvedThresholds: true,
      scoreInSolvedRange: true,
      solved: true,
    });
  });
});
