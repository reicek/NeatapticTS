import { formatEvolveXorExampleResult, runEvolveXorExample } from './index';

describe('evolveXor example', () => {
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
