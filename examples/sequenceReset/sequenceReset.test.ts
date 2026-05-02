import {
  formatSequenceResetExampleResult,
  runSequenceResetExample,
  type SequenceResetExampleResult,
} from './index';

describe('sequenceReset example', () => {
  it('demonstrates clear() reset semantics and carryover state in a recurrent LSTM', () => {
    // Arrange
    const exampleResult = runSequenceResetExample();

    // Act
    const formattedSummary = formatSequenceResetExampleResult(exampleResult);

    // Assert
    expect({
      afterResetMatchesFreshRun: afterResetMatchesFreshRun(exampleResult),
      carryoverDiffersFreshRun: carryoverDiffersFreshRun(exampleResult),
      outputStepCount: exampleResult.stateAccumulationOutputs.length,
      topologyType: exampleResult.architecture.topologyType,
      lstmHiddenSize: exampleResult.architecture.lstmHiddenSize,
      inputCount: exampleResult.architecture.inputCount,
      outputCount: exampleResult.architecture.outputCount,
      formattedSummaryHasHeader: formattedSummary.startsWith('Sequence Reset'),
      formattedSummaryHasResetLine: formattedSummary.includes(
        'clear() restores fresh-start behavior: true',
      ),
      formattedSummaryHasCarryoverLine: formattedSummary.includes(
        'Without clear(), carryover changes outputs: true',
      ),
    }).toEqual({
      afterResetMatchesFreshRun: true,
      carryoverDiffersFreshRun: true,
      outputStepCount: 5,
      topologyType: 'unconstrained',
      lstmHiddenSize: 4,
      inputCount: 1,
      outputCount: 1,
      formattedSummaryHasHeader: true,
      formattedSummaryHasResetLine: true,
      formattedSummaryHasCarryoverLine: true,
    });
  });
});

/**
 * Returns true when every after-reset output matches the baseline exactly.
 *
 * @param result - Sequence reset example result.
 * @returns Whether `afterResetOutputs` equals `stateAccumulationOutputs`.
 */
function afterResetMatchesFreshRun(
  result: SequenceResetExampleResult,
): boolean {
  return result.afterResetOutputs.every(
    (outputValue, stepIndex) =>
      outputValue === result.stateAccumulationOutputs[stepIndex],
  );
}

/**
 * Returns true when at least one carryover output differs from the baseline.
 *
 * @param result - Sequence reset example result.
 * @returns Whether any `carryoverOutputs` differs from `stateAccumulationOutputs`.
 */
function carryoverDiffersFreshRun(result: SequenceResetExampleResult): boolean {
  return result.carryoverOutputs.some(
    (outputValue, stepIndex) =>
      outputValue !== result.stateAccumulationOutputs[stepIndex],
  );
}
