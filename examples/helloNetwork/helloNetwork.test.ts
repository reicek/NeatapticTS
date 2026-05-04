import {
  formatHelloNetworkExampleResult,
  runHelloNetworkExample,
} from './index';

describe('helloNetwork example', () => {
  it('builds one deterministic feed-forward walkthrough for the starter learning path', () => {
    // Arrange
    const helloNetworkExampleResult = runHelloNetworkExample();

    // Act
    const formattedSummary = formatHelloNetworkExampleResult(
      helloNetworkExampleResult,
    );

    // Assert
    expect({
      architecture: helloNetworkExampleResult.architecture,
      formattedSummary,
      inputValues: helloNetworkExampleResult.inputValues,
      outputValues: helloNetworkExampleResult.outputValues,
    }).toEqual({
      architecture: {
        hiddenLayerSizes: [3],
        inputCount: 2,
        outputCount: 1,
        topologyIntent: 'feed-forward',
      },
      formattedSummary:
        'Hello Network\nArchitecture: 2 -> [3] -> 1 (feed-forward)\nInput values: 0.25, 0.75\nOutput values: 0.58307',
      inputValues: [0.25, 0.75],
      outputValues: [0.58307],
    });
  });
});
