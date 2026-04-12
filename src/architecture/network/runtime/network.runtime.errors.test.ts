import {
  NetworkRuntimeDropConnectProbabilityRangeError,
  NetworkRuntimeLayeredWeightNoiseRequiredError,
  NetworkRuntimePruningScheduleWindowError,
  NetworkRuntimeStochasticDepthEntryCountError,
  NetworkRuntimeStochasticDepthLayeredNetworkRequiredError,
  NetworkRuntimeStochasticDepthSurvivalArrayError,
  NetworkRuntimeStochasticDepthSurvivalRangeError,
  NetworkRuntimeTargetSparsityRangeError,
  NetworkRuntimeWeightNoiseConfigurationError,
  NetworkRuntimeWeightNoiseEntryCountError,
  NetworkRuntimeWeightNoisePerLayerRangeError,
  NetworkRuntimeWeightNoiseStdDevRangeError,
} from './network.runtime.errors';

function captureErrorSnapshot(error: Error): {
  message: string;
  name: string;
  cause: unknown;
} {
  return {
    message: error.message,
    name: error.name,
    cause: error.cause,
  };
}

describe('network runtime errors chapter', () => {
  describe('pruning and sparsity errors', () => {
    describe('given pruning schedule input is invalid', () => {
      it('preserves the pruning schedule window error metadata', () => {
        // Arrange
        const cause = new Error('bad pruning window');

        // Act
        const error = new NetworkRuntimePruningScheduleWindowError(
          'pruning schedule window is invalid',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'pruning schedule window is invalid',
          name: 'NetworkRuntimePruningScheduleWindowError',
          cause,
        });
      });
    });

    describe('given target sparsity falls outside the open interval', () => {
      it('preserves the target sparsity error metadata', () => {
        // Arrange
        const cause = new Error('bad target sparsity');

        // Act
        const error = new NetworkRuntimeTargetSparsityRangeError(
          'target sparsity must stay inside the open interval',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'target sparsity must stay inside the open interval',
          name: 'NetworkRuntimeTargetSparsityRangeError',
          cause,
        });
      });
    });
  });

  describe('weight noise errors', () => {
    describe('given weight-noise standard deviation is negative', () => {
      it('preserves the standard deviation range error metadata', () => {
        // Arrange
        const cause = new Error('negative std dev');

        // Act
        const error = new NetworkRuntimeWeightNoiseStdDevRangeError(
          'weight-noise standard deviation must be non-negative',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'weight-noise standard deviation must be non-negative',
          name: 'NetworkRuntimeWeightNoiseStdDevRangeError',
          cause,
        });
      });
    });

    describe('given weight-noise configuration shape is invalid', () => {
      it('preserves the weight-noise configuration error metadata', () => {
        // Arrange
        const cause = new Error('invalid shape');

        // Act
        const error = new NetworkRuntimeWeightNoiseConfigurationError(
          'weight-noise configuration shape is invalid',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'weight-noise configuration shape is invalid',
          name: 'NetworkRuntimeWeightNoiseConfigurationError',
          cause,
        });
      });
    });

    describe('given per-layer weight noise is requested on a non-layered network', () => {
      it('preserves the layered weight-noise requirement metadata', () => {
        // Arrange
        const cause = new Error('missing layered network');

        // Act
        const error = new NetworkRuntimeLayeredWeightNoiseRequiredError(
          'per-layer weight noise requires a layered network',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'per-layer weight noise requires a layered network',
          name: 'NetworkRuntimeLayeredWeightNoiseRequiredError',
          cause,
        });
      });
    });

    describe('given per-layer weight-noise entries do not match hidden layers', () => {
      it('preserves the entry count error metadata', () => {
        // Arrange
        const cause = new Error('entry count mismatch');

        // Act
        const error = new NetworkRuntimeWeightNoiseEntryCountError(
          'weight-noise entries must match hidden-layer count',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'weight-noise entries must match hidden-layer count',
          name: 'NetworkRuntimeWeightNoiseEntryCountError',
          cause,
        });
      });
    });

    describe('given a per-layer weight-noise value is negative', () => {
      it('preserves the per-layer range error metadata', () => {
        // Arrange
        const cause = new Error('negative per-layer noise');

        // Act
        const error = new NetworkRuntimeWeightNoisePerLayerRangeError(
          'per-layer weight-noise values must be non-negative',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'per-layer weight-noise values must be non-negative',
          name: 'NetworkRuntimeWeightNoisePerLayerRangeError',
          cause,
        });
      });
    });
  });

  describe('stochastic depth and dropconnect errors', () => {
    describe('given stochastic-depth survival input is not an array', () => {
      it('preserves the survival array error metadata', () => {
        // Arrange
        const cause = new Error('survival not array');

        // Act
        const error = new NetworkRuntimeStochasticDepthSurvivalArrayError(
          'stochastic-depth survival input must be an array',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'stochastic-depth survival input must be an array',
          name: 'NetworkRuntimeStochasticDepthSurvivalArrayError',
          cause,
        });
      });
    });

    describe('given a stochastic-depth survival probability is out of range', () => {
      it('preserves the survival range error metadata', () => {
        // Arrange
        const cause = new Error('survival out of range');

        // Act
        const error = new NetworkRuntimeStochasticDepthSurvivalRangeError(
          'stochastic-depth survival probability is out of range',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'stochastic-depth survival probability is out of range',
          name: 'NetworkRuntimeStochasticDepthSurvivalRangeError',
          cause,
        });
      });
    });

    describe('given stochastic depth is requested on a non-layered network', () => {
      it('preserves the layered-network requirement metadata', () => {
        // Arrange
        const cause = new Error('missing layered network');

        // Act
        const error = new NetworkRuntimeStochasticDepthLayeredNetworkRequiredError(
          'stochastic depth requires a layered network',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'stochastic depth requires a layered network',
          name: 'NetworkRuntimeStochasticDepthLayeredNetworkRequiredError',
          cause,
        });
      });
    });

    describe('given stochastic-depth entries do not match hidden-layer count', () => {
      it('preserves the stochastic-depth entry count metadata', () => {
        // Arrange
        const cause = new Error('stochastic depth count mismatch');

        // Act
        const error = new NetworkRuntimeStochasticDepthEntryCountError(
          'stochastic-depth entries must match hidden-layer count',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'stochastic-depth entries must match hidden-layer count',
          name: 'NetworkRuntimeStochasticDepthEntryCountError',
          cause,
        });
      });
    });

    describe('given dropconnect probability falls outside the valid interval', () => {
      it('preserves the dropconnect probability error metadata', () => {
        // Arrange
        const cause = new Error('dropconnect out of range');

        // Act
        const error = new NetworkRuntimeDropConnectProbabilityRangeError(
          'dropconnect probability must stay inside the valid interval',
          { cause },
        );

        // Assert
        expect(captureErrorSnapshot(error)).toEqual({
          message: 'dropconnect probability must stay inside the valid interval',
          name: 'NetworkRuntimeDropConnectProbabilityRangeError',
          cause,
        });
      });
    });
  });
});