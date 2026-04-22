import type Network from '../network';
import type { PruningMethod } from '../network.types';
import {
  clearStochasticDepthSchedule,
  clearWeightNoiseSchedule,
  configurePruning,
  disableStochasticDepth,
  disableWeightNoise,
  enableWeightNoise,
  getLastSkippedLayers,
  getRuntimeRegularizationStats,
  getTrainingStep,
  setRandom,
  setStochasticDepth,
  setStochasticDepthSchedule,
  setWeightNoiseSchedule,
  testForceOverflow,
} from './network.runtime.controls.utils';

type RuntimeNetworkState = {
  connections: Array<{ id: number }>;
  layers?: Array<{ nodes: unknown[] }>;
  _forceNextOverflow?: boolean;
  _initialConnectionCount?: number;
  _lastSkippedLayers?: number[];
  _lastStats?: Record<string, unknown>;
  _pruningConfig?: {
    start: number;
    end: number;
    targetSparsity: number;
    regrowFraction?: number;
    frequency?: number;
    method?: PruningMethod;
    lastPruneIter?: number;
  };
  _rand: () => number;
  _stochasticDepth: number[];
  _stochasticDepthSchedule?: (step: number, current: number[]) => number[];
  _trainingStep: number;
  _weightNoisePerHidden: number[];
  _weightNoiseSchedule?: (step: number) => number;
  _weightNoiseStd: number;
};

describe('network runtime controls utility chapter', () => {
  describe('configurePruning()', () => {
    describe('when the pruning window is invalid', () => {
      it('throws the shared pruning window error message', () => {
        // Arrange
        const network = createRuntimeNetwork();
        const runConfiguration = () =>
          configurePruning.call(asNetwork(network), {
            start: 3,
            end: 2,
            targetSparsity: 0.5,
          });

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'Invalid pruning schedule window',
          name: 'NetworkRuntimePruningScheduleWindowError',
        });
      });
    });

    describe('when target sparsity falls outside the open interval', () => {
      it('throws the shared target sparsity range error message', () => {
        // Arrange
        const network = createRuntimeNetwork();
        const runConfiguration = () =>
          configurePruning.call(asNetwork(network), {
            start: 1,
            end: 3,
            targetSparsity: 1,
          });

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'targetSparsity must be in (0,1)',
          name: 'NetworkRuntimeTargetSparsityRangeError',
        });
      });
    });

    describe('when optional pruning fields are omitted', () => {
      it('stores the default pruning configuration and captures the baseline connection count', () => {
        // Arrange
        const network = createRuntimeNetwork({ connectionCount: 6 });

        // Act
        configurePruning.call(asNetwork(network), {
          start: 2,
          end: 5,
          targetSparsity: 0.4,
        });

        // Assert
        expect({
          baselineConnections: network._initialConnectionCount,
          pruningConfig: network._pruningConfig,
        }).toEqual({
          baselineConnections: 6,
          pruningConfig: {
            end: 5,
            frequency: 1,
            lastPruneIter: undefined,
            method: 'magnitude',
            regrowFraction: 0,
            start: 2,
            targetSparsity: 0.4,
          },
        });
      });
    });

    describe('when every pruning field is provided explicitly', () => {
      it('stores the provided pruning configuration values', () => {
        // Arrange
        const network = createRuntimeNetwork({ connectionCount: 4 });

        // Act
        configurePruning.call(asNetwork(network), {
          start: 1,
          end: 8,
          targetSparsity: 0.6,
          regrowFraction: 0.25,
          frequency: 3,
          method: 'snip',
        });

        // Assert
        expect(network._pruningConfig).toEqual({
          end: 8,
          frequency: 3,
          lastPruneIter: undefined,
          method: 'snip',
          regrowFraction: 0.25,
          start: 1,
          targetSparsity: 0.6,
        });
      });
    });
  });

  describe('enableWeightNoise()', () => {
    describe('when one global standard deviation is negative', () => {
      it('throws the shared global standard-deviation range error', () => {
        // Arrange
        const network = createRuntimeNetwork();
        const runConfiguration = () =>
          enableWeightNoise.call(asNetwork(network), -0.1);

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'Weight noise stdDev must be >= 0',
          name: 'NetworkRuntimeWeightNoiseStdDevRangeError',
        });
      });
    });

    describe('when one global standard deviation is valid', () => {
      it('stores the global value and clears per-hidden-layer state', () => {
        // Arrange
        const network = createRuntimeNetwork({ weightNoisePerHidden: [0.2] });

        // Act
        enableWeightNoise.call(asNetwork(network), 0.35);

        // Assert
        expect({
          perHidden: network._weightNoisePerHidden,
          stdDev: network._weightNoiseStd,
        }).toEqual({
          perHidden: [],
          stdDev: 0.35,
        });
      });
    });

    describe('when the configuration shape is invalid', () => {
      it('throws the shared configuration-shape error', () => {
        // Arrange
        const network = createRuntimeNetwork();
        const runConfiguration = () =>
          enableWeightNoise.call(asNetwork(network), {} as unknown as number);

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'Invalid weight noise configuration',
          name: 'NetworkRuntimeWeightNoiseConfigurationError',
        });
      });
    });

    describe('when per-hidden-layer noise is requested on a non-layered network', () => {
      it('throws the shared layered-network requirement error', () => {
        // Arrange
        const network = createRuntimeNetwork({ layers: [] });
        const runConfiguration = () =>
          enableWeightNoise.call(asNetwork(network), { perHiddenLayer: [0.1] });

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Per-hidden-layer weight noise requires a layered network with at least one hidden layer',
          name: 'NetworkRuntimeLayeredWeightNoiseRequiredError',
        });
      });
    });

    describe('when the per-hidden-layer count does not match the hidden layers', () => {
      it('throws the shared entry-count error', () => {
        // Arrange
        const network = createRuntimeNetwork({ hiddenLayerCount: 2 });
        const runConfiguration = () =>
          enableWeightNoise.call(asNetwork(network), { perHiddenLayer: [0.1] });

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'Expected 2 std dev entries (one per hidden layer), got 1',
          name: 'NetworkRuntimeWeightNoiseEntryCountError',
        });
      });
    });

    describe('when one per-hidden-layer value is negative', () => {
      it('throws the shared per-layer range error', () => {
        // Arrange
        const network = createRuntimeNetwork({ hiddenLayerCount: 2 });
        const runConfiguration = () =>
          enableWeightNoise.call(asNetwork(network), {
            perHiddenLayer: [0.1, -0.2],
          });

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'Weight noise std devs must be >= 0',
          name: 'NetworkRuntimeWeightNoisePerLayerRangeError',
        });
      });
    });

    describe('when per-hidden-layer noise is valid', () => {
      it('stores the per-layer values and clears the global standard deviation', () => {
        // Arrange
        const network = createRuntimeNetwork({
          hiddenLayerCount: 2,
          weightNoiseStd: 0.5,
        });

        // Act
        enableWeightNoise.call(asNetwork(network), {
          perHiddenLayer: [0.05, 0.1],
        });

        // Assert
        expect({
          perHidden: network._weightNoisePerHidden,
          stdDev: network._weightNoiseStd,
        }).toEqual({
          perHidden: [0.05, 0.1],
          stdDev: 0,
        });
      });
    });
  });

  describe('disableWeightNoise()', () => {
    describe('when weight noise was configured earlier', () => {
      it('clears both the global and per-layer weight-noise state', () => {
        // Arrange
        const network = createRuntimeNetwork({
          weightNoiseStd: 0.4,
          weightNoisePerHidden: [0.2, 0.3],
        });

        // Act
        disableWeightNoise.call(asNetwork(network));

        // Assert
        expect({
          perHidden: network._weightNoisePerHidden,
          stdDev: network._weightNoiseStd,
        }).toEqual({
          perHidden: [],
          stdDev: 0,
        });
      });
    });
  });

  describe('weight-noise scheduling helpers', () => {
    describe('when a weight-noise schedule is installed', () => {
      it('stores the provided scheduler function', () => {
        // Arrange
        const network = createRuntimeNetwork();
        const schedule = (step: number) => step * 0.01;

        // Act
        setWeightNoiseSchedule.call(asNetwork(network), schedule);

        // Assert
        expect(network._weightNoiseSchedule).toBe(schedule);
      });
    });

    describe('when a weight-noise schedule is cleared', () => {
      it('removes the stored scheduler function', () => {
        // Arrange
        const network = createRuntimeNetwork();
        network._weightNoiseSchedule = (step: number) => step * 0.02;

        // Act
        clearWeightNoiseSchedule.call(asNetwork(network));

        // Assert
        expect(network._weightNoiseSchedule).toBeUndefined();
      });
    });
  });

  describe('randomness and overflow helpers', () => {
    describe('when a deterministic random function is installed', () => {
      it('stores the provided random function', () => {
        // Arrange
        const network = createRuntimeNetwork();
        const randomFunction = () => 0.75;

        // Act
        setRandom.call(asNetwork(network), randomFunction);

        // Assert
        expect(network._rand).toBe(randomFunction);
      });
    });

    describe('when the next overflow path is forced', () => {
      it('marks the runtime overflow flag', () => {
        // Arrange
        const network = createRuntimeNetwork();

        // Act
        testForceOverflow.call(asNetwork(network));

        // Assert
        expect(network._forceNextOverflow).toBe(true);
      });
    });

    describe('when the training step is read', () => {
      it('returns the current training-step counter', () => {
        // Arrange
        const network = createRuntimeNetwork({ trainingStep: 42 });

        // Act
        const trainingStep = getTrainingStep.call(asNetwork(network));

        // Assert
        expect(trainingStep).toBe(42);
      });
    });
  });

  describe('getLastSkippedLayers()', () => {
    describe('when no skipped-layer snapshot exists yet', () => {
      it('returns an empty array', () => {
        // Arrange
        const network = createRuntimeNetwork({ lastSkippedLayers: undefined });

        // Act
        const skippedLayers = getLastSkippedLayers.call(asNetwork(network));

        // Assert
        expect(skippedLayers).toEqual([]);
      });
    });

    describe('when a skipped-layer snapshot exists', () => {
      it('returns the stored skipped-layer indices', () => {
        // Arrange
        const network = createRuntimeNetwork({ lastSkippedLayers: [1, 3] });

        // Act
        const skippedLayers = getLastSkippedLayers.call(asNetwork(network));

        // Assert
        expect(skippedLayers).toEqual([1, 3]);
      });
    });
  });

  describe('stochastic-depth scheduling helpers', () => {
    describe('when a stochastic-depth schedule is installed', () => {
      it('stores the provided schedule function', () => {
        // Arrange
        const network = createRuntimeNetwork();
        const schedule = (step: number, current: number[]) =>
          current.map((value) => value + step * 0.01);

        // Act
        setStochasticDepthSchedule.call(asNetwork(network), schedule);

        // Assert
        expect(network._stochasticDepthSchedule).toBe(schedule);
      });
    });

    describe('when a stochastic-depth schedule is cleared', () => {
      it('removes the stored schedule function', () => {
        // Arrange
        const network = createRuntimeNetwork();
        network._stochasticDepthSchedule = (step: number, current: number[]) =>
          current.slice(0, step);

        // Act
        clearStochasticDepthSchedule.call(asNetwork(network));

        // Assert
        expect(network._stochasticDepthSchedule).toBeUndefined();
      });
    });
  });

  describe('getRuntimeRegularizationStats()', () => {
    describe('when no stats snapshot has been recorded yet', () => {
      it('returns null', () => {
        // Arrange
        const network = createRuntimeNetwork({ lastStats: undefined });

        // Act
        const statsSnapshot = getRuntimeRegularizationStats.call(
          asNetwork(network),
        );

        // Assert
        expect(statsSnapshot).toBeNull();
      });
    });

    describe('when a stats snapshot exists', () => {
      it('returns a deep-cloned copy of the stats payload', () => {
        // Arrange
        const network = createRuntimeNetwork({
          lastStats: { dropoutApplied: 0.2, nested: { weightNoiseStd: 0.1 } },
        });

        // Act
        const statsSnapshot = getRuntimeRegularizationStats.call(
          asNetwork(network),
        ) as {
          dropoutApplied: number;
          nested: { weightNoiseStd: number };
        };

        // Assert
        expect({
          nestedSameReference: statsSnapshot.nested === network._lastStats?.nested,
          statsSnapshot,
        }).toEqual({
          nestedSameReference: false,
          statsSnapshot: {
            dropoutApplied: 0.2,
            nested: { weightNoiseStd: 0.1 },
          },
        });
      });
    });
  });

  describe('setStochasticDepth()', () => {
    describe('when survival input is not an array', () => {
      it('throws the shared survival-array error', () => {
        // Arrange
        const network = createRuntimeNetwork();
        const runConfiguration = () =>
          setStochasticDepth.call(
            asNetwork(network),
            true as unknown as number[],
          );

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'survival must be an array',
          name: 'NetworkRuntimeStochasticDepthSurvivalArrayError',
        });
      });
    });

    describe('when one survival probability falls outside the valid interval', () => {
      it('throws the shared survival-range error', () => {
        // Arrange
        const network = createRuntimeNetwork({ hiddenLayerCount: 2 });
        const runConfiguration = () =>
          setStochasticDepth.call(asNetwork(network), [1, 0]);

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'Stochastic depth survival probs must be in (0,1]',
          name: 'NetworkRuntimeStochasticDepthSurvivalRangeError',
        });
      });
    });

    describe('when stochastic depth is requested on a non-layered network', () => {
      it('throws the shared layered-network requirement error', () => {
        // Arrange
        const network = createRuntimeNetwork({ layers: [] });
        const runConfiguration = () =>
          setStochasticDepth.call(asNetwork(network), []);

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message: 'Stochastic depth requires layer-based network',
          name: 'NetworkRuntimeStochasticDepthLayeredNetworkRequiredError',
        });
      });
    });

    describe('when the survival count does not match the hidden layers', () => {
      it('throws the shared entry-count error', () => {
        // Arrange
        const network = createRuntimeNetwork({ hiddenLayerCount: 2 });
        const runConfiguration = () =>
          setStochasticDepth.call(asNetwork(network), [0.9]);

        // Act
        const errorSnapshot = captureErrorSnapshot(runConfiguration);

        // Assert
        expect(errorSnapshot).toEqual({
          message:
            'Expected 2 survival probabilities for hidden layers, got 1',
          name: 'NetworkRuntimeStochasticDepthEntryCountError',
        });
      });
    });

    describe('when survival probabilities are valid', () => {
      it('stores a copied survival schedule for the hidden layers', () => {
        // Arrange
        const network = createRuntimeNetwork({ hiddenLayerCount: 2 });

        // Act
        setStochasticDepth.call(asNetwork(network), [0.95, 0.8]);

        // Assert
        expect(network._stochasticDepth).toEqual([0.95, 0.8]);
      });
    });
  });

  describe('disableStochasticDepth()', () => {
    describe('when stochastic depth was configured earlier', () => {
      it('clears the stored survival schedule', () => {
        // Arrange
        const network = createRuntimeNetwork({ stochasticDepth: [0.9, 0.7] });

        // Act
        disableStochasticDepth.call(asNetwork(network));

        // Assert
        expect(network._stochasticDepth).toEqual([]);
      });
    });
  });
});

function createRuntimeNetwork(options: {
  connectionCount?: number;
  hiddenLayerCount?: number;
  lastSkippedLayers?: number[] | undefined;
  lastStats?: Record<string, unknown> | undefined;
  layers?: Array<{ nodes: unknown[] }> | undefined;
  stochasticDepth?: number[];
  trainingStep?: number;
  weightNoisePerHidden?: number[];
  weightNoiseStd?: number;
} = {}): RuntimeNetworkState {
  const hiddenLayerCount = options.hiddenLayerCount ?? 1;
  const defaultLayers = Array.from({ length: hiddenLayerCount + 2 }, () => ({
    nodes: [],
  }));

  return {
    connections: Array.from(
      { length: options.connectionCount ?? 3 },
      (_unusedValue, index) => ({ id: index }),
    ),
    layers: options.layers ?? defaultLayers,
    _forceNextOverflow: false,
    _lastSkippedLayers: options.lastSkippedLayers,
    _lastStats: options.lastStats,
    _rand: () => 0.5,
    _stochasticDepth: options.stochasticDepth ?? [],
    _trainingStep: options.trainingStep ?? 0,
    _weightNoisePerHidden: options.weightNoisePerHidden ?? [],
    _weightNoiseStd: options.weightNoiseStd ?? 0,
  };
}

function asNetwork(runtimeNetwork: RuntimeNetworkState): Network {
  return runtimeNetwork as unknown as Network;
}

function captureErrorSnapshot(runAction: () => unknown): {
  message: string;
  name: string;
} {
  try {
    runAction();
  } catch (error: unknown) {
    if (error instanceof Error) {
      return { message: error.message, name: error.name };
    }

    return { message: String(error), name: 'UnknownError' };
  }

  return { message: 'No error thrown', name: 'NoError' };
}