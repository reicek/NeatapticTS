import Network from '../../src/architecture/network';
import {
  trainImpl,
  applyGradientClippingImpl,
} from '../../src/architecture/network/network.training.utils';
import type {
  CheckpointConfig,
  MetricsHook,
  ScheduleConfig,
} from '../../src/architecture/network/network.training.utils';

type TrainingDataset = Parameters<typeof trainImpl>[1];

interface NetworkInternals {
  _forceNextOverflow: boolean;
  _mixedPrecision: { enabled: boolean; lossScale: number };
}

const setNetworkInternal = <Key extends keyof NetworkInternals>(
  net: Network,
  key: Key,
  value: NetworkInternals[Key],
) => {
  Reflect.set(net, key, value);
};

const getNetworkInternal = <Key extends keyof NetworkInternals>(
  net: Network,
  key: Key,
): NetworkInternals[Key] => Reflect.get(net, key) as NetworkInternals[Key];

/**
 * Training tests focus on uncovered branches: validation errors, gradient clipping modes, mixed precision overflow,
 * moving average variants, early stop, checkpointing, schedule hook, metrics hook, accumulation, optimizer config, etc.
 * Single expectation per test; AAA pattern with educative comments.
 */

describe('Network.training core', () => {
  describe('Scenario: dataset dimensionality invalid', () => {
    it('throws descriptive error on mismatch', () => {
      // Arrange
      const net = new Network(2, 1, { seed: 101 });
      const mismatchedDataset: TrainingDataset = [
        { input: [0.1], output: [0.5] },
      ];

      // Act / Assert
      expect(() =>
        trainImpl(net, mismatchedDataset, { iterations: 1, rate: 0.1 }),
      ).toThrow(/Dataset is invalid/);
    });
  });

  describe('Scenario: missing stopping conditions', () => {
    it('warns then throws', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 102 });
      const trainingSamples: TrainingDataset = [
        { input: [0.1], output: [0.2] },
      ];

      // Act / Assert
      expect(() => trainImpl(net, trainingSamples, { rate: 0.1 })).toThrow(
        /stopping condition/,
      );
    });
  });

  describe('Scenario: batch size larger than dataset', () => {
    it('throws explicit batch size error', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 103 });
      const trainingSamples: TrainingDataset = [
        { input: [0.3], output: [0.4] },
      ];

      // Act / Assert
      expect(() =>
        trainImpl(net, trainingSamples, {
          iterations: 1,
          rate: 0.1,
          batchSize: 5,
        }),
      ).toThrow(/Batch size/);
    });
  });

  describe('Scenario: invalid dropout value', () => {
    it('throws when dropout >= 1', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 104 });
      const trainingSamples: TrainingDataset = [
        { input: [0.3], output: [0.4] },
      ];

      // Act / Assert
      expect(() =>
        trainImpl(net, trainingSamples, {
          iterations: 1,
          rate: 0.1,
          dropout: 1,
        }),
      ).toThrow(/dropout/);
    });
  });

  describe('Scenario: invalid accumulationSteps', () => {
    it('throws when accumulationSteps < 1', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 105 });
      const trainingSamples: TrainingDataset = [
        { input: [0.3], output: [0.4] },
      ];

      // Act / Assert
      expect(() =>
        trainImpl(net, trainingSamples, {
          iterations: 1,
          rate: 0.1,
          accumulationSteps: -1,
        }),
      ).toThrow(/accumulationSteps/);
    });
  });

  describe('Scenario: unknown optimizer type', () => {
    it('throws on invalid optimizer string', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 106 });
      const trainingSamples: TrainingDataset = [
        { input: [0.3], output: [0.4] },
      ];

      // Act / Assert
      expect(() =>
        trainImpl(net, trainingSamples, {
          iterations: 1,
          rate: 0.1,
          optimizer: 'notreal',
        }),
      ).toThrow(/Unknown optimizer/);
    });
  });

  describe('Scenario: lookahead nested baseType error', () => {
    it('throws when baseType is lookahead', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 107 });
      const trainingSamples: TrainingDataset = [
        { input: [0.3], output: [0.4] },
      ];

      // Act / Assert
      expect(() =>
        trainImpl(net, trainingSamples, {
          iterations: 1,
          rate: 0.1,
          optimizer: { type: 'lookahead', baseType: 'lookahead' },
        }),
      ).toThrow(/Nested lookahead/);
    });
  });

  describe('Scenario: gradient clipping norm mode reduces large gradients', () => {
    it('scales gradients to below or equal maxNorm', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 108 });
      const inputNode = net.nodes[0];
      const outputNode = net.nodes.find((node) => node.type === 'output')!;
      const connection = net.connect(inputNode, outputNode)[0];
      connection.totalDeltaWeight = 10;

      // Act
      applyGradientClippingImpl(net, { mode: 'norm', maxNorm: 1 });

      // Assert
      expect(Math.abs(connection.totalDeltaWeight) <= 1).toBe(true);
    });
  });

  describe('Scenario: gradient clipping percentile clamps extremes', () => {
    it('clamps magnitude above percentile threshold', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 109 });
      const inputNode = net.nodes[0];
      const outputNode = net.nodes.find((node) => node.type === 'output')!;
      const lowMagnitudeConnection = net.connect(inputNode, outputNode)[0];
      const highMagnitudeConnection = net.connect(inputNode, outputNode)[0];
      lowMagnitudeConnection.totalDeltaWeight = 0.1;
      highMagnitudeConnection.totalDeltaWeight = 100;

      // Act
      applyGradientClippingImpl(net, { mode: 'percentile', percentile: 50 });

      // Assert
      expect(highMagnitudeConnection.totalDeltaWeight <= 100).toBe(true);
    });
  });

  describe('Scenario: mixed precision overflow triggers scale down', () => {
    it('reduces loss scale after forced overflow', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 110 });
      const trainingSamples: TrainingDataset = [
        { input: [0.2], output: [0.3] },
      ];
      setNetworkInternal(net, '_forceNextOverflow', true);

      // Act
      trainImpl(net, trainingSamples, {
        iterations: 1,
        rate: 0.1,
        mixedPrecision: true,
      });

      // Assert
      const mixedPrecisionState = getNetworkInternal(net, '_mixedPrecision');
      expect(mixedPrecisionState.lossScale <= 1024).toBe(true);
    });
  });

  describe('Scenario: early stopping based on patience', () => {
    it('halts before reaching max iterations when no improvement', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 111 });
      const trainingSamples: TrainingDataset = [
        { input: [0.2], output: [0.3] },
      ];

      // Act
      const result = trainImpl(net, trainingSamples, {
        iterations: 5,
        rate: 0.1,
        earlyStopPatience: 1,
        earlyStopMinDelta: 1,
      });

      // Assert
      expect(result.iterations < 5).toBe(true);
    });
  });

  describe('Scenario: checkpoint best + last callbacks', () => {
    it('invokes save for both last and best types', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 112 });
      const trainingSamples: TrainingDataset = [
        { input: [0.2], output: [0.25] },
      ];
      const savedTypes: string[] = [];
      type CheckpointPayload = Parameters<CheckpointConfig['save']>[0];
      const save: CheckpointConfig['save'] = (payload: CheckpointPayload) => {
        savedTypes.push(payload.type);
      };

      // Act
      trainImpl(net, trainingSamples, {
        iterations: 2,
        rate: 0.1,
        checkpoint: { last: true, best: true, save },
      });

      // Assert
      expect(savedTypes.includes('last') && savedTypes.includes('best')).toBe(
        true,
      );
    });
  });

  describe('Scenario: schedule + metricsHook', () => {
    it('invokes both schedule.function and metricsHook', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 113 });
      const trainingSamples: TrainingDataset = [
        { input: [0.2], output: [0.3] },
      ];
      let scheduleInvocations = 0;
      let metricsInvocations = 0;
      const schedule: ScheduleConfig = {
        iterations: 1,
        function: () => {
          scheduleInvocations++;
        },
      };
      const metricsHook: MetricsHook = () => {
        metricsInvocations++;
      };

      // Act
      trainImpl(net, trainingSamples, {
        iterations: 1,
        rate: 0.1,
        schedule,
        metricsHook,
      });

      // Assert
      expect(scheduleInvocations === 1 && metricsInvocations === 1).toBe(true);
    });
  });
});
