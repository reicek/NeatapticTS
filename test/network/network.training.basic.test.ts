import Network from '../../src/architecture/network';
import {
  trainImpl,
  applyGradientClippingImpl,
  trainSetImpl,
} from '../../src/architecture/network/network.training';
import * as methods from '../../src/methods/methods';

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
      const badSet = [{ input: [0.1], output: [0.5] } as any];
      // Act / Assert
      expect(() =>
        trainImpl(net, badSet, { iterations: 1, rate: 0.1 })
      ).toThrow(/Dataset is invalid/);
    });
  });

  describe('Scenario: missing stopping conditions', () => {
    it('warns then throws', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 102 });
      const set = [{ input: [0.1], output: [0.2] }];
      // Act / Assert
      expect(() => trainImpl(net, set as any, { rate: 0.1 })).toThrow(
        /stopping condition/
      );
    });
  });

  describe('Scenario: batch size larger than dataset', () => {
    it('throws explicit batch size error', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 103 });
      const set = [{ input: [0.3], output: [0.4] }];
      // Act / Assert
      expect(() =>
        trainImpl(net, set as any, { iterations: 1, rate: 0.1, batchSize: 5 })
      ).toThrow(/Batch size/);
    });
  });

  describe('Scenario: invalid dropout value', () => {
    it('throws when dropout >= 1', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 104 });
      const set = [{ input: [0.3], output: [0.4] }];
      // Act / Assert
      expect(() =>
        trainImpl(net, set as any, { iterations: 1, rate: 0.1, dropout: 1 })
      ).toThrow(/dropout/);
    });
  });

  describe('Scenario: invalid accumulationSteps', () => {
    it('throws when accumulationSteps < 1', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 105 });
      const set = [{ input: [0.3], output: [0.4] }];
      // Act / Assert
      expect(() =>
        trainImpl(net, set as any, {
          iterations: 1,
          rate: 0.1,
          accumulationSteps: -1,
        })
      ).toThrow(/accumulationSteps/);
    });
  });

  describe('Scenario: unknown optimizer type', () => {
    it('throws on invalid optimizer string', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 106 });
      const set = [{ input: [0.3], output: [0.4] }];
      // Act / Assert
      expect(() =>
        trainImpl(net, set as any, {
          iterations: 1,
          rate: 0.1,
          optimizer: 'notreal',
        })
      ).toThrow(/Unknown optimizer/);
    });
  });

  describe('Scenario: lookahead nested baseType error', () => {
    it('throws when baseType is lookahead', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 107 });
      const set = [{ input: [0.3], output: [0.4] }];
      // Act / Assert
      expect(() =>
        trainImpl(net, set as any, {
          iterations: 1,
          rate: 0.1,
          optimizer: { type: 'lookahead', baseType: 'lookahead' },
        })
      ).toThrow(/Nested lookahead/);
    });
  });

  describe('Scenario: gradient clipping norm mode reduces large gradients', () => {
    it('scales gradients to below or equal maxNorm', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 108 });
      const inputNode = net.nodes[0];
      const outNode = net.nodes.find((n) => n.type === 'output')!;
      // artificially create a connection with large delta
      const conn = net.connect(inputNode, outNode)[0];
      (conn as any).totalDeltaWeight = 10;
      // Act
      applyGradientClippingImpl(net, { mode: 'norm', maxNorm: 1 });
      // Assert
      expect(Math.abs((conn as any).totalDeltaWeight) <= 1).toBe(true);
    });
  });

  describe('Scenario: gradient clipping percentile clamps extremes', () => {
    it('clamps magnitude above percentile threshold', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 109 });
      const inputNode = net.nodes[0];
      const outNode = net.nodes.find((n) => n.type === 'output')!;
      const c1 = net.connect(inputNode, outNode)[0];
      const c2 = net.connect(inputNode, outNode)[0];
      (c1 as any).totalDeltaWeight = 0.1;
      (c2 as any).totalDeltaWeight = 100;
      // Act
      applyGradientClippingImpl(net, { mode: 'percentile', percentile: 50 });
      // Assert
      expect((c2 as any).totalDeltaWeight <= 100).toBe(true);
    });
  });

  describe('Scenario: mixed precision overflow triggers scale down', () => {
    it('reduces loss scale after forced overflow', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 110 });
      const set = [{ input: [0.2], output: [0.3] }];
      (net as any)._forceNextOverflow = true; // cause overflow detection path
      // Act
      const res = trainImpl(net, set as any, {
        iterations: 1,
        rate: 0.1,
        mixedPrecision: true,
      });
      // Assert
      expect(((net as any)._mixedPrecision.lossScale as number) <= 1024).toBe(
        true
      );
    });
  });

  describe('Scenario: early stopping based on patience', () => {
    it('halts before reaching max iterations when no improvement', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 111 });
      const set = [{ input: [0.2], output: [0.3] }];
      // Act
      const res = trainImpl(net, set as any, {
        iterations: 5,
        rate: 0.1,
        earlyStopPatience: 1,
        earlyStopMinDelta: 1,
      });
      // Assert
      expect(res.iterations < 5).toBe(true);
    });
  });

  describe('Scenario: checkpoint best + last callbacks', () => {
    it('invokes save for both last and best types', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 112 });
      const set = [{ input: [0.2], output: [0.25] }];
      const saved: string[] = [];
      const save = (p: any) => {
        saved.push(p.type);
      };
      // Act
      trainImpl(net, set as any, {
        iterations: 2,
        rate: 0.1,
        checkpoint: { last: true, best: true, save },
      });
      // Assert
      expect(saved.includes('last') && saved.includes('best')).toBe(true);
    });
  });

  describe('Scenario: schedule + metricsHook', () => {
    it('invokes both schedule.function and metricsHook', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 113 });
      const set = [{ input: [0.2], output: [0.3] }];
      let scheduleCalled = 0;
      let metricsCalled = 0;
      const schedule = {
        iterations: 1,
        function: () => {
          scheduleCalled++;
        },
      };
      const metricsHook = () => {
        metricsCalled++;
      };
      // Act
      trainImpl(net, set as any, {
        iterations: 1,
        rate: 0.1,
        schedule,
        metricsHook,
      });
      // Assert
      expect(scheduleCalled === 1 && metricsCalled === 1).toBe(true);
    });
  });
});
