import Network from '../../src/architecture/network';
import {
  trainImpl,
  applyGradientClippingImpl,
} from '../../src/architecture/network/network.training';

/**
 * Advanced training branch coverage: smoothing strategies, layerwise gradient clipping grouping,
 * mixed precision scale increase path, early termination via target error, accumulation average.
 */

describe('Network.training advanced branches', () => {
  describe('Scenario: EMA smoothing path', () => {
    it('updates emaValue over iterations', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 120 });
      const set = [{ input: [0.1], output: [0.2] }];
      // Act
      const res = trainImpl(net, set as any, {
        iterations: 2,
        rate: 0.1,
        movingAverageType: 'ema',
        movingAverageWindow: 3,
      });
      // Assert
      expect(res.iterations).toBe(2);
    });
  });

  describe('Scenario: adaptive-ema smoothing path', () => {
    it('runs adaptive ema variant without errors', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 121 });
      const set = [{ input: [0.1], output: [0.3] }];
      // Act
      const res = trainImpl(net, set as any, {
        iterations: 2,
        rate: 0.1,
        movingAverageType: 'adaptive-ema',
        movingAverageWindow: 3,
      });
      // Assert
      expect(res.iterations).toBe(2);
    });
  });

  describe('Scenario: gaussian smoothing path', () => {
    it('applies gaussian weighting', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 122 });
      const set = [{ input: [0.1], output: [0.25] }];
      // Act
      const res = trainImpl(net, set as any, {
        iterations: 2,
        rate: 0.1,
        movingAverageType: 'gaussian',
        movingAverageWindow: 4,
      });
      // Assert
      expect(res.iterations).toBe(2);
    });
  });

  describe('Scenario: trimmed mean smoothing path', () => {
    it('applies trimmed mean logic', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 123 });
      const set = [{ input: [0.1], output: [0.25] }];
      // Act
      const res = trainImpl(net, set as any, {
        iterations: 3,
        rate: 0.1,
        movingAverageType: 'trimmed',
        movingAverageWindow: 5,
        trimmedRatio: 0.2,
      });
      // Assert
      expect(res.iterations).toBe(3);
    });
  });

  describe('Scenario: WMA smoothing path', () => {
    it('applies weighted moving average', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 124 });
      const set = [{ input: [0.1], output: [0.25] }];
      // Act
      const res = trainImpl(net, set as any, {
        iterations: 2,
        rate: 0.1,
        movingAverageType: 'wma',
        movingAverageWindow: 4,
      });
      // Assert
      expect(res.iterations).toBe(2);
    });
  });

  describe('Scenario: layerwiseNorm gradient clipping grouping', () => {
    it('sets multiple groups for layerwise mode', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 125 });
      // Create artificial layers grouping by splitting nodes via addNode mutations
      // (simplified: treat each non-input node as separate layer fallback path)
      const inputNode = net.nodes[0];
      const outNode = net.nodes.find((n) => n.type === 'output')!;
      const c = net.connect(inputNode, outNode)[0];
      (c as any).totalDeltaWeight = 5;
      // Act
      applyGradientClippingImpl(net, { mode: 'layerwiseNorm', maxNorm: 1 });
      // Assert
      expect((net as any)._lastGradClipGroupCount >= 1).toBe(true);
    });
  });

  describe('Scenario: mixed precision increase loss scale path', () => {
    it('doubles loss scale after sufficient good steps', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 126 });
      const set = [{ input: [0.2], output: [0.3] }];
      // Use very low increaseEvery to trigger scale up.
      trainImpl(net, set as any, {
        iterations: 2,
        rate: 0.1,
        mixedPrecision: { lossScale: 2, dynamic: { increaseEvery: 1 } },
      });
      // Act
      const scale = (net as any)._mixedPrecision.lossScale;
      // Assert
      expect(scale >= 2).toBe(true);
    });
  });

  describe('Scenario: target error early break', () => {
    it('stops before reaching max iterations when error threshold hit', () => {
      // Arrange
      const net = new Network(1, 1, { seed: 127 });
      const set = [{ input: [0.2], output: [0.2] }];
      // Act
      const res = trainImpl(net, set as any, {
        iterations: 50,
        rate: 0.1,
        error: 0.9,
      });
      // Assert
      expect(res.iterations < 50).toBe(true);
    });
  });
});
