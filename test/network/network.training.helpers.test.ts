import { __trainingInternals } from '../../src/architecture/network/network.training';

/**
 * Direct helper tests for computeMonitoredError / computePlateauMetric covering fast path and each branch.
 */

describe('Network.training helper smoothing functions', () => {
  const {
    computeMonitoredError,
    computePlateauMetric,
  } = __trainingInternals as any;

  describe('Scenario: fast path returns trainError when window<=1 and non-EMA types', () => {
    it('returns raw error', () => {
      // Arrange
      const err = 0.42;
      const recent = [0.3];
      // Act
      const out = computeMonitoredError(
        err,
        recent,
        { type: 'sma', window: 1 },
        {}
      );
      // Assert
      expect(out).toBe(err);
    });
  });

  describe('Scenario: median smoothing path', () => {
    it('returns median of odd length', () => {
      // Arrange
      const recent = [3, 1, 2];
      // Act
      const out = computeMonitoredError(
        2,
        recent,
        { type: 'median', window: 3 },
        {}
      );
      // Assert
      expect(out).toBe(2);
    });
  });

  describe('Scenario: ema smoothing initializes state', () => {
    it('stores emaValue on first pass', () => {
      // Arrange
      const state: any = {};
      // Act
      const out = computeMonitoredError(
        5,
        [5],
        { type: 'ema', window: 3, emaAlpha: 0.5 },
        state
      );
      // Assert
      expect(state.emaValue).toBe(out);
    });
  });

  describe('Scenario: adaptive-ema dual path', () => {
    it('returns min of base and adaptive', () => {
      // Arrange
      const state: any = {};
      const recent = [1, 2, 3, 4];
      // Act
      const out = computeMonitoredError(
        4,
        recent,
        { type: 'adaptive-ema', window: 4 },
        state
      );
      // Assert
      expect(out <= 4).toBe(true);
    });
  });

  describe('Scenario: gaussian smoothing', () => {
    it('produces weighted average', () => {
      // Arrange
      const recent = [1, 2, 3];
      // Act
      const out = computeMonitoredError(
        3,
        recent,
        { type: 'gaussian', window: 3 },
        {}
      );
      // Assert
      expect(out > 0 && out <= 3).toBe(true);
    });
  });

  describe('Scenario: trimmed smoothing', () => {
    it('drops tails before averaging', () => {
      // Arrange
      const recent = [1, 100, 2, 3, 4];
      // Act
      const out = computeMonitoredError(
        4,
        recent,
        { type: 'trimmed', window: 5, trimmedRatio: 0.2 },
        {}
      );
      // Assert
      expect(out < 100).toBe(true);
    });
  });

  describe('Scenario: wma smoothing', () => {
    it('applies linear weights', () => {
      // Arrange
      const recent = [1, 2, 3, 4];
      // Act
      const out = computeMonitoredError(
        4,
        recent,
        { type: 'wma', window: 4 },
        {}
      );
      // Assert
      expect(out <= 4).toBe(true);
    });
  });

  describe('Scenario: default sma fallback', () => {
    it('returns arithmetic mean for unrecognized type', () => {
      // Arrange
      const recent = [2, 4];
      // Act
      const out = computeMonitoredError(
        3,
        recent,
        { type: 'sma', window: 2 },
        {}
      );
      // Assert
      expect(out).toBe(3);
    });
  });

  describe('Scenario: plateau median path', () => {
    it('computes plateau median', () => {
      // Arrange
      const plateau = [5, 1, 3];
      // Act
      const out = computePlateauMetric(
        3,
        plateau,
        { type: 'median', window: 3 },
        {}
      );
      // Assert
      expect(out).toBe(3);
    });
  });

  describe('Scenario: plateau ema path', () => {
    it('updates plateauEmaValue', () => {
      // Arrange
      const state: any = {};
      // Act
      const out = computePlateauMetric(
        2,
        [2],
        { type: 'ema', window: 3, emaAlpha: 0.5 },
        state
      );
      // Assert
      expect(state.plateauEmaValue).toBe(out);
    });
  });

  describe('Scenario: plateau fast path window<=1', () => {
    it('returns raw train error', () => {
      // Arrange
      const out = computePlateauMetric(7, [7], { type: 'sma', window: 1 }, {});
      // Assert
      expect(out).toBe(7);
    });
  });
});
