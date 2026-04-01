import { __trainingInternals } from './network.training.utils';

describe('network training chapter', () => {
  const { computeMonitoredError, computePlateauMetric } = __trainingInternals;

  function createPlateauSmoothingState(): Parameters<
    typeof computePlateauMetric
  >[3] {
    return {};
  }

  function createPrimarySmoothingState(): Parameters<
    typeof computeMonitoredError
  >[3] {
    return {};
  }

  describe('smoothing helpers', () => {
    describe('given non-EMA smoothing with a window of one', () => {
      describe('when monitored error is computed', () => {
        it('returns the raw training error', () => {
          // Arrange
          const trainingError = 0.42;

          // Act
          const monitoredError = computeMonitoredError(
            trainingError,
            [0.3],
            { type: 'sma', window: 1 },
            {},
          );

          // Assert
          expect(monitoredError).toBe(trainingError);
        });
      });
    });

    describe('given median smoothing over an odd-length history', () => {
      describe('when monitored error is computed', () => {
        it('returns the median value', () => {
          // Arrange
          const recentErrors = [3, 1, 2];

          // Act
          const monitoredError = computeMonitoredError(
            2,
            recentErrors,
            { type: 'median', window: 3 },
            {},
          );

          // Assert
          expect(monitoredError).toBe(2);
        });
      });
    });

    describe('given EMA smoothing with a fresh state object', () => {
      describe('when monitored error is computed for the first time', () => {
        it('stores emaValue in the state object', () => {
          // Arrange
          const smoothingState = createPrimarySmoothingState();

          // Act
          const monitoredError = computeMonitoredError(
            5,
            [5],
            { type: 'ema', window: 3, emaAlpha: 0.5 },
            smoothingState,
          );

          // Assert
          expect(smoothingState.emaValue).toBe(monitoredError);
        });
      });
    });

    describe('given adaptive EMA smoothing over recent errors', () => {
      describe('when monitored error is computed', () => {
        it('returns a value no larger than the raw error', () => {
          // Arrange
          const smoothingState = createPrimarySmoothingState();

          // Act
          const monitoredError = computeMonitoredError(
            4,
            [1, 2, 3, 4],
            { type: 'adaptive-ema', window: 4 },
            smoothingState,
          );

          // Assert
          expect(monitoredError <= 4).toBe(true);
        });
      });
    });

    describe('given gaussian smoothing over recent errors', () => {
      describe('when monitored error is computed', () => {
        it('returns a bounded weighted average', () => {
          // Arrange
          const recentErrors = [1, 2, 3];

          // Act
          const monitoredError = computeMonitoredError(
            3,
            recentErrors,
            { type: 'gaussian', window: 3 },
            {},
          );

          // Assert
          expect(monitoredError > 0 && monitoredError <= 3).toBe(true);
        });
      });
    });

    describe('given trimmed smoothing over a history with an outlier', () => {
      describe('when monitored error is computed', () => {
        it('drops the tail before averaging', () => {
          // Arrange
          const recentErrors = [1, 100, 2, 3, 4];

          // Act
          const monitoredError = computeMonitoredError(
            4,
            recentErrors,
            { type: 'trimmed', window: 5, trimmedRatio: 0.2 },
            {},
          );

          // Assert
          expect(monitoredError < 100).toBe(true);
        });
      });
    });

    describe('given weighted moving average smoothing', () => {
      describe('when monitored error is computed', () => {
        it('returns a value bounded by the raw error', () => {
          // Arrange
          const recentErrors = [1, 2, 3, 4];

          // Act
          const monitoredError = computeMonitoredError(
            4,
            recentErrors,
            { type: 'wma', window: 4 },
            {},
          );

          // Assert
          expect(monitoredError <= 4).toBe(true);
        });
      });
    });

    describe('given SMA smoothing over a short history', () => {
      describe('when monitored error is computed', () => {
        it('returns the arithmetic mean', () => {
          // Arrange
          const recentErrors = [2, 4];

          // Act
          const monitoredError = computeMonitoredError(
            3,
            recentErrors,
            { type: 'sma', window: 2 },
            {},
          );

          // Assert
          expect(monitoredError).toBe(3);
        });
      });
    });

    describe('given plateau median smoothing over an odd-length history', () => {
      describe('when plateau error is computed', () => {
        it('returns the plateau median', () => {
          // Arrange
          const plateauErrors = [5, 1, 3];

          // Act
          const plateauMetric = computePlateauMetric(
            3,
            plateauErrors,
            { type: 'median', window: 3 },
            {},
          );

          // Assert
          expect(plateauMetric).toBe(3);
        });
      });
    });

    describe('given plateau EMA smoothing with a fresh state object', () => {
      describe('when plateau error is computed for the first time', () => {
        it('stores plateauEmaValue in the state object', () => {
          // Arrange
          const smoothingState = createPlateauSmoothingState();

          // Act
          const plateauMetric = computePlateauMetric(
            2,
            [2],
            { type: 'ema', window: 3, emaAlpha: 0.5 },
            smoothingState,
          );

          // Assert
          expect(smoothingState.plateauEmaValue).toBe(plateauMetric);
        });
      });
    });

    describe('given plateau smoothing with a window of one', () => {
      describe('when plateau error is computed', () => {
        it('returns the raw training error', () => {
          // Arrange
          const trainingError = 7;

          // Act
          const plateauMetric = computePlateauMetric(
            trainingError,
            [trainingError],
            { type: 'sma', window: 1 },
            {},
          );

          // Assert
          expect(plateauMetric).toBe(trainingError);
        });
      });
    });
  });
});
