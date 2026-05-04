import {
  buildMonitoredSmoothingConfig,
  resolveEmaAlpha,
} from './network.training.utils.types';

describe('network training utility types chapter', () => {
  describe('resolveEmaAlpha', () => {
    describe('given a valid explicit alpha override is provided', () => {
      it('returns the explicit alpha instead of the window-derived default', () => {
        // Arrange
        const smoothingWindow = 5;
        const explicitAlpha = 0.25;

        // Act
        const resolvedAlpha = resolveEmaAlpha(smoothingWindow, explicitAlpha);

        // Assert
        expect(resolvedAlpha).toBe(explicitAlpha);
      });
    });

    describe('given no valid explicit alpha override is provided', () => {
      it('returns the window-derived default alpha', () => {
        // Arrange
        const smoothingWindow = 5;

        // Act
        const resolvedAlpha = resolveEmaAlpha(smoothingWindow, undefined);

        // Assert
        expect(resolvedAlpha).toBe(2 / (smoothingWindow + 1));
      });
    });
  });

  describe('buildMonitoredSmoothingConfig', () => {
    describe('given monitored smoothing inputs are provided', () => {
      it('returns the normalized smoothing configuration object', () => {
        // Arrange
        const type = 'trimmed';
        const window = 7;
        const emaAlpha = 0.4;
        const trimmedRatio = 0.2;

        // Act
        const smoothingConfig = buildMonitoredSmoothingConfig(
          type,
          window,
          emaAlpha,
          trimmedRatio,
        );

        // Assert
        expect(smoothingConfig).toEqual({
          type,
          window,
          emaAlpha,
          trimmedRatio,
        });
      });
    });
  });
});
