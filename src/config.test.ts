import {
  config,
  DEFAULT_ACTIVATION_PRECISION,
  resolvePrecisionConfig,
} from './config';

describe('config chapter', () => {
  describe('resolvePrecisionConfig', () => {
    describe('given constructor activation precision overrides legacy float32 mode', () => {
      it('returns the explicit activation precision in the shared precision config', () => {
        // Arrange
        const memoryPrecisionFlags = {
          float32Mode: true,
        };

        // Act
        const precisionConfig = resolvePrecisionConfig(
          {
            activationPrecision: 'f64',
          },
          memoryPrecisionFlags,
        );

        // Assert
        expect(precisionConfig).toStrictEqual({
          activationPrecision: 'f64',
        });
      });
    });

    describe('given no explicit activation precision and float32 mode is enabled', () => {
      it('returns the float32 activation precision from the shared precision config', () => {
        // Arrange
        const memoryPrecisionFlags = {
          float32Mode: true,
        };

        // Act
        const precisionConfig = resolvePrecisionConfig({}, memoryPrecisionFlags);

        // Assert
        expect(precisionConfig).toStrictEqual({
          activationPrecision: 'f32',
        });
      });
    });

    describe('given no explicit precision inputs are supplied', () => {
      it('falls back to the default activation precision from the global config', () => {
        // Arrange
        const previousFloat32Mode = config.float32Mode;
        config.float32Mode = false;

        try {
          // Act
          const precisionConfig = resolvePrecisionConfig();

          // Assert
          expect(precisionConfig).toStrictEqual({
            activationPrecision: DEFAULT_ACTIVATION_PRECISION,
          });
        } finally {
          config.float32Mode = previousFloat32Mode;
        }
      });
    });
  });
});