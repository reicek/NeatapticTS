import { applyFastModeDefaults } from './telemetry.metrics.diversity';
import type { NeatOptions } from '../../shared/neat.shared.types';
import type { TelemetryDiversityOptions } from '../types/telemetry.types';

function createTelemetryOptions(input: {
  pairSample?: number;
  graphletSample?: number;
  noveltyK?: number;
}): NeatOptions & TelemetryDiversityOptions {
  return {
    fastMode: true,
    diversityMetrics: {
      enabled: true,
      pairSample: input.pairSample,
      graphletSample: input.graphletSample,
    },
    novelty: {
      enabled: true,
      k: input.noveltyK,
    },
  };
}

describe('neat telemetry diversity metrics chapter', () => {
  describe('applyFastModeDefaults', () => {
    describe('given fast mode with unspecified sampling knobs', () => {
      it('fills the diversity and novelty defaults once', () => {
        // Arrange
        const telemetryContext: { _fastModeTuned?: boolean } = {};
        const telemetryOptions = createTelemetryOptions({});

        // Act
        applyFastModeDefaults(telemetryContext, telemetryOptions);

        // Assert
        expect({
          pairSample: telemetryOptions.diversityMetrics?.pairSample,
          graphletSample: telemetryOptions.diversityMetrics?.graphletSample,
          noveltyK: telemetryOptions.novelty?.k,
          fastModeTuned: telemetryContext._fastModeTuned,
        }).toEqual({
          pairSample: 20,
          graphletSample: 30,
          noveltyK: 5,
          fastModeTuned: true,
        });
      });
    });

    describe('given fast mode with explicit sampling knobs', () => {
      it('preserves the caller supplied diversity and novelty values', () => {
        // Arrange
        const telemetryContext: { _fastModeTuned?: boolean } = {};
        const telemetryOptions = createTelemetryOptions({
          pairSample: 50,
          graphletSample: 70,
          noveltyK: 11,
        });

        // Act
        applyFastModeDefaults(telemetryContext, telemetryOptions);

        // Assert
        expect({
          pairSample: telemetryOptions.diversityMetrics?.pairSample,
          graphletSample: telemetryOptions.diversityMetrics?.graphletSample,
          noveltyK: telemetryOptions.novelty?.k,
          fastModeTuned: telemetryContext._fastModeTuned,
        }).toEqual({
          pairSample: 50,
          graphletSample: 70,
          noveltyK: 11,
          fastModeTuned: true,
        });
      });
    });
  });
});
