import type { NeatOptions } from '../../shared/neat.shared.types';
import type {
  TelemetryDiversityOptions,
  TelemetryEntryRecord,
} from '../types/telemetry.types';
import { applyRngState } from './telemetry.metrics.rng';

function createTelemetryOptions(
  rngState: boolean,
): NeatOptions & TelemetryDiversityOptions {
  return {
    rngState,
  };
}

function createTelemetryEntry(): TelemetryEntryRecord {
  return {
    best: 7,
    gen: 3,
    hyper: 0,
    species: 2,
  } as TelemetryEntryRecord;
}

describe('neat telemetry rng metrics chapter', () => {
  describe('applyRngState', () => {
    describe('when RNG telemetry is enabled and the context exposes a state snapshot', () => {
      it('attaches the numeric RNG state to the telemetry entry', () => {
        // Arrange
        const telemetryContext = { _rngState: 12345 };
        const telemetryOptions = createTelemetryOptions(true);
        const entry = createTelemetryEntry();

        // Act
        applyRngState(telemetryContext, telemetryOptions, entry);

        // Assert
        expect(entry).toMatchObject({ rng: 12345 });
      });
    });

    describe('when RNG telemetry is disabled', () => {
      it('leaves the telemetry entry without an RNG field', () => {
        // Arrange
        const telemetryContext = { _rngState: 12345 };
        const telemetryOptions = createTelemetryOptions(false);
        const entry = createTelemetryEntry();

        // Act
        applyRngState(telemetryContext, telemetryOptions, entry);

        // Assert
        expect(Object.hasOwn(entry, 'rng')).toBe(false);
      });
    });

    describe('when RNG telemetry is enabled but the context has no state snapshot', () => {
      it('leaves the telemetry entry without an RNG field', () => {
        // Arrange
        const telemetryContext = {};
        const telemetryOptions = createTelemetryOptions(true);
        const entry = createTelemetryEntry();

        // Act
        applyRngState(telemetryContext, telemetryOptions, entry);

        // Assert
        expect(Object.hasOwn(entry, 'rng')).toBe(false);
      });
    });
  });
});
