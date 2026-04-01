import {
  ensureTelemetryBuffer,
  safelyStreamTelemetryEntry,
  trimTelemetryBuffer,
} from './telemetry.runtime';
import type { TelemetryEntry } from '../../shared/neat.shared.types';

type TelemetryBufferHost = {
  _telemetry?: TelemetryEntry[];
};

type TelemetryStreamHost = {
  options?: {
    telemetryStream?: {
      enabled?: boolean;
      onEntry?: (entry: TelemetryEntry) => void;
    };
  };
};

function createTelemetryEntry(): TelemetryEntry {
  return {
    gen: 3,
    best: 7,
    species: 2,
    hyper: 0,
    ops: [],
    objImportance: {},
  };
}

describe('neat telemetry runtime chapter', () => {
  describe('ensureTelemetryBuffer', () => {
    describe('given no telemetry buffer exists yet', () => {
      it('initializes and returns a mutable telemetry buffer', () => {
        // Arrange
        const telemetryBufferHost: TelemetryBufferHost = {};

        // Act
        const telemetryBuffer = ensureTelemetryBuffer(telemetryBufferHost);

        // Assert
        expect(telemetryBuffer).toEqual([]);
      });
    });

    describe('given a telemetry buffer already exists', () => {
      it('returns the existing buffer reference', () => {
        // Arrange
        const existingBuffer = [createTelemetryEntry()];
        const telemetryBufferHost: TelemetryBufferHost = {
          _telemetry: existingBuffer,
        };

        // Act
        const telemetryBuffer = ensureTelemetryBuffer(telemetryBufferHost);

        // Assert
        expect(telemetryBuffer).toBe(existingBuffer);
      });
    });
  });

  describe('safelyStreamTelemetryEntry', () => {
    describe('given telemetry streaming is enabled', () => {
      it('forwards the telemetry entry to the configured callback', () => {
        // Arrange
        const streamedEntries: TelemetryEntry[] = [];
        const telemetryStreamHost: TelemetryStreamHost = {
          options: {
            telemetryStream: {
              enabled: true,
              onEntry: (entry) => streamedEntries.push(entry),
            },
          },
        };
        const telemetryEntry = createTelemetryEntry();

        // Act
        safelyStreamTelemetryEntry(telemetryStreamHost, telemetryEntry);

        // Assert
        expect(streamedEntries).toEqual([telemetryEntry]);
      });
    });

    describe('given the callback throws during streaming', () => {
      it('swallows the observer failure without throwing', () => {
        // Arrange
        const telemetryStreamHost: TelemetryStreamHost = {
          options: {
            telemetryStream: {
              enabled: true,
              onEntry: () => {
                throw new Error('stream callback failed');
              },
            },
          },
        };
        const telemetryEntry = createTelemetryEntry();

        // Act and Assert
        expect(() =>
          safelyStreamTelemetryEntry(telemetryStreamHost, telemetryEntry),
        ).not.toThrow();
      });
    });
  });

  describe('trimTelemetryBuffer', () => {
    describe('given the telemetry buffer exceeds the retention cap', () => {
      it('drops the oldest entry from the front of the buffer', () => {
        // Arrange
        const telemetryBuffer = [
          createTelemetryEntry(),
          { ...createTelemetryEntry(), gen: 4 },
          { ...createTelemetryEntry(), gen: 5 },
        ];

        // Act
        trimTelemetryBuffer(telemetryBuffer, 2);

        // Assert
        expect(telemetryBuffer.map((entry) => entry.gen)).toEqual([4, 5]);
      });
    });
  });
});
