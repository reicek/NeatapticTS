import {
  getTelemetryCoreSnapshot,
  mergeTelemetryCoreFields,
  safelyApplyTelemetrySelect,
  stripUnselectedTelemetryKeys,
} from './telemetry.metrics.selection';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('telemetry metrics selection chapter', () => {
  describe('getTelemetryCoreSnapshot()', () => {
    describe('given a source entry missing one of the requested core fields', () => {
      it('omits the missing field from the snapshot (line 24 FALSE arm)', () => {
        // Arrange: sourceEntry has 'generation' but not 'score'
        const sourceEntry = { generation: 5 };
        const fields = ['generation', 'score'] as unknown as string[] & {
          includes: (s: string) => boolean;
        };

        // Act
        const result = getTelemetryCoreSnapshot(
          sourceEntry,
          fields as Parameters<typeof getTelemetryCoreSnapshot>[1],
        );

        // Assert: snapshot only contains the field that existed
        expect(Object.keys(result)).toEqual(['generation']);
      });
    });

    describe('given a source entry with all requested core fields present', () => {
      it('includes all present core fields in the snapshot', () => {
        // Arrange
        const sourceEntry = { generation: 3, score: 0.9 };
        const fields = ['generation', 'score'] as unknown as Parameters<
          typeof getTelemetryCoreSnapshot
        >[1];

        // Act
        const result = getTelemetryCoreSnapshot(sourceEntry, fields);

        // Assert
        expect(result).toEqual({ generation: 3, score: 0.9 });
      });
    });
  });

  describe('stripUnselectedTelemetryKeys()', () => {
    describe('given an entry with a core field and an extra whitelisted field', () => {
      it('keeps both the core field and the whitelisted field', () => {
        // Arrange
        const entry = { generation: 1, score: 0.8, extra: 'keep' };
        const selection = new Set(['extra']);
        const fields = ['generation'] as unknown as Parameters<
          typeof stripUnselectedTelemetryKeys
        >[2];

        // Act
        const result = stripUnselectedTelemetryKeys(entry, selection, fields);

        // Assert
        expect(Object.keys(result)).toEqual(
          expect.arrayContaining(['generation', 'extra']),
        );
      });
    });

    describe('given an entry with a non-core field not in the selection set', () => {
      it('removes the non-whitelisted field from the entry', () => {
        // Arrange
        const entry = { generation: 2, droppedKey: 'remove' };
        const selection = new Set<string>();
        const fields = ['generation'] as unknown as Parameters<
          typeof stripUnselectedTelemetryKeys
        >[2];

        // Act
        stripUnselectedTelemetryKeys(entry, selection, fields);

        // Assert
        expect('droppedKey' in entry).toBe(false);
      });
    });
  });

  describe('mergeTelemetryCoreFields()', () => {
    describe('given a snapshot with core fields', () => {
      it('re-applies the snapshot fields onto the entry', () => {
        // Arrange
        const entry: Record<string, unknown> = {};
        const snapshot = { generation: 7, score: 0.5 };

        // Act
        mergeTelemetryCoreFields(entry, snapshot);

        // Assert
        expect(entry).toEqual({ generation: 7, score: 0.5 });
      });
    });
  });

  describe('safelyApplyTelemetrySelect()', () => {
    describe('given a selection function that throws', () => {
      it('swallows the error without rethrowing', () => {
        // Arrange
        const context = { options: {} };
        const entry = { generation: 1 };
        const throwingFn = () => {
          throw new Error('selection error');
        };

        // Act + Assert: must not propagate the thrown error
        expect(() =>
          safelyApplyTelemetrySelect(
            context as Parameters<typeof safelyApplyTelemetrySelect>[0],
            entry as unknown as Parameters<
              typeof safelyApplyTelemetrySelect
            >[1],
            throwingFn as Parameters<typeof safelyApplyTelemetrySelect>[2],
          ),
        ).not.toThrow();
      });
    });
  });
});
