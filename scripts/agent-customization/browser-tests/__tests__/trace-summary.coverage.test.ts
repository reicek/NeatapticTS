import { createTraceSummary, TraceInput } from '../trace-summary';

describe('trace-summary branch coverage', () => {
  describe('createTraceSummary error cases', () => {
    it('throws when input is null', () => {
      expect(() => createTraceSummary(null as unknown as TraceInput)).toThrow(
        'Trace input must include a scenarioUrl string.',
      );
    });

    it('throws when input is undefined', () => {
      expect(() =>
        createTraceSummary(undefined as unknown as TraceInput),
      ).toThrow('Trace input must include a scenarioUrl string.');
    });

    it('throws when scenarioUrl is not a string', () => {
      expect(() =>
        createTraceSummary({
          scenarioUrl: 123,
          durationMs: 100,
        } as unknown as TraceInput),
      ).toThrow('Trace input must include a scenarioUrl string.');
    });

    it('throws when durationMs is not a number', () => {
      expect(() =>
        createTraceSummary({
          scenarioUrl: 'http://localhost:8080/test.html',
          durationMs: '100',
        } as unknown as TraceInput),
      ).toThrow('Trace input must include a durationMs number.');
    });

    it('throws when durationMs is missing', () => {
      expect(() =>
        createTraceSummary({
          scenarioUrl: 'http://localhost:8080/test.html',
        } as unknown as TraceInput),
      ).toThrow('Trace input must include a durationMs number.');
    });
  });

  describe('createTraceSummary edge cases', () => {
    it('provides default metrics when none are supplied', () => {
      const summary = createTraceSummary({
        scenarioUrl: 'http://localhost:8080/test.html',
        durationMs: 100,
        success: true,
      });

      expect(summary.metrics).toEqual({ keyCount: 0, success: true });
    });

    it('clamps negative durationMs to zero', () => {
      const summary = createTraceSummary({
        scenarioUrl: 'http://localhost:8080/test.html',
        durationMs: -50,
        success: false,
      });

      expect(summary.durationMs).toBe(0);
      expect(summary.success).toBe(false);
    });

    it('uses provided metrics and preserves positive durationMs', () => {
      const summary = createTraceSummary({
        scenarioUrl: 'http://localhost:8080/test.html',
        durationMs: 1234,
        success: true,
        metrics: { maxAbsDiff: 0.05, keyCount: 3 },
      });

      expect(summary.durationMs).toBe(1234);
      expect(summary.success).toBe(true);
      expect(summary.metrics).toEqual({ maxAbsDiff: 0.05, keyCount: 3 });
    });
  });
});
