describe('browser harness trace summary', () => {
  describe('createTraceSummary', () => {
    it('returns a summary object with scenarioUrl, durationMs, success, and metrics', async () => {
      const { createTraceSummary } = await import('../trace-summary');
      const input = {
        scenarioUrl:
          'http://localhost:8080/docs/browser-tests/webgpu-inference-smoke.html',
        durationMs: 1234,
        success: false,
        metrics: {},
      };

      const summary = createTraceSummary(input);

      expect(summary).toMatchObject({
        scenarioUrl: expect.any(String),
        durationMs: expect.any(Number),
        success: expect.any(Boolean),
        metrics: expect.any(Object),
      });
    });
  });
});
