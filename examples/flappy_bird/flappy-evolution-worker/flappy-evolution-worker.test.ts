describe('FLAPPY_EVOLUTION_WORKER_INTERNALS', () => {
  it('keeps recurrent browser profiles on the worker-local evaluator path', async () => {
    const originalSelf = globalThis.self;

    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: {
        onmessage: undefined,
        postMessage: jest.fn(),
      },
    });

    try {
      const { FLAPPY_EVOLUTION_WORKER_INTERNALS } =
        await import('./flappy-evolution-worker');
      const evaluationWorkerPool =
        FLAPPY_EVOLUTION_WORKER_INTERNALS.createWorkerEvaluationPoolIfSupported(
          {
            architectureProfileId: 'narx',
            populationSize: 10,
            elitismCount: 2,
            rngSeed: 12345,
          },
        );

      expect(evaluationWorkerPool).toBeUndefined();
    } finally {
      Object.defineProperty(globalThis, 'self', {
        configurable: true,
        value: originalSelf,
      });
    }
  });
});
