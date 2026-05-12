describe('FLAPPY_EVOLUTION_WORKER_INTERNALS', () => {
  it('keeps recurrent browser profiles on the worker-local evaluator path when shared memory is unavailable', async () => {
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

  it('enables the shared-memory evaluation pool when browser worker capabilities are available', async () => {
    const originalCrossOriginIsolated = Object.getOwnPropertyDescriptor(
      globalThis,
      'crossOriginIsolated',
    );
    const originalLocation = Object.getOwnPropertyDescriptor(
      globalThis,
      'location',
    );
    const originalSelf = globalThis.self;
    const originalSharedArrayBuffer = globalThis.SharedArrayBuffer;
    const originalWorker = globalThis.Worker;

    Object.defineProperty(globalThis, 'crossOriginIsolated', {
      configurable: true,
      value: true,
    });
    Object.defineProperty(globalThis, 'location', {
      configurable: true,
      value: {
        href: 'https://example.test/assets/flappy-evolution.worker.bundle.js',
      },
    });
    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: {
        onmessage: undefined,
        postMessage: jest.fn(),
      },
    });
    Object.defineProperty(globalThis, 'SharedArrayBuffer', {
      configurable: true,
      value: SharedArrayBuffer,
    });
    Object.defineProperty(globalThis, 'Worker', {
      configurable: true,
      value: class Worker {},
    });

    try {
      const { FLAPPY_EVOLUTION_WORKER_INTERNALS } =
        await import('./flappy-evolution-worker');
      const evaluationWorkerPool =
        FLAPPY_EVOLUTION_WORKER_INTERNALS.createWorkerEvaluationPoolIfSupported(
          {
            architectureProfileId: 'lstm',
            populationSize: 10,
            elitismCount: 2,
            rngSeed: 12345,
          },
        );

      await evaluationWorkerPool?.dispose();

      expect(Boolean(evaluationWorkerPool)).toBe(true);
    } finally {
      restoreGlobalProperty('crossOriginIsolated', originalCrossOriginIsolated);
      restoreGlobalProperty('location', originalLocation);
      Object.defineProperty(globalThis, 'self', {
        configurable: true,
        value: originalSelf,
      });
      Object.defineProperty(globalThis, 'SharedArrayBuffer', {
        configurable: true,
        value: originalSharedArrayBuffer,
      });
      Object.defineProperty(globalThis, 'Worker', {
        configurable: true,
        value: originalWorker,
      });
    }
  });

  it('resolves direct and shared-memory evaluation status payloads', async () => {
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
      const directStatus =
        FLAPPY_EVOLUTION_WORKER_INTERNALS.resolveWorkerEvaluationRuntimeStatusPayload(
          undefined,
        );
      const sharedStatus =
        FLAPPY_EVOLUTION_WORKER_INTERNALS.resolveWorkerEvaluationRuntimeStatusPayload(
          {} as never,
        );

      expect({
        directPhase: directStatus.phase,
        directStatusText: directStatus.statusText,
        sharedPhase: sharedStatus.phase,
        sharedStatusText: sharedStatus.statusText,
      }).toEqual({
        directPhase: 'evaluating-direct',
        directStatusText: 'direct eval fallback',
        sharedPhase: 'evaluating-shared-memory',
        sharedStatusText: 'parallel eval ready',
      });
    } finally {
      Object.defineProperty(globalThis, 'self', {
        configurable: true,
        value: originalSelf,
      });
    }
  });
});

/**
 * Restores a temporary global property override used by browser-capability tests.
 *
 * @param propertyName - Global property name that was overridden.
 * @param descriptor - Original descriptor captured before the override.
 * @returns Nothing.
 */
function restoreGlobalProperty(
  propertyName: 'crossOriginIsolated' | 'location',
  descriptor: PropertyDescriptor | undefined,
): void {
  if (descriptor) {
    Object.defineProperty(globalThis, propertyName, descriptor);
    return;
  }

  Reflect.deleteProperty(globalThis, propertyName);
}
