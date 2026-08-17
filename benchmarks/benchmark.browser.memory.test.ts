import {
  collectBrowserPerformanceMemorySnapshot,
  runBrowserBenchPayload,
  type BrowserBenchPerformanceApi,
} from '../testing/browser-benchmarks/harness';

describe('browser memory benchmark harness', () => {
  describe('given Chromium-style heap and UA memory readings are available', () => {
    describe('when collecting the browser memory snapshot', () => {
      it('returns every available reading', async () => {
        // Arrange
        const performanceApi: BrowserBenchPerformanceApi = {
          now: () => 0,
          memory: {
            usedJSHeapSize: 256,
            totalJSHeapSize: 512,
            jsHeapSizeLimit: 1024,
          },
          measureUserAgentSpecificMemory: async () => ({
            bytes: 2048,
          }),
        };

        // Act
        const memorySnapshot =
          await collectBrowserPerformanceMemorySnapshot(performanceApi);

        // Assert
        expect(memorySnapshot).toStrictEqual({
          usedJSHeapSize: 256,
          totalJSHeapSize: 512,
          jsHeapSizeLimit: 1024,
          userAgentSpecificMemoryBytes: 2048,
        });
      });
    });
  });

  describe('given the browser harness runs with deterministic stubs', () => {
    describe('when transport comparisons are disabled for a narrow unit test', () => {
      it('returns a payload with a normalized benchmark row and memory snapshot', async () => {
        // Arrange
        const performanceApi = createDeterministicPerformanceApi();

        // Act
        const browserBenchPayload = await runBrowserBenchPayload({
          asyncBuildProbeRunner: async () => [
            {
              scenario: 'async-probe',
              connectionBudget: 12,
              requestedChunkSize: 4,
              chunkTargetMs: 1,
              frameBudgetMs: 16.7,
              elapsedMs: 2,
              microtaskYieldCount: 3,
              asyncBuildsDelta: 1,
              macrotaskHeartbeatCount: 0,
              averageMacrotaskGapMs: 2,
              animationFrameCount: 1,
              maxAnimationFrameGapMs: 8,
              maxMacrotaskGapMs: 4,
            },
          ],
          enableTransportComparisons: false,
          generatedAtFactory: () => '2026-05-08T00:00:00.000Z',
          mode: 'unit-test',
          networkFactory: () => createDeterministicBenchNetwork(),
          performanceApi,
          randomSource: () => 0.5,
          sizes: [2],
          transportScenarios: [],
        });

        // Assert
        expect(browserBenchPayload).toStrictEqual({
          mode: 'unit-test',
          generatedAt: '2026-05-08T00:00:00.000Z',
          asyncBuildComparisons: [
            {
              scenario: 'async-probe',
              connectionBudget: 12,
              requestedChunkSize: 4,
              chunkTargetMs: 1,
              frameBudgetMs: 16.7,
              elapsedMs: 2,
              microtaskYieldCount: 3,
              asyncBuildsDelta: 1,
              macrotaskHeartbeatCount: 0,
              averageMacrotaskGapMs: 2,
              animationFrameCount: 1,
              maxAnimationFrameGapMs: 8,
              maxMacrotaskGapMs: 4,
            },
          ],
          performanceMemory: {
            usedJSHeapSize: 111,
            totalJSHeapSize: 222,
            jsHeapSizeLimit: 333,
          },
          results: [
            {
              size: 2,
              buildMs: 1,
              fwdAvgMs: 0.2,
              fwdTotalMs: 1,
              conn: 2,
              nodes: 3,
              iterations: 5,
            },
          ],
          transportComparisons: [],
        });
      });
    });
  });
});

function createDeterministicPerformanceApi(): BrowserBenchPerformanceApi {
  const nowValues = [0, 1, 2, 3];
  let nowIndex = 0;

  return {
    now: () => {
      const nowValue = nowValues[nowIndex] ?? nowValues.at(-1) ?? 0;
      nowIndex += 1;
      return nowValue;
    },
    memory: {
      usedJSHeapSize: 111,
      totalJSHeapSize: 222,
      jsHeapSizeLimit: 333,
    },
  };
}

function createDeterministicBenchNetwork() {
  const connections = [
    {
      from: { id: 'input-0' },
      to: { id: 'output-0' },
    },
    {
      from: { id: 'input-1' },
      to: { id: 'output-1' },
    },
  ];

  return {
    activate: (inputValues: number[]) => {
      void inputValues;
      return [0.5];
    },
    connections,
    disconnect: (fromNode: object, toNode: object) => {
      const connectionIndex = connections.findIndex((connection) => {
        return connection.from === fromNode && connection.to === toNode;
      });
      if (connectionIndex >= 0) {
        connections.splice(connectionIndex, 1);
      }
    },
    input: 2,
    nodes: {
      length: 3,
    },
    toJSON: () => ({
      nodes: 3,
      connections: 2,
    }),
  };
}
