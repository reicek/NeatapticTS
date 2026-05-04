import type { MemoryStats, NetworkView, SlabAllocStats } from './memory';
import {
  CONNECTION_OBJECT_BYTES,
  HEURISTIC_BYTES,
  NODE_OBJECT_BYTES,
  aggregateNetworkStats,
  buildFlagSnapshot,
  buildMemoryStatsSnapshot,
  captureEnvironmentMetrics,
  normalizeNetworks,
  safeGetSlabAllocationStats,
  type Accumulators,
  type BuildMemoryStatsInput,
  type ConfigSnapshot,
} from './memory.utils';

const originalWindowDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'window',
);
const originalPerformanceDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'performance',
);
const originalProcessDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  'process',
);

describe('utils memory helper chapter', () => {
  afterEach(() => {
    jest.restoreAllMocks();
    restoreGlobalProperty('window', originalWindowDescriptor);
    restoreGlobalProperty('performance', originalPerformanceDescriptor);
    restoreGlobalProperty('process', originalProcessDescriptor);
  });

  describe('normalizeNetworks', () => {
    describe('given one explicit target network is provided', () => {
      it('wraps the single network in an array', () => {
        // Arrange
        const targetNetwork = { connections: [{}] } as NetworkView;
        const trackedNetworks = [{ connections: [] }] as NetworkView[];

        // Act
        const normalizedNetworks = normalizeNetworks(
          targetNetwork,
          trackedNetworks,
        );

        // Assert
        expect(normalizedNetworks).toEqual([targetNetwork]);
      });
    });

    describe('given explicit target networks are already provided as an array', () => {
      it('returns the same target array without falling back to the tracked registry', () => {
        // Arrange
        const targetNetworks = [{}, {}] as NetworkView[];
        const trackedNetworks = [{ connections: [] }] as NetworkView[];

        // Act
        const normalizedNetworks = normalizeNetworks(
          targetNetworks,
          trackedNetworks,
        );

        // Assert
        expect(normalizedNetworks).toBe(targetNetworks);
      });
    });

    describe('given no explicit targets are provided', () => {
      it('falls back to the tracked registry array', () => {
        // Arrange
        const trackedNetworks = [{ connections: [{}] }] as NetworkView[];

        // Act
        const normalizedNetworks = normalizeNetworks(
          undefined,
          trackedNetworks,
        );

        // Assert
        expect(normalizedNetworks).toBe(trackedNetworks);
      });
    });
  });

  describe('safeGetSlabAllocationStats', () => {
    describe('given the slab allocator probe succeeds', () => {
      it('returns the reported allocator stats unchanged', () => {
        // Arrange
        const expectedStats = { fresh: 2, pooled: 3 };

        // Act
        const allocationStats = safeGetSlabAllocationStats(() => expectedStats);

        // Assert
        expect(allocationStats).toEqual(expectedStats);
      });
    });

    describe('given the slab allocator probe throws', () => {
      it('returns null instead of propagating the allocator failure', () => {
        // Arrange
        const getSlabAllocationStats = () => {
          throw new Error('allocator probe failed');
        };

        // Act
        const allocationStats = safeGetSlabAllocationStats(
          getSlabAllocationStats,
        );

        // Assert
        expect(allocationStats).toBeNull();
      });
    });
  });

  describe('aggregateNetworkStats', () => {
    describe('given mixed network shapes including null entries and slab-backed arrays', () => {
      it('skips invalid count shapes while accumulating slab bytes, reserved capacity, and later metadata', () => {
        // Arrange
        const invalidCountNetwork = {
          connections: {} as unknown as unknown[],
          nodes: {} as unknown as unknown[],
        } as NetworkView;
        const slabBackedNetwork = {
          connections: [{}, {}],
          nodes: [{}],
          _connWeights: new Float32Array(4),
          _connFrom: new Uint32Array(4),
          _connTo: new Uint32Array(4),
          _connFlags: new Uint8Array(4),
          _connGain: new Float32Array(4),
          _connPlastic: new Float32Array(4),
          _fastA: new Float32Array(4),
          _fastS: new Float32Array(4),
          _connCapacity: 4,
          _connCount: 2,
          _useFloat32Weights: true,
          _slabVersion: null as unknown as number,
          _slabAsyncBuilds: null as unknown as number,
        } as NetworkView;
        const laterMetadataNetwork = {
          connections: [],
          nodes: [],
          _slabVersion: 7,
          _slabAsyncBuilds: 5,
        } as NetworkView;

        // Act
        const accumulators = aggregateNetworkStats(
          [
            null as unknown as NetworkView,
            invalidCountNetwork,
            slabBackedNetwork,
            laterMetadataNetwork,
          ],
          HEURISTIC_BYTES,
        );

        // Assert
        expect(accumulators).toEqual({
          totalConnections: 2,
          totalNodes: 1,
          slabBytes: 116,
          slabArrayCount: 8,
          totalReservedBytes: 84,
          totalUsedBytes: 42,
          objectConnOverheadBytes: 2 * CONNECTION_OBJECT_BYTES,
          slabVersion: 7,
          slabAsyncBuilds: 5,
        });
      });

      it('uses float64 widths and omits optional gain and plastic bytes when those slabs are absent', () => {
        // Arrange
        const float64Network = {
          connections: [{}, {}],
          nodes: [{}],
          _connCapacity: 2,
          _connCount: 1,
          _useFloat32Weights: false,
        } as NetworkView;

        // Act
        const accumulators = aggregateNetworkStats(
          [float64Network],
          HEURISTIC_BYTES,
        );

        // Assert
        expect({
          reservedBytes: accumulators.totalReservedBytes,
          usedBytes: accumulators.totalUsedBytes,
        }).toEqual({
          reservedBytes: 34,
          usedBytes: 17,
        });
      });
    });
  });

  describe('captureEnvironmentMetrics', () => {
    describe('given browser heap counters and node memory usage are available', () => {
      it('captures both browser and node environment metrics in one snapshot', () => {
        // Arrange
        setGlobalProperty('window', {});
        setGlobalProperty('performance', {
          memory: {
            usedJSHeapSize: 10,
            totalJSHeapSize: 20,
            jsHeapSizeLimit: 30,
          },
        });
        setGlobalProperty('process', {
          memoryUsage: () =>
            ({
              rss: 40,
              heapUsed: 50,
              heapTotal: 60,
              external: 70,
              arrayBuffers: 80,
            }) as NodeJS.MemoryUsage,
        });

        // Act
        const environmentMetrics = captureEnvironmentMetrics();

        // Assert
        expect(environmentMetrics).toEqual({
          isBrowser: true,
          usedJSHeapSize: 10,
          totalJSHeapSize: 20,
          jsHeapSizeLimit: 30,
          rss: 40,
          heapUsed: 50,
          heapTotal: 60,
          external: 70,
        });
      });
    });

    describe('given the browser and node memory probes throw during sampling', () => {
      it('swallows the probe failures and returns only the base environment flag', () => {
        // Arrange
        const performanceWithThrowingMemory = {};
        Object.defineProperty(performanceWithThrowingMemory, 'memory', {
          configurable: true,
          get() {
            throw new Error('browser probe failed');
          },
        });
        setGlobalProperty('performance', performanceWithThrowingMemory);
        setGlobalProperty('process', {
          memoryUsage: () => {
            throw new Error('node probe failed');
          },
        });

        // Act
        const environmentMetrics = captureEnvironmentMetrics();

        // Assert
        expect(environmentMetrics).toEqual({
          isBrowser: false,
        });
      });
    });

    describe('given performance exists but does not expose heap counters', () => {
      it('leaves the browser heap fields undefined', () => {
        // Arrange
        setGlobalProperty('window', {});
        setGlobalProperty('performance', {
          memory: undefined,
        });
        setGlobalProperty('process', {});

        // Act
        const environmentMetrics = captureEnvironmentMetrics();

        // Assert
        expect(environmentMetrics).toEqual({
          isBrowser: true,
        });
      });
    });

    describe('given the process object exists without a memoryUsage function', () => {
      it('leaves the node-specific counters undefined', () => {
        // Arrange
        setGlobalProperty('performance', {});
        setGlobalProperty('process', {});

        // Act
        const environmentMetrics = captureEnvironmentMetrics();

        // Assert
        expect(environmentMetrics).toEqual({
          isBrowser: false,
        });
      });
    });
  });

  describe('buildFlagSnapshot', () => {
    describe('given node pooling is omitted from the config snapshot', () => {
      it('defaults the node-pooling flag to false while preserving the other config fields', () => {
        // Arrange
        const configSnapshot: ConfigSnapshot = {
          warnings: 'warn',
          float32Mode: 'float32',
          deterministicChainMode: 'chain',
          enableGatingTraces: 'gates',
          poolMaxPerBucket: null,
          poolPrewarmCount: null,
        };

        // Act
        const flagSnapshot = buildFlagSnapshot(configSnapshot, null);

        // Assert
        expect(flagSnapshot).toEqual({
          warnings: 'warn',
          float32Mode: 'float32',
          deterministicChainMode: 'chain',
          enableGatingTraces: 'gates',
          poolMaxPerBucket: null,
          poolPrewarmCount: null,
          enableNodePooling: false,
          allocStats: null,
        });
      });
    });
  });

  describe('buildMemoryStatsSnapshot', () => {
    describe('given allocation stats are unavailable for the slab snapshot', () => {
      it('builds a memory snapshot with zero bytes per connection and a null pooled fraction', () => {
        // Arrange
        jest.spyOn(Date, 'now').mockReturnValue(1_234);
        const input = buildMemoryStatsInput({
          networks: [],
          accumulators: buildAccumulators({
            totalConnections: 0,
            totalNodes: 2,
            slabBytes: 16,
          }),
          allocationStats: null,
          flagSnapshot: {
            warnings: false,
            float32Mode: false,
            deterministicChainMode: false,
            enableGatingTraces: false,
            poolMaxPerBucket: null,
            poolPrewarmCount: null,
            enableNodePooling: false,
            allocStats: null,
          },
        });

        // Act
        const memorySnapshot = buildMemoryStatsSnapshot(input);

        // Assert
        expect(memorySnapshot).toEqual({
          timestamp: 1_234,
          connections: 0,
          nodes: 2,
          bytesPerConnection: 0,
          estimatedTotalBytes: 2 * NODE_OBJECT_BYTES + 16,
          slabs: {
            slabBytes: 16,
            slabArrayCount: 0,
            fragmentationPct: null,
            reservedBytes: null,
            usedBytes: null,
            slabVersion: null,
            asyncBuilds: 0,
            pooledFraction: null,
          },
          pools: {
            nodePool: null,
          },
          flags: {
            snapshot: {
              warnings: false,
              float32Mode: false,
              deterministicChainMode: false,
              enableGatingTraces: false,
              poolMaxPerBucket: null,
              poolPrewarmCount: null,
              enableNodePooling: false,
              allocStats: null,
            },
          },
          env: {
            isBrowser: false,
          },
        });
      });
    });

    describe('given reserved bytes and pooled allocations are available', () => {
      it('computes bytes per connection, fragmentation percentage, and pooled fraction from the accumulators', () => {
        // Arrange
        jest.spyOn(Date, 'now').mockReturnValue(4_321);
        const input = buildMemoryStatsInput({
          networks: [{ _slabVersion: 9, _slabAsyncBuilds: 2 } as NetworkView],
          accumulators: buildAccumulators({
            totalConnections: 4,
            totalNodes: 1,
            slabBytes: 36,
            slabArrayCount: 3,
            totalReservedBytes: 100,
            totalUsedBytes: 25,
            objectConnOverheadBytes: CONNECTION_OBJECT_BYTES,
            slabVersion: 4,
            slabAsyncBuilds: 1,
          }),
          allocationStats: { fresh: 3, pooled: 1 },
          flagSnapshot: {
            warnings: true,
            float32Mode: true,
            deterministicChainMode: true,
            enableGatingTraces: true,
            poolMaxPerBucket: 8,
            poolPrewarmCount: 2,
            enableNodePooling: true,
            allocStats: { fresh: 3, pooled: 1 },
          },
        });

        // Act
        const memorySnapshot = buildMemoryStatsSnapshot(input);

        // Assert
        expect({
          timestamp: memorySnapshot.timestamp,
          bytesPerConnection: memorySnapshot.bytesPerConnection,
          fragmentationPct: memorySnapshot.slabs.fragmentationPct,
          reservedBytes: memorySnapshot.slabs.reservedBytes,
          usedBytes: memorySnapshot.slabs.usedBytes,
          slabVersion: memorySnapshot.slabs.slabVersion,
          asyncBuilds: memorySnapshot.slabs.asyncBuilds,
          pooledFraction: memorySnapshot.slabs.pooledFraction,
        }).toEqual({
          timestamp: 4_321,
          bytesPerConnection: 43,
          fragmentationPct: 75,
          reservedBytes: 100,
          usedBytes: 25,
          slabVersion: 9,
          asyncBuilds: 2,
          pooledFraction: 0.25,
        });
      });
    });

    describe('given allocator stats report no fresh or pooled allocations', () => {
      it('returns a null pooled fraction instead of dividing by zero', () => {
        // Arrange
        jest.spyOn(Date, 'now').mockReturnValue(9_999);
        const input = buildMemoryStatsInput({
          allocationStats: { fresh: 0, pooled: 0 },
        });

        // Act
        const memorySnapshot = buildMemoryStatsSnapshot(input);

        // Assert
        expect(memorySnapshot.slabs.pooledFraction).toBeNull();
      });
    });
  });
});

function setGlobalProperty(
  propertyName: 'window' | 'performance' | 'process',
  value: unknown,
): void {
  Object.defineProperty(globalThis, propertyName, {
    configurable: true,
    writable: true,
    value,
  });
}

function restoreGlobalProperty(
  propertyName: 'window' | 'performance' | 'process',
  descriptor: PropertyDescriptor | undefined,
): void {
  if (descriptor) {
    Object.defineProperty(globalThis, propertyName, descriptor);
    return;
  }

  Reflect.deleteProperty(globalThis, propertyName);
}

function buildAccumulators(
  overrides: Partial<Accumulators> = {},
): Accumulators {
  return {
    totalConnections: 0,
    totalNodes: 0,
    slabBytes: 0,
    slabArrayCount: 0,
    totalReservedBytes: 0,
    totalUsedBytes: 0,
    objectConnOverheadBytes: 0,
    slabVersion: null,
    slabAsyncBuilds: 0,
    ...overrides,
  };
}

function buildMemoryStatsInput(
  overrides: Partial<BuildMemoryStatsInput> = {},
): BuildMemoryStatsInput {
  return {
    networks: [],
    accumulators: buildAccumulators(),
    env: {
      isBrowser: false,
    },
    heuristics: HEURISTIC_BYTES,
    allocationStats: { fresh: 1, pooled: 3 } as SlabAllocStats,
    nodePoolSnapshot: null,
    flagSnapshot: {
      warnings: false,
      float32Mode: false,
      deterministicChainMode: false,
      enableGatingTraces: false,
      poolMaxPerBucket: null,
      poolPrewarmCount: null,
      enableNodePooling: false,
      allocStats: null,
    } as MemoryStats['flags']['snapshot'],
    ...overrides,
  };
}
