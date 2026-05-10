import { config, type NeatapticConfig } from '../config';
import {
  MEMORY_DEFAULT_ACTIVATION_POOL_PREWARM_COUNT,
  MEMORY_DEFAULT_SLAB_POOL_MAX_PER_KEY,
  MEMORY_SLAB_GROWTH_FACTOR_BROWSER,
  MEMORY_SLAB_GROWTH_FACTOR_NODE,
} from './config';
import { MemoryManager } from './manager';

function buildConfig(): NeatapticConfig {
  return {
    warnings: false,
    float32Mode: false,
    deterministicChainMode: false,
    enableGatingTraces: true,
    enableNodePooling: false,
    enableSlabArrayPooling: false,
  };
}

describe('memory manager', () => {
  describe('given one sparse config omits optional memory flags', () => {
    describe('when reading a browser runtime snapshot', () => {
      it('fills the documented boolean defaults from the manager layer', () => {
        // Arrange
        const manager = new MemoryManager({
          warnings: false,
          float32Mode: false,
        } as NeatapticConfig);

        // Act
        const snapshot = manager.getConfig('browser');

        // Assert
        expect({
          deterministicChainMode: snapshot.deterministicChainMode,
          enableGatingTraces: snapshot.enableGatingTraces,
          enableNodePooling: snapshot.enableNodePooling,
          enableSlabArrayPooling: snapshot.enableSlabArrayPooling,
        }).toStrictEqual({
          deterministicChainMode: false,
          enableGatingTraces: true,
          enableNodePooling: false,
          enableSlabArrayPooling: false,
        });
      });
    });
  });

  describe('given the manager uses the shared global config by default', () => {
    describe('when reading one implicit runtime snapshot', () => {
      it('resolves the current node environment without an explicit constructor config', () => {
        // Arrange
        const manager = new MemoryManager();

        // Act
        const snapshot = manager.getConfig();

        // Assert
        expect({
          environment: snapshot.environment,
          warnings: snapshot.warnings,
        }).toStrictEqual({
          environment: 'node',
          warnings: config.warnings,
        });
      });

      it('resolves the browser environment when window exists and no runtime override is provided', () => {
        // Arrange
        const originalWindow = globalThis.window;
        Object.defineProperty(globalThis, 'window', {
          configurable: true,
          value: {},
        });
        const manager = new MemoryManager(buildConfig());

        try {
          // Act
          const snapshot = manager.getConfig();

          // Assert
          expect(snapshot.environment).toBe('browser');
        } finally {
          if (typeof originalWindow === 'undefined') {
            Reflect.deleteProperty(globalThis, 'window');
            return;
          }

          Object.defineProperty(globalThis, 'window', {
            configurable: true,
            value: originalWindow,
          });
        }
      });
    });
  });

  describe('given config uses implicit defaults', () => {
    describe('when reading a node runtime snapshot', () => {
      it('resolves the documented memory defaults', () => {
        // Arrange
        const manager = new MemoryManager(buildConfig());

        // Act
        const snapshot = manager.getConfig('node');

        // Assert
        expect({
          activationPoolPrewarmCount: snapshot.activationPoolPrewarmCount,
          environment: snapshot.environment,
          slabGrowthFactor: snapshot.slabGrowthFactor,
          slabPoolMaxPerKey: snapshot.slabPoolMaxPerKey,
        }).toStrictEqual({
          activationPoolPrewarmCount:
            MEMORY_DEFAULT_ACTIVATION_POOL_PREWARM_COUNT,
          environment: 'node',
          slabGrowthFactor: MEMORY_SLAB_GROWTH_FACTOR_NODE,
          slabPoolMaxPerKey: MEMORY_DEFAULT_SLAB_POOL_MAX_PER_KEY,
        });
      });
    });
  });

  describe('given config uses explicit memory overrides', () => {
    describe('when reading a browser runtime snapshot', () => {
      it('preserves explicit flags while switching browser growth policy', () => {
        // Arrange
        const config = buildConfig();
        config.enableSlabArrayPooling = true;
        config.poolMaxPerBucket = 12;
        config.poolPrewarmCount = 6;
        config.slabPoolMaxPerKey = 1.8;
        const manager = new MemoryManager(config);

        // Act
        const snapshot = manager.getConfig('browser');

        // Assert
        expect({
          enableSlabArrayPooling: snapshot.enableSlabArrayPooling,
          environment: snapshot.environment,
          poolMaxPerBucket: snapshot.poolMaxPerBucket,
          slabGrowthFactor: snapshot.slabGrowthFactor,
          slabPoolMaxPerKey: snapshot.slabPoolMaxPerKey,
        }).toStrictEqual({
          enableSlabArrayPooling: true,
          environment: 'browser',
          poolMaxPerBucket: 12,
          slabGrowthFactor: MEMORY_SLAB_GROWTH_FACTOR_BROWSER,
          slabPoolMaxPerKey: 1,
        });
      });

      it('surfaces the configured node and browser soft memory targets', () => {
        // Arrange
        const configuredMemoryFlags = buildConfig();
        configuredMemoryFlags.browserMemoryBudgetMB = 64;
        configuredMemoryFlags.nodeHeapSoftLimitMB = 256;
        const manager = new MemoryManager(configuredMemoryFlags);

        // Act
        const snapshot = manager.getConfig('browser');

        // Assert
        expect({
          browserMemoryBudgetMB: snapshot.browserMemoryBudgetMB,
          nodeHeapSoftLimitMB: snapshot.nodeHeapSoftLimitMB,
        }).toStrictEqual({
          browserMemoryBudgetMB: 64,
          nodeHeapSoftLimitMB: 256,
        });
      });
    });
  });

  describe('given one pool is registered', () => {
    describe('when its stats are queried through the manager', () => {
      it('returns the registered pool snapshot', () => {
        // Arrange
        const manager = new MemoryManager(buildConfig());
        manager.registerPool('demoPool', {
          stats: () => ({ retained: 2 }),
        });

        // Act
        const snapshot = manager.getPoolStats('demoPool');

        // Assert
        expect(snapshot).toStrictEqual({ retained: 2 });
      });
    });
  });

  describe('given slab-array pooling is enabled through the manager allocator surface', () => {
    describe('when one released typed array is reacquired', () => {
      it('reuses the retained array and records pooled allocator stats', () => {
        // Arrange
        const config = buildConfig();
        config.enableSlabArrayPooling = true;
        const manager = new MemoryManager(config);
        const firstArray = manager.allocateTypedArray(
          'manager-owned-slab',
          Float32Array,
          4,
          Float32Array.BYTES_PER_ELEMENT,
        ) as Float32Array;

        manager.releaseTypedArray(
          'manager-owned-slab',
          Float32Array.BYTES_PER_ELEMENT,
          firstArray,
        );

        // Act
        const reacquiredArray = manager.allocateTypedArray(
          'manager-owned-slab',
          Float32Array,
          4,
          Float32Array.BYTES_PER_ELEMENT,
        ) as Float32Array;
        const snapshot = manager.getTypedArrayAllocationStats();

        // Assert
        expect({
          created: snapshot.pool['manager-owned-slab:4:4']?.created,
          fresh: snapshot.fresh,
          pooled: snapshot.pooled,
          reused: snapshot.pool['manager-owned-slab:4:4']?.reused,
          reusedArray: reacquiredArray === firstArray,
        }).toStrictEqual({
          created: 1,
          fresh: 1,
          pooled: 1,
          reused: 1,
          reusedArray: true,
        });
      });

      it('uses the constructor byte width by default when no explicit key width is provided', () => {
        // Arrange
        const config = buildConfig();
        config.enableSlabArrayPooling = true;
        const manager = new MemoryManager(config);
        const firstArray = manager.allocateTypedArray(
          'manager-default-width',
          Float32Array,
          2,
        ) as Float32Array;

        manager.releaseTypedArray(
          'manager-default-width',
          Float32Array.BYTES_PER_ELEMENT,
          firstArray,
        );

        // Act
        const reacquiredArray = manager.allocateTypedArray(
          'manager-default-width',
          Float32Array,
          2,
        ) as Float32Array;

        // Assert
        expect(reacquiredArray === firstArray).toBe(true);
      });
    });
  });

  describe('given init overrides mutate the memory flags', () => {
    describe('when teardown runs', () => {
      it('resets registered pools and restores the original config state', () => {
        // Arrange
        const config = buildConfig();
        let resetCount = 0;
        const manager = new MemoryManager(config);
        manager.registerPool('demoPool', {
          reset: () => {
            resetCount += 1;
          },
        });

        manager.init(
          {
            enableNodePooling: true,
            poolPrewarmCount: 9,
          },
          'node',
        );

        // Act
        manager.teardown();

        // Assert
        expect({
          enableNodePooling: config.enableNodePooling,
          poolPrewarmCount: config.poolPrewarmCount ?? null,
          resetCount,
        }).toStrictEqual({
          enableNodePooling: false,
          poolPrewarmCount: null,
          resetCount: 1,
        });
      });
    });
  });

  describe('given the manager retains typed arrays in its internal allocator', () => {
    describe('when teardown runs after one retained release', () => {
      it('clears the allocator state alongside the baseline config restore', () => {
        // Arrange
        const config = buildConfig();
        const manager = new MemoryManager(config);
        manager.init({
          enableSlabArrayPooling: true,
        });
        const retainedArray = manager.allocateTypedArray(
          'teardown-slab',
          Float32Array,
          3,
          Float32Array.BYTES_PER_ELEMENT,
        ) as Float32Array;

        manager.releaseTypedArray(
          'teardown-slab',
          Float32Array.BYTES_PER_ELEMENT,
          retainedArray,
        );

        // Act
        manager.teardown();
        const snapshot = manager.getTypedArrayAllocationStats();

        // Assert
        expect({
          enableSlabArrayPooling: config.enableSlabArrayPooling,
          fresh: snapshot.fresh,
          poolKeys: Object.keys(snapshot.pool).length,
          pooled: snapshot.pooled,
        }).toStrictEqual({
          enableSlabArrayPooling: false,
          fresh: 0,
          poolKeys: 0,
          pooled: 0,
        });
      });
    });
  });

  describe('given no pools are registered on a fresh manager', () => {
    describe('when teardown runs before init and init is later called twice with defaults', () => {
      it('treats the early teardown as a no-op and returns null for unknown pools', () => {
        // Arrange
        const manager = new MemoryManager(buildConfig());
        manager.teardown();

        // Act
        const firstSnapshot = manager.init();
        const secondSnapshot = manager.init();
        const missingPoolSnapshot = manager.getPoolStats('missingPool');
        manager.teardown();

        // Assert
        expect({
          firstEnvironment: firstSnapshot.environment,
          missingPoolSnapshot,
          secondEnvironment: secondSnapshot.environment,
        }).toStrictEqual({
          firstEnvironment: 'node',
          missingPoolSnapshot: null,
          secondEnvironment: 'node',
        });
      });
    });
  });
});