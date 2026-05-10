import Network from '../network/network';
import { config } from '../../config';
import { activationArrayPool } from './activationArrayPool';

type RuntimeActivationArrayPool = {
  scheduleCompactionAfterLargePrune: (prunedConnectionCount: number) => void;
};

function createIdentityActivation(): (
  value: number,
  derivative?: boolean,
) => number {
  return (value, derivative = false) => (derivative ? 1 : value);
}

function configureDeterministicOutputNode(network: Network): void {
  const outputNode = network.nodes.find(
    (nodeEntry) => nodeEntry.type === 'output',
  );

  if (!outputNode) {
    throw new Error('Expected one output node to exist.');
  }

  outputNode.bias = 0;
  outputNode.squash = createIdentityActivation();
}

describe('activationArrayPool direct coverage chapter', () => {
  beforeEach(() => {
    config.float32Mode = false;
    activationArrayPool.clear();
    activationArrayPool.setMaxPerBucket(Number.POSITIVE_INFINITY);
  });

  afterEach(() => {
    config.float32Mode = false;
    activationArrayPool.clear();
    activationArrayPool.setMaxPerBucket(Number.POSITIVE_INFINITY);
  });

  describe('given float32 mode reuses a retained typed buffer', () => {
    it('zero-fills the recycled Float32Array before reuse', () => {
      // Arrange
      config.float32Mode = true;
      const retainedArray = activationArrayPool.acquire(4);

      if (!(retainedArray instanceof Float32Array)) {
        throw new Error(
          'Expected a Float32Array when float32 mode is enabled.',
        );
      }

      retainedArray[0] = 7;
      activationArrayPool.release(retainedArray);

      // Act
      const reusedArray = activationArrayPool.acquire(4);

      // Assert
      expect({
        firstValue: reusedArray[0],
        sameReference: reusedArray === retainedArray,
      }).toEqual({
        firstValue: 0,
        sameReference: true,
      });
    });
  });

  describe('given a released Float64Array is reused from the pool', () => {
    it('zero-fills the recycled Float64Array before reuse', () => {
      // Arrange
      const retainedArray = new Float64Array([9, 8]);
      activationArrayPool.release(retainedArray);

      // Act
      const reusedArray = activationArrayPool.acquire(2);

      // Assert
      expect({
        firstValue: reusedArray[0],
        sameReference: reusedArray === retainedArray,
      }).toEqual({
        firstValue: 0,
        sameReference: true,
      });
    });
  });

  describe('given a bucket is already at its retention cap', () => {
    it('drops the extra released array instead of growing the bucket', () => {
      // Arrange
      activationArrayPool.setMaxPerBucket(1);
      const firstArray = activationArrayPool.acquire(2);
      const secondArray = activationArrayPool.acquire(2);
      activationArrayPool.release(firstArray);

      // Act
      activationArrayPool.release(secondArray);

      // Assert
      expect(activationArrayPool.bucketSize(2)).toBe(1);
    });
  });

  describe('given stats are requested after one creation and one reuse', () => {
    it('returns the current pool counters', () => {
      // Arrange
      const createdArray = activationArrayPool.acquire(3);
      activationArrayPool.release(createdArray);
      activationArrayPool.acquire(3);

      // Act
      const poolStats = activationArrayPool.stats();

      // Assert
      expect(poolStats).toEqual({
        bucketCount: 1,
        compactionCount: 0,
        created: 1,
        retainedArrayCount: 0,
        reused: 1,
        trimmedArrays: 0,
        trimmedBuckets: 0,
      });
    });
  });

  describe('given compaction runs after three retained buckets were created', () => {
    it('reports the compaction counters and retained array count', () => {
      // Arrange
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);
      const refreshedBucketArray = activationArrayPool.acquire(2);
      activationArrayPool.release(refreshedBucketArray);

      // Act
      activationArrayPool.compact(2);

      // Assert
      expect(activationArrayPool.stats()).toEqual({
        bucketCount: 2,
        compactionCount: 1,
        created: 3,
        retainedArrayCount: 2,
        reused: 1,
        trimmedArrays: 1,
        trimmedBuckets: 1,
      });
    });
  });

  describe('given the retained bucket cap is lowered after prewarm', () => {
    it('compacts the retained arrays immediately through setMaxPerBucket', () => {
      // Arrange
      activationArrayPool.prewarm(3, 3);

      // Act
      activationArrayPool.setMaxPerBucket(1);

      // Assert
      expect(activationArrayPool.stats()).toEqual({
        bucketCount: 1,
        compactionCount: 1,
        created: 3,
        retainedArrayCount: 1,
        reused: 0,
        trimmedArrays: 2,
        trimmedBuckets: 0,
      });
    });
  });

  describe('given the retained bucket cap is lowered to zero after prewarm', () => {
    it('drops the emptied bucket during compaction', () => {
      // Arrange
      activationArrayPool.prewarm(3, 2);

      // Act
      activationArrayPool.setMaxPerBucket(0);

      // Assert
      expect(activationArrayPool.stats()).toEqual({
        bucketCount: 0,
        compactionCount: 1,
        created: 2,
        retainedArrayCount: 0,
        reused: 0,
        trimmedArrays: 2,
        trimmedBuckets: 1,
      });
    });
  });

  describe('given compaction does not need to evict anything', () => {
    it('keeps the compaction counters unchanged', () => {
      // Arrange
      activationArrayPool.prewarm(5, 1);

      // Act
      activationArrayPool.compact(2);

      // Assert
      expect(activationArrayPool.stats()).toEqual({
        bucketCount: 1,
        compactionCount: 0,
        created: 1,
        retainedArrayCount: 1,
        reused: 0,
        trimmedArrays: 0,
        trimmedBuckets: 0,
      });
    });
  });

  describe('given pooled raw activation compacts between repeated runs', () => {
    it('keeps the activation output identical for the same network and input', () => {
      // Arrange
      const inputValue = Math.PI / 7;
      const connectionWeight = Math.E / 11;
      const pooledNetwork = new Network(1, 1, {
        activationPrecision: 'f64',
        enforceAcyclic: true,
        returnTypedActivations: true,
        reuseActivationArrays: true,
        seed: 902,
      });
      configureDeterministicOutputNode(pooledNetwork);
      pooledNetwork.connections[0].weight = connectionWeight;
      const firstOutput = Array.from(pooledNetwork.activateRaw([inputValue]));
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);

      // Act
      activationArrayPool.compact(1);
      const secondOutput = Array.from(pooledNetwork.activateRaw([inputValue]));

      // Assert
      expect(secondOutput).toEqual(firstOutput);
    });
  });

  describe('given a large prune schedules deferred compaction in Node', () => {
    it('trims colder retained buckets on the next idle tick instead of immediately', async () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);
      runtimePool.scheduleCompactionAfterLargePrune(64);
      const statsBeforeIdleTick = activationArrayPool.stats();

      // Act
      await new Promise<void>((resolve) => {
        setImmediate(resolve);
      });
      const statsAfterIdleTick = activationArrayPool.stats();

      // Assert
      expect({
        after: statsAfterIdleTick,
        before: statsBeforeIdleTick,
      }).toEqual({
        after: {
          bucketCount: 2,
          compactionCount: 1,
          created: 3,
          retainedArrayCount: 2,
          reused: 0,
          trimmedArrays: 1,
          trimmedBuckets: 1,
        },
        before: {
          bucketCount: 3,
          compactionCount: 0,
          created: 3,
          retainedArrayCount: 3,
          reused: 0,
          trimmedArrays: 0,
          trimmedBuckets: 0,
        },
      });
    });
  });

  describe('given only one retained bucket exists during a large prune', () => {
    it('skips deferred compaction because there is no colder bucket to trim', async () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      activationArrayPool.prewarm(3, 1);
      runtimePool.scheduleCompactionAfterLargePrune(64);

      // Act
      await new Promise<void>((resolve) => {
        setImmediate(resolve);
      });

      // Assert
      expect(activationArrayPool.stats()).toEqual({
        bucketCount: 1,
        compactionCount: 0,
        created: 1,
        retainedArrayCount: 1,
        reused: 0,
        trimmedArrays: 0,
        trimmedBuckets: 0,
      });
    });
  });

  describe('given a deferred compaction pass is already pending', () => {
    it('coalesces duplicate large-prune notifications into one compaction pass', async () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);
      runtimePool.scheduleCompactionAfterLargePrune(64);
      runtimePool.scheduleCompactionAfterLargePrune(64);

      // Act
      await new Promise<void>((resolve) => {
        setImmediate(resolve);
      });

      // Assert
      expect(activationArrayPool.stats()).toEqual({
        bucketCount: 2,
        compactionCount: 1,
        created: 3,
        retainedArrayCount: 2,
        reused: 0,
        trimmedArrays: 1,
        trimmedBuckets: 1,
      });
    });
  });

  describe('given Node falls back to setTimeout for idle scheduling', () => {
    it('still compacts colder retained buckets', () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      const originalSetImmediate = Reflect.get(globalThis, 'setImmediate');
      const originalSetTimeout = Reflect.get(globalThis, 'setTimeout');
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);

      Reflect.set(globalThis, 'setImmediate', undefined);
      Reflect.set(
        globalThis,
        'setTimeout',
        ((callback: TimerHandler) => {
          if (typeof callback === 'function') {
            callback();
          }

          return 0;
        }) as unknown as typeof setTimeout,
      );

      try {
        // Act
        runtimePool.scheduleCompactionAfterLargePrune(64);

        // Assert
        expect(activationArrayPool.stats()).toEqual({
          bucketCount: 2,
          compactionCount: 1,
          created: 3,
          retainedArrayCount: 2,
          reused: 0,
          trimmedArrays: 1,
          trimmedBuckets: 1,
        });
      } finally {
        Reflect.set(globalThis, 'setImmediate', originalSetImmediate);
        Reflect.set(globalThis, 'setTimeout', originalSetTimeout);
      }
    });
  });

  describe('given no idle scheduler is available', () => {
    it('leaves the retained buckets unchanged', () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      const originalSetImmediate = Reflect.get(globalThis, 'setImmediate');
      const originalSetTimeout = Reflect.get(globalThis, 'setTimeout');
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);

      Reflect.set(globalThis, 'setImmediate', undefined);
      Reflect.set(globalThis, 'setTimeout', undefined);

      try {
        // Act
        runtimePool.scheduleCompactionAfterLargePrune(64);

        // Assert
        expect(activationArrayPool.stats()).toEqual({
          bucketCount: 3,
          compactionCount: 0,
          created: 3,
          retainedArrayCount: 3,
          reused: 0,
          trimmedArrays: 0,
          trimmedBuckets: 0,
        });
      } finally {
        Reflect.set(globalThis, 'setImmediate', originalSetImmediate);
        Reflect.set(globalThis, 'setTimeout', originalSetTimeout);
      }
    });
  });

  describe('given browser cooperative compaction uses idle callbacks', () => {
    it('evicts one cold bucket per idle slice until the retained budget is reached', () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      const originalWindow = Reflect.get(globalThis, 'window');
      const originalRequestIdleCallback = Reflect.get(
        globalThis,
        'requestIdleCallback',
      );
      const idleCallbacks: Array<() => void> = [];
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);
      activationArrayPool.prewarm(8, 1);

      Reflect.set(globalThis, 'window', {});
      Reflect.set(
        globalThis,
        'requestIdleCallback',
        ((callback: IdleRequestCallback) => {
          idleCallbacks.push(() => {
            callback({
              didTimeout: false,
              timeRemaining: () => 10,
            } as IdleDeadline);
          });

          return idleCallbacks.length;
        }) as unknown as typeof requestIdleCallback,
      );

      try {
        runtimePool.scheduleCompactionAfterLargePrune(64);
        const statsBeforeIdleSlice = activationArrayPool.stats();
        idleCallbacks.shift()?.();
        const statsAfterFirstIdleSlice = activationArrayPool.stats();
        idleCallbacks.shift()?.();
        const statsAfterSecondIdleSlice = activationArrayPool.stats();

        // Assert
        expect({
          afterFirst: statsAfterFirstIdleSlice,
          afterSecond: statsAfterSecondIdleSlice,
          before: statsBeforeIdleSlice,
        }).toEqual({
          afterFirst: {
            bucketCount: 3,
            compactionCount: 1,
            created: 4,
            retainedArrayCount: 3,
            reused: 0,
            trimmedArrays: 1,
            trimmedBuckets: 1,
          },
          afterSecond: {
            bucketCount: 2,
            compactionCount: 2,
            created: 4,
            retainedArrayCount: 2,
            reused: 0,
            trimmedArrays: 2,
            trimmedBuckets: 2,
          },
          before: {
            bucketCount: 4,
            compactionCount: 0,
            created: 4,
            retainedArrayCount: 4,
            reused: 0,
            trimmedArrays: 0,
            trimmedBuckets: 0,
          },
        });
      } finally {
        Reflect.set(globalThis, 'window', originalWindow);
        Reflect.set(
          globalThis,
          'requestIdleCallback',
          originalRequestIdleCallback,
        );
      }
    });
  });

  describe('given browser cooperative compaction falls back to timer turns', () => {
    it('evicts one cold bucket per fallback timer slice', () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      const originalWindow = Reflect.get(globalThis, 'window');
      const originalRequestIdleCallback = Reflect.get(
        globalThis,
        'requestIdleCallback',
      );
      const originalSetTimeout = Reflect.get(globalThis, 'setTimeout');
      const timerCallbacks: Array<() => void> = [];
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);
      activationArrayPool.prewarm(8, 1);

      Reflect.set(globalThis, 'window', {});
      Reflect.set(globalThis, 'requestIdleCallback', undefined);
      Reflect.set(
        globalThis,
        'setTimeout',
        ((callback: TimerHandler) => {
          if (typeof callback === 'function') {
            timerCallbacks.push(callback as () => void);
          }

          return timerCallbacks.length;
        }) as unknown as typeof setTimeout,
      );

      try {
        runtimePool.scheduleCompactionAfterLargePrune(64);
        const statsBeforeTimerSlice = activationArrayPool.stats();
        timerCallbacks.shift()?.();
        const statsAfterFirstTimerSlice = activationArrayPool.stats();
        timerCallbacks.shift()?.();
        const statsAfterSecondTimerSlice = activationArrayPool.stats();

        // Assert
        expect({
          afterFirst: statsAfterFirstTimerSlice,
          afterSecond: statsAfterSecondTimerSlice,
          before: statsBeforeTimerSlice,
        }).toEqual({
          afterFirst: {
            bucketCount: 3,
            compactionCount: 1,
            created: 4,
            retainedArrayCount: 3,
            reused: 0,
            trimmedArrays: 1,
            trimmedBuckets: 1,
          },
          afterSecond: {
            bucketCount: 2,
            compactionCount: 2,
            created: 4,
            retainedArrayCount: 2,
            reused: 0,
            trimmedArrays: 2,
            trimmedBuckets: 2,
          },
          before: {
            bucketCount: 4,
            compactionCount: 0,
            created: 4,
            retainedArrayCount: 4,
            reused: 0,
            trimmedArrays: 0,
            trimmedBuckets: 0,
          },
        });
      } finally {
        Reflect.set(globalThis, 'window', originalWindow);
        Reflect.set(
          globalThis,
          'requestIdleCallback',
          originalRequestIdleCallback,
        );
        Reflect.set(globalThis, 'setTimeout', originalSetTimeout);
      }
    });
  });

  describe('given browser has no cooperative scheduler available', () => {
    it('leaves the retained buckets unchanged', () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      const originalWindow = Reflect.get(globalThis, 'window');
      const originalRequestIdleCallback = Reflect.get(
        globalThis,
        'requestIdleCallback',
      );
      const originalSetTimeout = Reflect.get(globalThis, 'setTimeout');
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);

      Reflect.set(globalThis, 'window', {});
      Reflect.set(globalThis, 'requestIdleCallback', undefined);
      Reflect.set(globalThis, 'setTimeout', undefined);

      try {
        // Act
        runtimePool.scheduleCompactionAfterLargePrune(64);

        // Assert
        expect(activationArrayPool.stats()).toEqual({
          bucketCount: 3,
          compactionCount: 0,
          created: 3,
          retainedArrayCount: 3,
          reused: 0,
          trimmedArrays: 0,
          trimmedBuckets: 0,
        });
      } finally {
        Reflect.set(globalThis, 'window', originalWindow);
        Reflect.set(
          globalThis,
          'requestIdleCallback',
          originalRequestIdleCallback,
        );
        Reflect.set(globalThis, 'setTimeout', originalSetTimeout);
      }
    });
  });

  describe('given browser reaches the retained budget before a queued slice runs', () => {
    it('clears the scheduled state without running an extra compaction pass', () => {
      // Arrange
      const runtimePool =
        activationArrayPool as unknown as RuntimeActivationArrayPool;
      const originalWindow = Reflect.get(globalThis, 'window');
      const originalRequestIdleCallback = Reflect.get(
        globalThis,
        'requestIdleCallback',
      );
      const idleCallbacks: Array<() => void> = [];
      activationArrayPool.prewarm(2, 1);
      activationArrayPool.prewarm(4, 1);
      activationArrayPool.prewarm(6, 1);

      Reflect.set(globalThis, 'window', {});
      Reflect.set(
        globalThis,
        'requestIdleCallback',
        ((callback: IdleRequestCallback) => {
          idleCallbacks.push(() => {
            callback({
              didTimeout: false,
              timeRemaining: () => 10,
            } as IdleDeadline);
          });

          return idleCallbacks.length;
        }) as unknown as typeof requestIdleCallback,
      );

      try {
        runtimePool.scheduleCompactionAfterLargePrune(64);
        activationArrayPool.compact(2);
        const statsAfterManualCompaction = activationArrayPool.stats();
        idleCallbacks.shift()?.();
        const statsAfterIdleCallback = activationArrayPool.stats();
        runtimePool.scheduleCompactionAfterLargePrune(64);

        // Assert
        expect({
          queuedCallbackCountAfterReschedule: idleCallbacks.length,
          statsAfterIdleCallback,
          statsAfterManualCompaction,
        }).toEqual({
          queuedCallbackCountAfterReschedule: 1,
          statsAfterIdleCallback: {
            bucketCount: 2,
            compactionCount: 1,
            created: 3,
            retainedArrayCount: 2,
            reused: 0,
            trimmedArrays: 1,
            trimmedBuckets: 1,
          },
          statsAfterManualCompaction: {
            bucketCount: 2,
            compactionCount: 1,
            created: 3,
            retainedArrayCount: 2,
            reused: 0,
            trimmedArrays: 1,
            trimmedBuckets: 1,
          },
        });
      } finally {
        Reflect.set(globalThis, 'window', originalWindow);
        Reflect.set(
          globalThis,
          'requestIdleCallback',
          originalRequestIdleCallback,
        );
      }
    });
  });

  describe('given float32 mode prewarms a bucket', () => {
    it('retains a Float32Array for the later acquisition', () => {
      // Arrange
      config.float32Mode = true;
      activationArrayPool.prewarm(5, 1);

      // Act
      const prewarmedArray = activationArrayPool.acquire(5);

      // Assert
      expect(prewarmedArray instanceof Float32Array).toBe(true);
    });
  });

  describe('given an invalid cap and normalized prewarm counts are provided', () => {
    it('ignores the invalid cap, skips the negative request, and floors the fractional request', () => {
      // Arrange
      activationArrayPool.setMaxPerBucket(-1);
      activationArrayPool.prewarm(3, -2);

      // Act
      activationArrayPool.prewarm(3, 2.8);

      // Assert
      expect(activationArrayPool.bucketSize(3)).toBe(2);
    });
  });

  describe('given a missing bucket size is requested', () => {
    it('returns zero', () => {
      // Arrange
      const missingBucketSize = 99;

      // Act
      const retainedBucketSize =
        activationArrayPool.bucketSize(missingBucketSize);

      // Assert
      expect(retainedBucketSize).toBe(0);
    });
  });
});
