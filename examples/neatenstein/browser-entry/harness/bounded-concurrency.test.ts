/**
 * Red-phase contract tests for Step B4 item 11:
 * Simulation cost at scale — bounded concurrency for enemy inference.
 *
 * With 16-32 enemies, `Promise.all` serialization risk arises when enemies
 * exceed the worker count. This module provides bounded concurrency that
 * processes inference in batches of `workerCount` rather than unbounded
 * `Promise.all`.
 *
 * All tests in this file MUST fail (RED) until the B4 implementation lands.
 *
 * @module
 */

import { describe, expect, it } from '@jest/globals';

import * as BoundedConcurrencyModule from './bounded-concurrency';

/** Access unknown (future) exports on the bounded-concurrency module. */
const boundedConcurrency = BoundedConcurrencyModule as Record<string, unknown>;

/** Counter for concurrent invocations in concurrency tests. */
let concurrentCount = 0;
let maxConcurrent = 0;

/** Reset the concurrency tracking counters before each test group. */
function resetConcurrencyTracking(): void {
  concurrentCount = 0;
  maxConcurrent = 0;
}

/** Create a mock async task that tracks concurrent execution. */
function createTrackingTask(delayMs: number): () => Promise<number> {
  return () => {
    concurrentCount += 1;
    if (concurrentCount > maxConcurrent) {
      maxConcurrent = concurrentCount;
    }
    return new Promise<number>((resolve) => {
      setTimeout(() => {
        concurrentCount -= 1;
        resolve(concurrentCount);
      }, delayMs);
    });
  };
}

describe('B4.11: Bounded concurrency for enemy inference at scale', () => {
  describe('B4.11: runBoundedConcurrency export', () => {
    it('exports runBoundedConcurrency as a function', () => {
      expect(typeof boundedConcurrency.runBoundedConcurrency).toBe('function');
    });

    it('processes all tasks and returns their results in order', async () => {
      resetConcurrencyTracking();
      const tasks = [
        Promise.resolve(1),
        Promise.resolve(2),
        Promise.resolve(3),
      ];
      const results = await (
        boundedConcurrency.runBoundedConcurrency as (
          ...a: unknown[]
        ) => Promise<unknown[]>
      )(tasks, 2);
      expect(Array.isArray(results)).toBe(true);
      expect(results).toEqual([1, 2, 3]);
    });
  });

  describe('B4.11: concurrency limit enforcement', () => {
    it('never exceeds the workerCount concurrency limit', async () => {
      resetConcurrencyTracking();
      const taskFactories = Array.from({ length: 10 }, () =>
        createTrackingTask(20),
      );
      // Pass factory functions (not pre-started promises) so the bounded
      // concurrency runner calls them lazily and the concurrency limit is
      // actually enforced. Pre-starting via `.map(f => f())` would bypass
      // the limiter since all promises are already in flight.
      await (
        boundedConcurrency.runBoundedConcurrency as (
          ...a: unknown[]
        ) => Promise<unknown[]>
      )(taskFactories, 3);
      expect(maxConcurrent).toBeLessThanOrEqual(3);
    });

    it('processes more tasks than the concurrency limit without dropping any', async () => {
      resetConcurrencyTracking();
      const tasks = Array.from({ length: 20 }, (_, i) => Promise.resolve(i));
      const results = await (
        boundedConcurrency.runBoundedConcurrency as (
          ...a: unknown[]
        ) => Promise<unknown[]>
      )(tasks, 4);
      expect(results.length).toBe(20);
    });
  });

  describe('B4.11: batch processing', () => {
    it('exports createBatchProcessor as a function', () => {
      expect(typeof boundedConcurrency.createBatchProcessor).toBe('function');
    });

    it('creates a batch processor with the given batch size', () => {
      const processor = (
        boundedConcurrency.createBatchProcessor as (
          ...a: unknown[]
        ) => Record<string, unknown>
      )(4);
      expect(processor).toBeDefined();
      expect(typeof processor.process).toBe('function');
    });
  });
});
