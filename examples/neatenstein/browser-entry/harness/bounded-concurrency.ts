/**
 * Bounded concurrency helper for simulation cost at scale in the Neatenstein
 * enemy AI inference path.
 *
 * Processes enemy inference in batches of `workerCount` rather than unbounded
 * `Promise.all` when enemies exceed the worker count.
 *
 * Tests that need synchronous timer semantics should use `jest.useFakeTimers`
 * in the test file — this module does NOT mutate global timers.
 *
 * @module
 */

/**
 * A task can be either a Promise (already started) or a function that returns
 * a Promise (lazy factory). Functions are called only when a worker slot is
 * available, enabling true concurrency limiting.
 */
type Task<T> = Promise<T> | (() => Promise<T>);

/**
 * Runs tasks with bounded concurrency, processing at most `workerCount` tasks
 * simultaneously. Results are returned in the original task order.
 *
 * If a task is a function (factory), it is called lazily when a worker slot
 * opens. If a task is already a Promise, it is awaited directly.
 *
 * @param tasks Array of tasks (Promises or factory functions).
 * @param workerCount Maximum concurrent tasks.
 * @returns Array of results in the same order as the input tasks.
 */
export async function runBoundedConcurrency<T>(
  tasks: Task<T>[],
  workerCount: number,
): Promise<T[]> {
  const results = new Array<T>(tasks.length);
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (nextIndex < tasks.length) {
      const currentIndex = nextIndex;
      nextIndex++;
      const task = tasks[currentIndex];
      const promise =
        typeof task === 'function' ? (task as () => Promise<T>)() : task;
      results[currentIndex] = await promise;
    }
  }

  const workers: Promise<void>[] = [];
  const numWorkers = Math.min(workerCount, tasks.length);
  for (let i = 0; i < numWorkers; i++) {
    workers.push(worker());
  }
  await Promise.all(workers);
  return results;
}

/**
 * Batch processor that processes arrays of tasks in fixed-size batches.
 */
export interface BatchProcessor {
  /**
   * Processes an array of tasks in batches of the configured size.
   *
   * @param tasks Array of tasks (Promises or factory functions).
   * @returns Array of results in the original order.
   */
  process: <T>(tasks: Task<T>[]) => Promise<T[]>;
}

/**
 * Creates a batch processor with the given batch size.
 *
 * @param batchSize The number of tasks to process concurrently per batch.
 * @returns A batch processor with a `process` method.
 */
export function createBatchProcessor(batchSize: number): BatchProcessor {
  return {
    process<T>(tasks: Task<T>[]): Promise<T[]> {
      return runBoundedConcurrency(tasks, batchSize);
    },
  };
}