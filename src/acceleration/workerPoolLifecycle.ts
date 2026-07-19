/**
 * Centralized worker pool lifecycle manager for the generic acceleration layer.
 *
 * `WorkerPoolLifecycle` owns creation, reuse, and teardown of worker pools.
 * It is instantiated through {@link createWorkerPoolLifecycle} and is never a
 * module-level singleton, so multiple acceleration contexts can each own an
 * isolated pool.
 *
 * The manager intentionally stays decoupled from `src/architecture/`: it only
 * knows how to create `Worker` instances, broadcast messages to every worker in
 * a pool, and terminate them. Higher-level inference logic (payload encoding,
 * task scheduling, and result aggregation) lives in the worker-payload layer.
 *
 * Background: Web Workers let scripts run computation on background threads
 * and communicate via message passing. See
 * [Web worker (Wikipedia)](https://en.wikipedia.org/wiki/Web_worker) and the
 * [WHATWG HTML Standard — Web Workers](https://html.spec.whatwg.org/multipage/workers.html)
 * for the underlying standard.
 */

import { resolveAccelerationConfig } from './acceleration.config';
import { DEFAULT_ACCELERATION_MAX_WORKERS } from './acceleration.constants';
import type { AccelerationConfig } from './acceleration.types';
import type {
  AccelerationBackendChangeEvent,
  AccelerationObserver,
} from './acceleration.observer';

/**
 * Minimal worker script URL used when no caller-specific script is supplied.
 *
 * This is a no-op data URL so the lifecycle manager can instantiate workers
 * without loading external resources. Real acceleration callers override this
 * with their own worker entry point before dispatching inference tasks.
 */
const DEFAULT_WORKER_SCRIPT_URL = 'data:,';

/**
 * Handle returned by {@link WorkerPoolLifecycle.create}. It exposes pool-level
 * broadcast and terminate operations and keeps the underlying worker set
 * private.
 *
 * @example
 * ```ts
 * const lifecycle = createWorkerPoolLifecycle({ maxWorkers: 2 });
 * const pool = await lifecycle.create();
 * pool.broadcast({ kind: 'eval', batch: [1, 2, 3] });
 * await pool.terminate();
 * ```
 */
export interface WorkerPoolHandle {
  /**
   * Broadcast a message to every worker in the pool.
   *
   * Messages are posted through the standard `Worker.postMessage` API, so the
   * payload must be
   * [structured-clone safe](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm).
   * Pass `Transferable` objects in `transferables` to move large buffers instead
   * of copying them.
   *
   * @param message - Message payload to post.
   * @param transferables - Optional array of owned `Transferable` objects to
   *   move rather than copy.
   *
   * @example
   * ```ts
   * pool.broadcast({ kind: 'ping' });
   * ```
   */
  broadcast(message: unknown, transferables?: Transferable[]): void;

  /**
   * Terminate every worker in the pool and release the handle.
   *
   * After termination the owning lifecycle stops reporting the pool as active
   * and {@link WorkerPoolLifecycle.reuse} returns `undefined`.
   *
   * @example
   * ```ts
   * await pool.terminate();
   * console.log(lifecycle.reuse()); // undefined
   * ```
   */
  terminate(): Promise<void>;
}

/**
 * Lifecycle manager for a scoped worker pool.
 *
 * Each instance tracks at most one active pool at a time. Creating a new pool
 * replaces any previously terminated pool, while `reuse` returns the active
 * pool handle when one exists.
 *
 * @example
 * ```ts
 * const lifecycle = createWorkerPoolLifecycle({ maxWorkers: 4 });
 * const pool = await lifecycle.create();
 * const samePool = lifecycle.reuse();
 * console.log(samePool === pool); // true
 * await lifecycle.dispose();
 * ```
 */
export interface WorkerPoolLifecycle {
  /**
   * Create a new worker pool handle.
   *
   * When workers are disabled by config, the returned handle owns no workers
   * and reports an active count of zero. Otherwise the pool size is capped by
   * `config.maxWorkers`.
   *
   * @returns A handle that can broadcast to and terminate its workers.
   * @throws Error when the lifecycle has already been disposed.
   *
   * @example
   * ```ts
   * const pool = await lifecycle.create();
   * console.log(lifecycle.activeCount()); // >= 1, <= config.maxWorkers
   * ```
   */
  create(): Promise<WorkerPoolHandle>;

  /**
   * Return the active pool handle if one exists and has not been terminated.
   *
   * @returns The current handle, or `undefined` when no active pool exists.
   *
   * @example
   * ```ts
   * const pool = lifecycle.reuse();
   * if (pool) {
   *   pool.broadcast({ kind: 'reuse' });
   * }
   * ```
   */
  reuse(): WorkerPoolHandle | undefined;

  /**
   * Terminate the active pool, if any, and mark the lifecycle as disposed.
   *
   * After disposal the active count is zero and `reuse` returns `undefined`.
   *
   * @example
   * ```ts
   * await lifecycle.dispose();
   * console.log(lifecycle.activeCount()); // 0
   * console.log(lifecycle.reuse()); // undefined
   * ```
   */
  dispose(): Promise<void>;

  /**
   * Report the number of workers in the currently active pool.
   *
   * @returns Worker count, or `0` when no pool is active.
   *
   * @example
   * ```ts
   * await lifecycle.create();
   * console.log(lifecycle.activeCount());
   * ```
   */
  activeCount(): number;
}

/**
 * Read the global `Worker` constructor.
 *
 * The constructor is read lazily so tests can install a mock before calling
 * `create()`.
 *
 * @internal
 */
function readGlobalWorker(): typeof Worker | undefined {
  return (globalThis as unknown as { Worker?: typeof Worker }).Worker;
}

/**
 * Notify the observer that the active backend has changed.
 *
 * @param observer - Optional observer to notify.
 * @param current - New active backend.
 * @param previous - Previous backend, or `null` on first transition.
 * @param reason - Human-readable reason for the transition.
 *
 * @internal
 */
function notifyBackendChange(
  observer: AccelerationObserver | undefined,
  current: AccelerationBackendChangeEvent['current'],
  previous: AccelerationBackendChangeEvent['previous'],
  reason: string,
): void {
  if (observer?.onBackendChange === undefined) {
    return;
  }

  observer.onBackendChange({
    previous,
    current,
    reason,
    timestamp:
      typeof performance !== 'undefined' ? performance.now() : Date.now(),
  });
}

/**
 * Create a scoped worker pool lifecycle manager.
 *
 * The returned instance is isolated from every other instance; it does not
 * share worker sets or state. The lifecycle manager only instantiates workers
 * when `create()` is called and `disableWorkers` is not set.
 *
 * The pool uses the standard Web Worker API so it can run in any environment
 * that exposes a global `Worker` constructor. When the constructor is missing
 * (for example in some test or Node environments), `create()` still returns a
 * valid handle that owns no workers and reports an active count of zero.
 *
 * @param config - Optional acceleration config overrides. Defaults are filled
 *   in from {@link resolveAccelerationConfig}.
 * @param observer - Optional observer that receives backend-change events.
 * @returns A new, scoped lifecycle instance.
 * @throws Error when `create()` is called after the lifecycle has been disposed.
 *
 * @example
 * ```ts
 * const lifecycle = createWorkerPoolLifecycle({ maxWorkers: 4 });
 * const pool = await lifecycle.create();
 * pool.broadcast({ kind: 'ping' });
 * await pool.terminate();
 * ```
 */
export function createWorkerPoolLifecycle(
  config: Partial<AccelerationConfig> = {},
  observer?: AccelerationObserver,
): WorkerPoolLifecycle {
  const resolvedConfig = resolveAccelerationConfig(config);
  let activeHandle: WorkerPoolHandle | undefined;
  let activeWorkers: Worker[] = [];
  let disposed = false;

  function setActive(
    handle: WorkerPoolHandle | undefined,
    workers: Worker[],
  ): void {
    activeHandle = handle;
    activeWorkers = workers;
  }

  /**
   * Build a handle that operates on the provided worker set.
   *
   * @param workers - Workers owned by the handle.
   * @returns A handle with broadcast and terminate methods.
   */
  function buildHandle(workers: Worker[]): WorkerPoolHandle {
    let terminated = false;

    return {
      broadcast(message: unknown, transferables?: Transferable[]): void {
        if (terminated) {
          return;
        }

        for (const worker of workers) {
          worker.postMessage(message, transferables ?? []);
        }
      },

      async terminate(): Promise<void> {
        if (terminated) {
          return;
        }

        terminated = true;
        for (const worker of workers) {
          worker.terminate();
        }

        if (activeHandle === this) {
          setActive(undefined, []);
          notifyBackendChange(
            observer,
            'cpu',
            'worker',
            'worker pool terminated',
          );
        }
      },
    };
  }

  return {
    async create(): Promise<WorkerPoolHandle> {
      if (disposed) {
        throw new Error('WorkerPoolLifecycle has been disposed');
      }

      // Dispose of any previous active workers before creating a new pool.
      if (activeHandle !== undefined) {
        await activeHandle.terminate();
      }

      // When workers are explicitly disabled, return an empty handle that
      // keeps the lifecycle API contract intact without spawning threads.
      if (resolvedConfig.disableWorkers) {
        const emptyHandle = buildHandle([]);
        setActive(emptyHandle, []);
        return emptyHandle;
      }

      // Read the Worker constructor lazily so tests can install a mock before
      // calling create(). If the constructor is missing, return an empty handle.
      const WorkerCtor = readGlobalWorker();
      if (WorkerCtor === undefined) {
        const emptyHandle = buildHandle([]);
        setActive(emptyHandle, []);
        return emptyHandle;
      }

      // Create the worker set, capped by maxWorkers.
      const poolSize = Math.max(
        1,
        Math.min(resolvedConfig.maxWorkers!, DEFAULT_ACCELERATION_MAX_WORKERS),
      );
      const workers: Worker[] = [];
      for (let index = 0; index < poolSize; index++) {
        workers.push(new WorkerCtor(DEFAULT_WORKER_SCRIPT_URL));
      }

      // Build and register the handle, then notify observers of the backend change.
      const handle = buildHandle(workers);
      setActive(handle, workers);
      notifyBackendChange(observer, 'worker', null, 'worker pool created');

      return handle;
    },

    reuse(): WorkerPoolHandle | undefined {
      return activeHandle;
    },

    async dispose(): Promise<void> {
      if (disposed) {
        return;
      }

      disposed = true;
      if (activeHandle !== undefined) {
        await activeHandle.terminate();
      }
    },

    activeCount(): number {
      return activeWorkers.length;
    },
  };
}
