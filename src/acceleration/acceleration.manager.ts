/**
 * Lifecycle manager for the generic network acceleration layer.
 *
 * `AccelerationManager` owns acceleration state over time: construction,
 * initialization, enable/disable transitions, status queries, observer
 * notifications, and teardown. It delegates backend selection to
 * {@link autoEnableAcceleration} and wraps a {@link LifecycleAccelerationPolicy}
 * without owning worker-pool resources.
 */

import { autoEnableAcceleration } from './acceleration.orchestrator';
import type { AccelerationConfig } from './acceleration.types';
import type { AccelerationObserver } from './acceleration.observer';
import type { AccelerationStatus } from './acceleration.types';

/**
 * Options accepted by the {@link AccelerationManager} constructor.
 */
export interface AccelerationManagerOptions {
  /** Optional config overrides for thresholds, caps, and disable flags. */
  config?: Partial<AccelerationConfig>;

  /** Optional observer that receives backend-change events. */
  observer?: AccelerationObserver;
}

/**
 * Build the safe CPU fallback status used before initialization and after
 * teardown.
 *
 * @returns A status snapshot with CPU as the active mode.
 */
function createDefaultStatus(): AccelerationStatus {
  return {
    mode: 'cpu',
    gpu: { available: false, reason: 'Acceleration not initialized' },
    worker: {
      available: false,
      count: 0,
      reason: 'Acceleration not initialized',
    },
    cpu: { available: true },
  };
}

/**
 * Owns acceleration state over time for a single network or evaluation context.
 *
 * The manager provides a small lifecycle API: construct, initialize, enable,
 * disable, query status, re-evaluate after topology changes, and teardown. It
 * does not instantiate worker pools or GPU buffers directly; it only decides
 * which backend should be active and notifies observers of transitions.
 *
 * @example
 * ```ts
 * const manager = new AccelerationManager({ config: { backend: 'auto' } });
 * const status = await manager.init(2048);
 * console.log(status.mode); // 'gpu', 'worker', or 'cpu'
 * ```
 */
export class AccelerationManager {
  private readonly config: Partial<AccelerationConfig>;
  private readonly observer?: AccelerationObserver;
  private status: AccelerationStatus;
  private initialized = false;
  private enabled = false;

  /**
   * Create a new acceleration manager.
   *
   * @param options - Optional config overrides and observer.
   */
  constructor(options: AccelerationManagerOptions = {}) {
    this.config = options.config ?? {};
    this.observer = options.observer;
    this.status = createDefaultStatus();
  }

  /**
   * Initialize the manager and select the best available backend.
   *
   * Delegates to {@link autoEnableAcceleration} to probe the environment and
   * choose between GPU, worker, and CPU backends. The resolved status is cached
   * and returned by {@link getStatus}. Calling `init()` more than once returns
   * the existing status without re-probing or re-notifying observers.
   *
   * @param nodeCount - Number of nodes in the network being evaluated.
   * @returns The resolved acceleration status.
   */
  async init(nodeCount: number): Promise<AccelerationStatus> {
    if (this.initialized) {
      return this.status;
    }

    this.status = await autoEnableAcceleration({
      nodeCount,
      config: this.config,
      observer: this.observer,
    });
    this.initialized = true;
    return this.status;
  }

  /**
   * Return the current acceleration status.
   *
   * Before `init()` is called this returns a safe CPU fallback status. After
   * initialization it returns the exact status object resolved by `init()` or
   * the most recent `reEvaluate()` call.
   *
   * @returns Current acceleration status.
   */
  getStatus(): AccelerationStatus {
    return this.status;
  }

  /**
   * Mark the selected backend as active.
   *
   * @returns `true` when the manager has been initialized and the backend is
   *   now considered active; `false` if called before `init()`.
   */
  async enable(): Promise<boolean> {
    if (!this.initialized) {
      return false;
    }
    this.enabled = true;
    return true;
  }

  /**
   * Disable acceleration and fall back to CPU.
   *
   * Safe to call before initialization; in that case it is a no-op. When the
   * manager is initialized, the active mode is forced to `'cpu'` and observers
   * are notified of the backend change.
   */
  async disable(): Promise<void> {
    if (!this.initialized) {
      return;
    }

    const previous = this.status.mode;
    this.status = { ...this.status, mode: 'cpu' };
    this.enabled = false;
    this.observer?.onBackendChange?.({
      previous,
      current: 'cpu',
      reason: 'Acceleration disabled',
      timestamp: Date.now(),
    });
  }

  /**
   * Re-evaluate backend eligibility after a topology mutation.
   *
   * Re-runs the auto-enable decision with the updated node count and updates
   * the cached status. Observers are notified of the re-evaluation so callers
   * can track backend drift after structural changes.
   *
   * @param nodeCount - Updated network node count.
   * @returns The newly resolved acceleration status.
   */
  async reEvaluate(nodeCount: number): Promise<AccelerationStatus> {
    const previous = this.status.mode;
    this.status = await autoEnableAcceleration({
      nodeCount,
      config: this.config,
    });

    this.observer?.onBackendChange?.({
      previous,
      current: this.status.mode,
      reason: 'Backend re-evaluated after topology mutation',
      timestamp: Date.now(),
    });

    return this.status;
  }

  /**
   * Tear down the manager and reset it to a clean state.
   *
   * The cached status is reset to the default CPU fallback, and the
   * initialized and enabled flags are cleared.
   */
  async teardown(): Promise<void> {
    this.status = createDefaultStatus();
    this.initialized = false;
    this.enabled = false;
  }
}
