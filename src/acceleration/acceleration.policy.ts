/**
 * Policy-driven backend selection for the generic network acceleration layer.
 *
 * `AccelerationPolicy` encodes the decision rules that map a detected
 * {@link AccelerationStatus} to a concrete backend choice. It is intentionally
 * separate from environment detection so callers can swap or extend policy
 * behavior without changing how capabilities are probed.
 *
 * `LifecycleAccelerationPolicy` extends the base policy with topology-dirty
 * tracking so callers such as NGE can re-verify backend eligibility after
 * structural mutations. The policy itself only records the dirty flag; the
 * actual re-verification is performed on the next `evaluate()` call.
 */

import { detectAcceleration } from './acceleration.detect';
import { resolveAccelerationMode } from './acceleration.resolve';
import type { AccelerationConfig } from './acceleration.types';
import type { AccelerationObserver } from './acceleration.observer';
import type {
  AccelerationMode,
  AccelerationStatus,
  BackendMode,
} from './acceleration.types';

export type { AccelerationConfig, AccelerationStatus, BackendMode };

/**
 * Concrete backend choice returned by an {@link AccelerationPolicy}.
 *
 * The reason string is meant for diagnostics and telemetry; it should explain
 * why the policy selected the reported backend.
 */
export interface AccelerationDecision {
  /** Selected backend mode. */
  backend: AccelerationMode;

  /** Human-readable explanation of the decision. */
  reason: string;
}

/**
 * Default policy for mapping a detected acceleration status to a backend choice.
 *
 * The policy supports explicit backend overrides (`cpu`, `gpu`, `worker`) and
 * an `auto` mode that follows the detected status. A shared default instance is
 * available via {@link AccelerationPolicy.default} for callers that do not need
 * custom configuration.
 *
 * @example
 * ```ts
 * const policy = new AccelerationPolicy({ backend: 'gpu' });
 * const decision = policy.decide(status);
 * console.log(decision.backend); // 'gpu' or the resolved fallback
 * ```
 */
export class AccelerationPolicy {
  /** Shared default policy instance for callers without custom configuration. */
  static default = new AccelerationPolicy();

  /**
   * Create a new acceleration policy.
   *
   * @param config - Optional default configuration. Explicit overrides passed
   *   to {@link decide} take precedence over this stored config.
   */
  constructor(protected readonly config: Partial<AccelerationConfig> = {}) {}

  /**
   * Decide which backend should be used for the given acceleration status.
   *
   * The method honors explicit backend preferences and falls back to the mode
   * reported by environment detection when no override is supplied.
   *
   * @param status - Detected acceleration status with capability reports.
   * @param network - Optional network context reserved for future policy use.
   * @param override - Optional backend override for this decision.
   * @returns A concrete backend decision with a diagnostic reason.
   */
  decide(
    status: AccelerationStatus,
    network?: unknown,
    override?: Partial<AccelerationConfig>,
  ): AccelerationDecision {
    const backend = override?.backend ?? this.config.backend ?? 'auto';
    const resolved = resolveAccelerationMode(status, { backend });

    return {
      backend: resolved.mode,
      reason: `Policy selected ${resolved.mode} backend${
        backend === 'auto'
          ? ' from environment detection'
          : ` via '${backend}' override`
      }`,
    };
  }
}

/**
 * Lifecycle-aware policy that re-evaluates backend eligibility after mutations.
 *
 * In addition to the base {@link AccelerationPolicy.decide} behavior, this
 * policy exposes `onMutated()` / `clearDirty()` hooks and an `evaluate()`
 * helper that runs environment detection and mode resolution in one call.
 * Topology dirty flags let callers know when a structural change may have
 * invalidated the current backend choice.
 *
 * @example
 * ```ts
 * const policy = new LifecycleAccelerationPolicy();
 * policy.onMutated();
 * const status = policy.evaluate({ backend: 'auto' }, 2048);
 * console.log(status.mode);
 * ```
 */
export class LifecycleAccelerationPolicy extends AccelerationPolicy {
  /** True after `onMutated()` until `clearDirty()` is called. */
  isTopologyDirty = false;

  /** True after `onMutated()` until `clearDirty()` is called. */
  needsReverification = false;

  /**
   * Mark the policy as needing backend re-verification.
   *
   * Call this after a network mutation that could change backend eligibility.
   */
  onMutated(): void {
    this.isTopologyDirty = true;
    this.needsReverification = true;
  }

  /**
   * Clear the topology-dirty and re-verification flags.
   *
   * Call this once backend eligibility has been re-confirmed.
   */
  clearDirty(): void {
    this.isTopologyDirty = false;
    this.needsReverification = false;
  }

  /**
   * Detect capabilities and resolve the active backend in one call.
   *
   * The method merges any constructor-supplied defaults with the partial config
   * passed here, then probes the environment and applies policy resolution.
   * When the resolved mode differs from the detected mode, the observer is
   * notified through `onBackendChange`.
   *
   * @param partial - Optional user overrides, including `backend` and
   *   `hasActiveWorker`.
   * @param nodeCount - Number of nodes in the network being evaluated.
   * @param observer - Optional observer for backend-change telemetry.
   * @returns A resolved acceleration status with the active mode.
   */
  evaluate(
    partial: Partial<AccelerationConfig> = {},
    nodeCount: number = 0,
    observer?: AccelerationObserver,
  ): AccelerationStatus {
    const config = { ...this.config, ...partial };
    const detected = detectAcceleration(config, nodeCount);

    return resolveAccelerationMode(detected, {
      backend: config.backend,
      hasActiveWorker: config.hasActiveWorker,
      observer,
    });
  }
}
