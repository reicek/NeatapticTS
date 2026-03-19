import type Network from '../../../architecture/network';
import {
  ensureMinHiddenNodes as ensureMinHiddenNodesImpl,
  ensureNoDeadEnds as ensureNoDeadEndsImpl,
} from '../../mutation/mutation';
import { computeMinimumHiddenSize } from '../../mutation/repair/mutation.min-hidden';
import type { NeatControllerForMutation } from '../../mutation/shared/mutation.types';

type NeatMaintenanceOptions = NeatControllerForMutation['options'] & {
  minHidden?: number;
  minHiddenMultiplier?: number;
};

/**
 * Public maintenance facade helpers for the stable `Neat` entrypoint.
 *
 * This chapter is the public maintenance-policy bridge for callers who want to
 * keep one concrete network structurally viable without stepping down into the
 * broader mutation orchestration chapter. The heavy repair mechanics still live
 * under `mutation/`, but the stable `Neat` facade preserves three small wrapper
 * behaviors that users already expect: read the current hidden-node floor,
 * enforce that floor when needed, and patch obvious dead ends without turning
 * routine maintenance into a full mutation pass.
 *
 * Keeping those wrappers in `maintenance/facade/` preserves two boundaries at
 * once. The mutation chapter stays responsible for topology-changing operator
 * logic and repair mechanics, while this facade remains responsible for the
 * stable public entrypoints and their intentionally conservative semantics.
 * Read this file when you want the user-facing maintenance story rather than
 * the lower-level graph-editing implementation details.
 *
 * Invariant: this boundary only maintains baseline structural viability for a
 * single network. It does not change mutation operator selection, speciation,
 * or public import paths.
 *
 * The three helpers form one small policy flow:
 *
 * - `getMinimumHiddenSize()` explains the current target,
 * - `ensureMinHiddenNodes()` enforces that target,
 * - `ensureNoDeadEnds()` repairs the most obvious connectivity failures after
 *   edits or imports.
 */

/**
 * Narrow `Neat` host surface required by the public maintenance facade.
 *
 * This boundary keeps the contract focused on topology-maintenance state:
 * mutation constraints, innovation bookkeeping, endpoint counts, and the
 * legacy helper used to compute a minimum hidden-node target.
 *
 * The facade does not need the full controller. It only needs enough state to
 * forward repair work into the mutation layer while preserving the stable
 * wrapper semantics of the public `Neat` API. Keeping the contract narrow makes
 * it clearer that maintenance is a thin policy surface, not a second mutation
 * facade.
 */
export interface NeatMaintenanceFacadeHost extends Omit<
  NeatControllerForMutation,
  'options'
> {
  input: number;
  output: number;
  options: NeatMaintenanceOptions;
  getMinimumHiddenSize: (multiplierOverride?: number) => number;
}

/**
 * Ensure a network satisfies the configured minimum hidden-node policy.
 *
 * This is the enforcing half of the maintenance policy. The wrapper first uses
 * the current `Neat` configuration as the source of truth for how much hidden
 * capacity a network should keep, then delegates the actual graph edits to the
 * mutation repair layer.
 *
 * The underlying helper may add hidden nodes and wire them into the graph so
 * later mutation and evaluation steps start from a minimally viable structure.
 * That makes this function more than a tiny threshold check: it is a targeted
 * topology-maintenance pass that preserves user intent without exposing the
 * larger mutation chapter as the primary public surface for routine upkeep.
 *
 * @param host - `Neat` instance exposing mutation constraints and innovation tables.
 * @param network - Network whose hidden-node floor should be enforced.
 * @param multiplierOverride - Optional one-off multiplier overriding the configured policy.
 * @returns Promise that resolves after any required topology repair finishes.
 *
 * @example
 * ```ts
 * const minimumHidden = neat.getMinimumHiddenSize();
 * await neat.ensureMinHiddenNodes(network);
 * console.log(minimumHidden <= network.nodes.length);
 * ```
 */
export async function ensureMinHiddenNodes(
  host: NeatMaintenanceFacadeHost,
  network: Network,
  multiplierOverride?: number,
): Promise<void> {
  return ensureMinHiddenNodesImpl.call(
    host as never,
    network as never,
    multiplierOverride,
  );
}

/**
 * Repair input, output, and hidden nodes that have become structural dead ends.
 *
 * This wrapper preserves the historical best-effort behavior of
 * `neat.ensureNoDeadEnds()`: maintenance should be additive and low-friction,
 * not a new fatal step in the public workflow.
 *
 * Use it when an imported, repaired, or recently edited network may contain
 * stranded inputs, outputs, or hidden nodes that would make later controller
 * stages harder to interpret. The mutation chapter still owns the repair
 * details; this facade owns the promise that routine cleanup attempts do not
 * escalate into user-facing control-flow failures.
 *
 * @param host - `Neat` instance exposing mutation constraints and innovation tables.
 * @param network - Network whose endpoint connectivity should be repaired.
 * @returns Nothing. The network is patched in place when repairs are possible.
 */
export function ensureNoDeadEnds(
  host: NeatMaintenanceFacadeHost,
  network: Network,
): void {
  try {
    ensureNoDeadEndsImpl.call(host as never, network as never);
  } catch {
    // Dead-end repair is best-effort to preserve the stable public wrapper semantics.
  }
}

/**
 * Compute the minimum hidden-node target for the current `Neat` configuration.
 *
 * This is the read-only half of the maintenance policy. The public `Neat`
 * facade historically exposed it so callers could inspect the configured floor
 * before deciding whether to repair, import, or validate a network.
 *
 * Keeping it beside the repair wrappers makes the generated docs tell the full
 * story in one place: one helper defines the target size, one enforces it, and
 * one cleans up obvious connectivity failures that remain after topology edits.
 *
 * @param host - `Neat` instance exposing input/output counts and maintenance options.
 * @param multiplierOverride - Optional one-off multiplier overriding the configured policy.
 * @returns Minimum hidden-node count implied by explicit or multiplier-based settings.
 *
 * @example
 * ```ts
 * const minimumHidden = neat.getMinimumHiddenSize();
 * await neat.ensureMinHiddenNodes(network);
 * console.log(minimumHidden, network.nodes.length);
 * ```
 */
export function getMinimumHiddenSize(
  host: NeatMaintenanceFacadeHost,
  multiplierOverride?: number,
): number {
  const optionBag = host.options;
  const multiplier = multiplierOverride ?? optionBag.minHiddenMultiplier;

  return computeMinimumHiddenSize(
    host.input,
    host.output,
    optionBag.minHidden,
    multiplier,
  );
}
