import type Network from '../architecture/network';
import {
  ensureMinHiddenNodes as ensureMinHiddenNodesImpl,
  ensureNoDeadEnds as ensureNoDeadEndsImpl,
} from './neat.mutation';
import { computeMinimumHiddenSize } from './neat.mutation.min-hidden.utils';
import type { NeatControllerForMutation } from './neat.mutation.types';

type NeatMaintenanceOptions = NeatControllerForMutation['options'] & {
  minHidden?: number;
  minHiddenMultiplier?: number;
};

/**
 * Narrow `Neat` host surface required by the public maintenance facade.
 *
 * This boundary keeps the contract focused on topology-maintenance state:
 * mutation constraints, innovation bookkeeping, endpoint counts, and the
 * legacy helper used to compute a minimum hidden-node target.
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
 * Public maintenance facade helpers for the stable `Neat` entrypoint.
 *
 * These wrappers keep a small cluster of topology-repair helpers out of the
 * main [src/neat.ts](src/neat.ts) class body: enforcing a minimum hidden-node
 * budget, repairing dead-end connectivity, and exposing the computed minimum
 * hidden target. Grouping them here makes the public facade easier to scan
 * without changing the long-standing method names or the mutation helpers that
 * already own the real repair logic.
 *
 * Invariant: this boundary only maintains baseline structural viability for a
 * single network. It does not change mutation operator selection, speciation,
 * or public import paths.
 */

/**
 * Ensure a network satisfies the configured minimum hidden-node policy.
 *
 * The underlying mutation helper may add hidden nodes and wire them into the
 * graph so later mutation and evaluation steps start from a minimally viable
 * structure.
 *
 * @param host - `Neat` instance exposing mutation constraints and innovation tables.
 * @param network - Network whose hidden-node floor should be enforced.
 * @param multiplierOverride - Optional one-off multiplier overriding the configured policy.
 * @returns Promise that resolves after any required topology repair finishes.
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
 * This preserves the historical best-effort behavior of `neat.ensureNoDeadEnds()`:
 * if the underlying repair helper throws, the public facade suppresses that
 * failure so maintenance stays additive rather than fatal.
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
 * The public `Neat` facade historically exposed this as a read-only policy
 * helper. Keeping it beside the repair wrappers makes the generated docs tell a
 * clearer story: one boundary defines the target size, and the neighboring
 * helpers enforce it on concrete networks.
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
