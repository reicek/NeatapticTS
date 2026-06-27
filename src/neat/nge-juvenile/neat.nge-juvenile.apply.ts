/**
 * Morph applier for the NGE juvenile phase.
 *
 * This module owns the translation from dry-run `NgeMorphDelta` structural plans
 * into concrete `network.mutate()` calls. Each morph kind maps to a specific
 * NEAT mutation operator (or a documented no-op for `slotExpand`), and every
 * growth or prune action is re-validated against the supplied DNA budget before
 * the network is touched.
 *
 * The applier is the final gate between planning and structural commitment: a
 * delta that passes the planner's dry-run validation can still be rejected here
 * if the live network state has drifted past a budget cap or floor since the
 * plan was produced.
 */

import mutation from '../../methods/mutation/mutation';
import type Network from '../../architecture/network';
import { NgeJuvenile_BudgetError } from './neat.nge-juvenile.errors';
import type {
  NgeGrowthBudget,
  NgeMorphDelta,
  NgePruneBudget,
} from './neat.nge-juvenile.types';

/**
 * Combined growth and prune budget consumed by the morph applier.
 * The applier re-validates both before mutating.
 */
export interface MorphApplyBudget {
  /** DNA-configured growth caps re-validated before each growth mutation. */
  growth: NgeGrowthBudget;
  /** DNA-configured prune floors re-validated before each prune mutation. */
  prune: NgePruneBudget;
}

/**
 * Outcome produced for one morph delta after the applier processes it.
 */
export interface MorphApplyOutcome {
  /** The morph kind that was processed. */
  kind: NgeMorphDelta['kind'];
  /** Whether the delta was applied or skipped. */
  status: 'applied' | 'skipped';
  /** Optional reason explaining why a delta was skipped. */
  reason?: string;
}

/**
 * Apply a batch of juvenile morph deltas to a network, re-validating DNA
 * budgets before each structural mutation.
 *
 * Each delta is translated into the corresponding NEAT mutation operator:
 *
 * - `edgeDensify` → `ADD_CONN` called N times (N = `detail.proposedAdditions`).
 * - `nodeAdd` → `ADD_NODE`.
 * - `edgePrune` → direct disconnect of the specific connection identified by
 *   `detail.candidateId` (not random `SUB_CONN`).
 * - `compact` → `SUB_NODE`.
 * - `slotExpand` → documented no-op; returns a skipped outcome.
 *
 * Growth mutations (`edgeDensify`, `nodeAdd`) are guarded by the growth budget
 * caps (`maxEdges`, `maxNodes`). Prune mutations (`edgePrune`, `compact`) are
 * guarded by the prune budget floors (`minEdges`, `minNodes`). If a budget
 * would be violated, {@link NgeJuvenile_BudgetError} is thrown.
 *
 * @param network - The live network to mutate in place.
 * @param deltas - Ordered list of dry-run morph deltas to apply.
 * @param budget - Combined growth and prune budgets for re-validation.
 * @returns One outcome per input delta, preserving order.
 * @throws {NgeJuvenile_BudgetError} When a growth mutation would exceed a cap
 *   or a prune mutation would violate a floor.
 *
 * @example
 * ```ts
 * const outcomes = applyMorphDeltas(network, deltas, budget);
 * for (const outcome of outcomes) {
 *   if (outcome.status === 'skipped') {
 *     console.log(`${outcome.kind} skipped: ${outcome.reason}`);
 *   }
 * }
 * ```
 */
export function applyMorphDeltas(
  network: Network,
  deltas: readonly NgeMorphDelta[],
  budget: MorphApplyBudget,
): MorphApplyOutcome[] {
  return deltas.map((delta) => applyOneDelta(network, delta, budget));
}

/**
 * Dispatch one morph delta to its handler based on `delta.kind`.
 *
 * @param network - The live network to mutate in place.
 * @param delta - One dry-run morph delta.
 * @param budget - Combined growth and prune budgets for re-validation.
 * @returns One morph apply outcome.
 */
function applyOneDelta(
  network: Network,
  delta: NgeMorphDelta,
  budget: MorphApplyBudget,
): MorphApplyOutcome {
  switch (delta.kind) {
    case 'edgeDensify':
      return applyEdgeDensify(network, delta, budget.growth);
    case 'nodeAdd':
      return applyNodeAdd(network, delta, budget.growth);
    case 'edgePrune':
      return applyEdgePrune(network, delta, budget.prune);
    case 'compact':
      return applyCompact(network, delta, budget.prune);
    case 'slotExpand':
      return {
        kind: 'slotExpand',
        status: 'skipped',
        reason: 'No NEAT mutation equivalent for episodic slot expansion.',
      };
  }
}

/**
 * Assert that a projected count does not exceed the DNA growth cap.
 *
 * @param projectedCount - The count that would result after the growth mutation.
 * @param maxCount - The DNA-configured maximum allowed count.
 * @param kind - The morph kind label for the error message.
 * @param moduleId - The target module identifier for the error message.
 * @throws {NgeJuvenile_BudgetError} When the projected count exceeds the cap.
 */
function assertGrowthBudget(
  projectedCount: number,
  maxCount: number,
  kind: string,
  moduleId: string,
): void {
  if (projectedCount > maxCount) {
    throw new NgeJuvenile_BudgetError(
      `Growth morph ${kind} would exceed the budget cap for ${moduleId}.`,
    );
  }
}

/**
 * Assert that a projected count does not drop below the DNA prune floor.
 *
 * @param projectedCount - The count that would result after the prune mutation.
 * @param minCount - The DNA-configured minimum required count.
 * @param kind - The morph kind label for the error message.
 * @param moduleId - The target module identifier for the error message.
 * @throws {NgeJuvenile_BudgetError} When the projected count drops below the floor.
 */
function assertPruneBudget(
  projectedCount: number,
  minCount: number,
  kind: string,
  moduleId: string,
): void {
  if (projectedCount < minCount) {
    throw new NgeJuvenile_BudgetError(
      `Prune morph ${kind} would violate the floor for ${moduleId}.`,
    );
  }
}

/**
 * Apply an `edgeDensify` delta by calling `ADD_CONN` N times.
 *
 * @param network - The live network to mutate in place.
 * @param delta - The edge densification delta carrying `detail.proposedAdditions`.
 * @param budget - The growth budget for re-validation.
 * @returns An applied outcome.
 */
function applyEdgeDensify(
  network: Network,
  delta: NgeMorphDelta,
  budget: NgeGrowthBudget,
): MorphApplyOutcome {
  const additions = delta.detail.proposedAdditions as number;

  assertGrowthBudget(
    network.connections.length + additions,
    budget.maxEdges,
    'edgeDensify',
    delta.targetModuleId,
  );

  for (let i = 0; i < additions; i++) {
    network.mutate(mutation.ADD_CONN);
  }

  return { kind: 'edgeDensify', status: 'applied' };
}

/**
 * Apply a `nodeAdd` delta by calling `ADD_NODE`.
 *
 * @param network - The live network to mutate in place.
 * @param delta - The node addition delta.
 * @param budget - The growth budget for re-validation.
 * @returns An applied outcome.
 */
function applyNodeAdd(
  network: Network,
  delta: NgeMorphDelta,
  budget: NgeGrowthBudget,
): MorphApplyOutcome {
  assertGrowthBudget(
    network.nodes.length + 1,
    budget.maxNodes,
    'nodeAdd',
    delta.targetModuleId,
  );

  network.mutate(mutation.ADD_NODE);

  return { kind: 'nodeAdd', status: 'applied' };
}

/**
 * Apply an `edgePrune` delta by disconnecting the specific connection
 * identified by `detail.candidateId`.
 *
 * Unlike `SUB_CONN` which picks a random edge, this finds the exact
 * connection whose innovation ID matches the candidate and disconnects
 * it directly via `network.disconnect(from, to)`.
 *
 * @param network - The live network to mutate in place.
 * @param delta - The edge prune delta carrying `detail.candidateId`.
 * @param budget - The prune budget for re-validation.
 * @returns An applied outcome.
 */
function applyEdgePrune(
  network: Network,
  delta: NgeMorphDelta,
  budget: NgePruneBudget,
): MorphApplyOutcome {
  assertPruneBudget(
    network.connections.length - 1,
    budget.minEdges,
    'edgePrune',
    delta.targetModuleId,
  );

  const candidateId = delta.detail.candidateId as string;
  const target = network.connections.find(
    (conn) => String(conn.innovation) === candidateId,
  );

  network.disconnect(target!.from, target!.to);

  return { kind: 'edgePrune', status: 'applied' };
}

/**
 * Apply a `compact` delta by calling `SUB_NODE` to remove a hidden node.
 *
 * @param network - The live network to mutate in place.
 * @param delta - The compact delta.
 * @param budget - The prune budget for re-validation.
 * @returns An applied outcome.
 */
function applyCompact(
  network: Network,
  delta: NgeMorphDelta,
  budget: NgePruneBudget,
): MorphApplyOutcome {
  assertPruneBudget(
    countHiddenNodes(network) - 1,
    budget.minNodes,
    'compact',
    delta.targetModuleId,
  );

  network.mutate(mutation.SUB_NODE);

  return { kind: 'compact', status: 'applied' };
}

/**
 * Count the hidden nodes currently in the network.
 *
 * @param network - The live network to inspect.
 * @returns The number of nodes whose type is `'hidden'`.
 */
function countHiddenNodes(network: Network): number {
  return network.nodes.filter((node) => node.type === 'hidden').length;
}
