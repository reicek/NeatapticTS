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
import type Node from '../../architecture/node';
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
 * - `edgeDensify` → distinct forward edges added in one deterministic batch via
 *   a bounded lazy sampler and `network.connectBatch()` (N = `detail.proposedAdditions`).
 *   The sampler draws source/target indices from the same forward-only ranges used
 *   by `ADD_CONN`, rejects pairs that already project, deduplicates accepted pairs,
 *   and stops after a fixed attempt budget so densification stays cheap even near
 *   the 8,000-node / 32,000-edge capacity ceiling.
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
 * Apply an `edgeDensify` delta by adding `detail.proposedAdditions` distinct
 * forward edges through a bounded lazy sampler.
 *
 * Instead of enumerating all O(N²) candidate pairs, the sampler draws
 * source/target indices from the same forward-only ranges used by `ADD_CONN`,
 * rejects pairs that already project, deduplicates accepted pairs in a local
 * `Set`, and commits the accepted batch through `network.connectBatch()`. A
 * fixed attempt budget (`MAX_ATTEMPTS_PER_EDGE = 20`) keeps densification cheap
 * even as the network approaches the 8,000-node / 32,000-edge capacity ceiling.
 *
 * The sample is deterministic whenever `network.getRandomFn()` is seeded; the
 * same network state and budget therefore always produce the same edges. If no
 * missing pairs can be found within the attempt budget, the delta is reported
 * as skipped rather than falsely applied.
 *
 * @param network - The live network to mutate in place.
 * @param delta - The edge densification delta carrying `detail.proposedAdditions`.
 * @param budget - The growth budget for re-validation.
 * @returns An applied or skipped outcome with a reason when no net growth occurred.
 */
function applyEdgeDensify(
  network: Network,
  delta: NgeMorphDelta,
  budget: NgeGrowthBudget,
): MorphApplyOutcome {
  const requestedAdditions = (delta.detail.proposedAdditions as number) ?? 1;

  assertGrowthBudget(
    network.connections.length + requestedAdditions,
    budget.maxEdges,
    'edgeDensify',
    delta.targetModuleId,
  );

  const additions = requestedAdditions;
  const MAX_ATTEMPTS_PER_EDGE = 20;

  if (additions <= 0) {
    return {
      kind: 'edgeDensify',
      status: 'skipped',
      reason:
        'ADD_CONN produced no net edges (saturated graph or sparsity budget pruning).',
    };
  }

  // Lazily sample missing forward (source, target) pairs instead of
  // enumerating all O(N²) candidates. This keeps edge densification cheap
  // even when the network grows toward the 8,000-node / 32,000-edge target.
  const nodeCount = network.nodes.length;
  const inputCount = network.input;
  const outputCount = network.output;
  const sourceEnd = nodeCount - outputCount;
  const rng = network.getRandomFn()!;

  const accepted: { from: Node; to: Node }[] = [];
  const seen = new Set<string>();
  const maxAttempts = additions * MAX_ATTEMPTS_PER_EDGE;

  for (
    let attempt = 0;
    attempt < maxAttempts && accepted.length < additions;
    attempt++
  ) {
    const sourceIndex = Math.floor(rng() * sourceEnd);
    const source = network.nodes[sourceIndex] as Node;
    const targetStart = Math.max(sourceIndex + 1, inputCount);
    const targetIndex =
      targetStart + Math.floor(rng() * (nodeCount - targetStart));
    const target = network.nodes[targetIndex] as Node;

    if (source.isProjectingTo(target)) {
      continue;
    }

    const key = `${sourceIndex}->${targetIndex}`;
    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    accepted.push({ from: source, to: target });
  }

  if (accepted.length === 0) {
    return {
      kind: 'edgeDensify',
      status: 'skipped',
      reason:
        'ADD_CONN produced no net edges (saturated graph or sparsity budget pruning).',
    };
  }

  network.connectBatch(accepted);

  return { kind: 'edgeDensify', status: 'applied' };
}

/**
 * Apply a `nodeAdd` delta by calling `ADD_NODE` the requested number of times.
 * If no hidden node is inserted (for example, because the network has no
 * eligible connection to split), the delta is reported as skipped.
 *
 * @param network - The live network to mutate in place.
 * @param delta - The node addition delta carrying `detail.proposedAdditions`.
 * @param budget - The growth budget for re-validation.
 * @returns An applied or skipped outcome.
 */
function applyNodeAdd(
  network: Network,
  delta: NgeMorphDelta,
  budget: NgeGrowthBudget,
): MorphApplyOutcome {
  const additions = (delta.detail.proposedAdditions as number) ?? 1;

  assertGrowthBudget(
    network.nodes.length + additions,
    budget.maxNodes,
    'nodeAdd',
    delta.targetModuleId,
  );

  const hiddenNodesBefore = countHiddenNodes(network);

  for (let i = 0; i < additions; i++) {
    network.mutate(mutation.ADD_NODE);
  }

  const hiddenNodesAfter = countHiddenNodes(network);
  if (hiddenNodesAfter <= hiddenNodesBefore) {
    return {
      kind: 'nodeAdd',
      status: 'skipped',
      reason: 'ADD_NODE produced no net hidden nodes.',
    };
  }

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
