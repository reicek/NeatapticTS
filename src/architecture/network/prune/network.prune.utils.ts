import type Network from '../../network/network';
import type { PruningMethod } from '../network.types';
import {
  buildEvolutionaryPruneSelection,
  buildEvolutionaryTarget,
  disconnectEvolutionaryConnections,
  getOrCaptureEvolutionaryBaseline,
  markEvolutionaryTopologyDirty,
  normalizeEvolutionaryTargetSparsity,
} from './network.prune.evolutionary.utils';
import { maybeRunRegrowth } from './network.prune.regrowth.utils';
import {
  buildPruneSelection,
  buildScheduledTarget,
  disconnectConnections,
  getInitialConnectionBaseline,
  getPruningConfig,
  markPruneIteration,
  markTopologyDirty,
  resolvePruningMethod,
  shouldRunScheduledPrune,
} from './network.prune.schedule.utils';
import {
  calculateSparsityFromBaseline,
  readInitialSparsityBaseline,
} from './network.prune.sparsity.utils';
export {
  configureSparsityBudget,
  getSparsityBudgetSnapshot,
} from './network.prune.budget.utils';
import { PRUNING_METHOD_MAGNITUDE } from './network.prune.utils.types';

/**
 * Structured and dynamic pruning utilities for networks.
 *
 * Features:
 *  - Scheduled pruning during gradient-based training ({@link maybePrune}) with linear sparsity ramp.
 *  - Evolutionary generation pruning toward a target sparsity ({@link pruneToSparsity}).
 *  - Two ranking heuristics:
 *      magnitude: |w|
 *      snip: |w * g| approximation (g approximated via accumulated delta stats; falls back to |w|)
 *  - Optional stochastic regrowth during scheduled pruning (dynamic sparse training), preserving acyclic constraints.
 *
 * Internal state fields (attached to Network through a loose internal bridge):
 *  - _pruningConfig: user-specified schedule & options (start, end, frequency, targetSparsity, method, regrowFraction, lastPruneIter)
 *  - _initialConnectionCount: baseline connection count captured outside (first training iteration)
 *  - _evoInitialConnCount: baseline for evolutionary pruning (first invocation of pruneToSparsity)
 *  - _rand: deterministic RNG function
 *  - _enforceAcyclic: boolean flag enforcing forward-only connectivity ordering
 *  - _topoDirty: topology order invalidation flag consumed by activation fast path / topological sorting
 */

/**
 * Opportunistically perform scheduled pruning during gradient-based training.
 *
 * Scheduling model:
 *  - start / end define an iteration window (inclusive) during which pruning may occur
 *  - frequency defines cadence (every N iterations inside the window)
 *  - targetSparsity is linearly annealed from 0 to its final value across the window
 *  - method chooses ranking heuristic (magnitude | snip)
 *  - optional regrowFraction allows dynamic sparse training: after removing edges we probabilistically regrow
 *    a fraction of them at random unused positions (respecting acyclic constraint if enforced)
 *
 * SNIP heuristic:
 *  - Uses |w * grad| style saliency approximation (here reusing stored delta stats as gradient proxy)
 *  - Falls back to pure magnitude if gradient stats absent.
 */
/**
 * Perform scheduled pruning at a given training iteration if conditions are met.
 *
 * Scheduling fields (cfg): start, end, frequency, targetSparsity, method ('magnitude' | 'snip'), regrowFraction.
 * The target sparsity ramps linearly from 0 at start to cfg.targetSparsity at end.
 *
 * @param iteration Current (0-based or 1-based) training iteration counter used for scheduling.
 */
export function maybePrune(this: Network, iteration: number): void {
  // Step 1: Collect required schedule and baseline context.
  const pruningConfig = getPruningConfig(this);
  if (!pruningConfig) return;

  // Step 2: Exit early when this iteration is not eligible for pruning.
  if (!shouldRunScheduledPrune(iteration, pruningConfig)) return;

  const initialConnectionBaseline = getInitialConnectionBaseline(this);
  if (!initialConnectionBaseline) return;

  // Step 3: Compute current target density and required removals.
  const scheduledTarget = buildScheduledTarget(
    {
      iteration,
      scheduleStart: pruningConfig.start,
      scheduleEnd: pruningConfig.end,
      targetSparsity: pruningConfig.targetSparsity,
      baselineConnectionCount: initialConnectionBaseline,
    },
    this.connections.length,
  );

  if (scheduledTarget.excessConnectionCount <= 0) {
    markPruneIteration(pruningConfig, iteration);
    return;
  }

  // Step 4: Select and remove the least important connections.
  const pruneSelection = buildPruneSelection({
    connections: this.connections,
    removalCount: scheduledTarget.excessConnectionCount,
    method: resolvePruningMethod(pruningConfig.method),
  });

  disconnectConnections(this, pruneSelection.connectionsToPrune);

  // Step 5: Optionally regrow random valid edges.
  maybeRunRegrowth(this, {
    prunedConnectionCount: pruneSelection.connectionsToPrune.length,
    regrowFraction: pruningConfig.regrowFraction,
    desiredRemainingConnections: scheduledTarget.desiredRemainingConnections,
  });

  // Step 6: Persist bookkeeping for topology and schedule state.
  markPruneIteration(pruningConfig, iteration);
  markTopologyDirty(this);
}

/**
 * Evolutionary (generation-based) pruning toward a target sparsity baseline.
 * Unlike maybePrune this operates immediately relative to the first invocation's connection count
 * (stored separately as _evoInitialConnCount) and does not implement scheduling or regrowth.
 *
 * @param targetSparsity - Requested target sparsity.
 * @param method - Connection ranking heuristic.
 * @returns Nothing.
 */
export function pruneToSparsity(
  this: Network,
  targetSparsity: number,
  method: PruningMethod = PRUNING_METHOD_MAGNITUDE,
): void {
  // Step 1: Normalize user target and short-circuit no-op requests.
  const normalizedTargetSparsity =
    normalizeEvolutionaryTargetSparsity(targetSparsity);
  if (normalizedTargetSparsity <= 0) return;

  // Step 2: Resolve baseline and compute required removals.
  const evolutionaryBaseline = getOrCaptureEvolutionaryBaseline(this);
  const evolutionaryTarget = buildEvolutionaryTarget(
    {
      targetSparsity: normalizedTargetSparsity,
      baselineConnectionCount: evolutionaryBaseline,
    },
    this.connections.length,
  );
  if (evolutionaryTarget.excessConnectionCount <= 0) return;

  // Step 3: Select and remove least important connections.
  const pruneSelection = buildEvolutionaryPruneSelection({
    connections: this.connections,
    removalCount: evolutionaryTarget.excessConnectionCount,
    method,
  });

  disconnectEvolutionaryConnections(this, pruneSelection.connectionsToPrune);

  // Step 4: Mark topology cache invalid after structural change.
  markEvolutionaryTopologyDirty(this);
}

/**
 * Current sparsity fraction relative to the training-time pruning baseline.
 *
 * @returns Current sparsity in the [0,1] range when baseline is available.
 */
export function getCurrentSparsity(this: Network): number {
  // Step 1: Resolve baseline and return dense default when unavailable.
  const initialBaseline = readInitialSparsityBaseline(this);
  if (!initialBaseline) return 0;

  // Step 2: Compute current sparsity from baseline.
  return calculateSparsityFromBaseline(
    this.connections.length,
    initialBaseline,
  );
}

// Explicit export object to keep module side-effects clear (tree-shaking friendliness)
export {};
