import type Network from '../network';
import Node from '../node';
import Connection from '../connection';

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
 * Internal State Fields (attached to Network via `any` casting):
 *  - _pruningConfig: user-specified schedule & options (start, end, frequency, targetSparsity, method, regrowFraction, lastPruneIter)
 *  - _initialConnectionCount: baseline connection count captured outside (first training iteration)
 *  - _evoInitialConnCount: baseline for evolutionary pruning (first invocation of pruneToSparsity)
 *  - _rand: deterministic RNG function
 *  - _enforceAcyclic: boolean flag enforcing forward-only connectivity ordering
 *  - _topoDirty: topology order invalidation flag consumed by activation fast path / topological sorting
 */

// ---------------------------------------------------------------------------
// Internal helpers (not exported)
// ---------------------------------------------------------------------------

/** Rank connections ascending by removal priority according to a method. */
function rankConnections(
  conns: Connection[],
  method: 'magnitude' | 'snip'
): Connection[] {
  /** Shallow copy of connections to be sorted by removal priority (ascending). */
  const ranked = [...conns];
  if (method === 'snip') {
    ranked.sort((a: any, b: any) => {
      /** Gradient magnitude proxy for connection A (uses accumulated or last delta). */
      const gradMagA =
        Math.abs(a.totalDeltaWeight) || Math.abs(a.previousDeltaWeight) || 0;
      /** Gradient magnitude proxy for connection B (uses accumulated or last delta). */
      const gradMagB =
        Math.abs(b.totalDeltaWeight) || Math.abs(b.previousDeltaWeight) || 0;
      /** Saliency estimate for connection A (|w| * |g| fallback to |w|). */
      const saliencyA = gradMagA
        ? Math.abs(a.weight) * gradMagA
        : Math.abs(a.weight);
      /** Saliency estimate for connection B (|w| * |g| fallback to |w|). */
      const saliencyB = gradMagB
        ? Math.abs(b.weight) * gradMagB
        : Math.abs(b.weight);
      return saliencyA - saliencyB; // ascending => remove lowest first
    });
  } else {
    ranked.sort((a, b) => Math.abs(a.weight) - Math.abs(b.weight));
  }
  return ranked;
}

/** Attempt stochastic regrowth of pruned connections up to a desired remaining count. */
function regrowConnections(
  network: Network,
  desiredRemaining: number,
  maxAttempts: number
) {
  /** Internal network reference for private fields (_rand, _enforceAcyclic). */
  const netAny = network as any;
  /** Number of attempted regrowth trials so far. */
  let attempts = 0;
  while (
    network.connections.length < desiredRemaining &&
    attempts < maxAttempts
  ) {
    attempts++;
    /** Random source node candidate for a new connection. */
    const fromNode =
      network.nodes[Math.floor(netAny._rand() * network.nodes.length)];
    /** Random target node candidate for a new connection. */
    const toNode =
      network.nodes[Math.floor(netAny._rand() * network.nodes.length)];
    if (!fromNode || !toNode || fromNode === toNode) continue; // invalid pair
    if (network.connections.some((c) => c.from === fromNode && c.to === toNode))
      continue; // duplicate
    if (
      netAny._enforceAcyclic &&
      network.nodes.indexOf(fromNode) > network.nodes.indexOf(toNode)
    )
      continue; // violates order
    network.connect(fromNode, toNode);
  }
}

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
  /** Active pruning configuration attached to the network (or undefined if disabled). */
  const cfg: any = (this as any)._pruningConfig; // internal schedule/config
  if (!cfg) return; // disabled
  if (iteration < cfg.start || iteration > cfg.end) return; // outside schedule window
  if (cfg.lastPruneIter != null && iteration === cfg.lastPruneIter) return; // already pruned this iteration
  if ((iteration - cfg.start) % (cfg.frequency || 1) !== 0) return; // off-cycle
  /** Baseline connection count captured at training start for scheduled pruning reference. */
  const initialConnectionBaseline = (this as any)._initialConnectionCount;
  if (!initialConnectionBaseline) return; // baseline not captured yet

  /** Progress fraction (0..1) through pruning window. */
  const progressFraction =
    (iteration - cfg.start) / Math.max(1, cfg.end - cfg.start);
  /** Instantaneous target sparsity (linearly annealed). */
  const targetSparsityNow =
    cfg.targetSparsity * Math.min(1, Math.max(0, progressFraction));
  /** Desired remaining connection count based on baseline & current sparsity. */
  const desiredRemainingConnections = Math.max(
    1,
    Math.floor(initialConnectionBaseline * (1 - targetSparsityNow))
  );
  /** Excess connections present right now that should be removed to hit schedule target. */
  const excessConnectionCount =
    this.connections.length - desiredRemainingConnections;
  if (excessConnectionCount <= 0) {
    cfg.lastPruneIter = iteration;
    return;
  }

  /** Ranked connections ascending by removal priority. */
  const rankedConnections = rankConnections(
    this.connections,
    cfg.method || 'magnitude'
  );
  /** Subset of connections to prune this iteration. */
  const connectionsToPrune = rankedConnections.slice(0, excessConnectionCount);
  connectionsToPrune.forEach((conn) => this.disconnect(conn.from, conn.to));

  // Dynamic sparse regrowth (optional) to maintain target density while allowing exploration.
  if (cfg.regrowFraction && cfg.regrowFraction > 0) {
    /** Intended number of new connections to attempt to regrow (before attempt limit multiplier). */
    const intendedRegrowCount = Math.floor(
      connectionsToPrune.length * cfg.regrowFraction
    );
    regrowConnections(
      this,
      desiredRemainingConnections,
      intendedRegrowCount * 10
    );
  }

  cfg.lastPruneIter = iteration; // record bookkeeping
  (this as any)._topoDirty = true; // structural change => invalidate cached order
}

/**
 * Evolutionary (generation-based) pruning toward a target sparsity baseline.
 * Unlike maybePrune this operates immediately relative to the first invocation's connection count
 * (stored separately as _evoInitialConnCount) and does not implement scheduling or regrowth.
 */
export function pruneToSparsity(
  this: Network,
  targetSparsity: number,
  method: 'magnitude' | 'snip' = 'magnitude'
): void {
  if (targetSparsity <= 0) return; // trivial
  if (targetSparsity >= 1) targetSparsity = 0.999; // safety clamp
  /** Internal network reference for private evolutionary baseline. */
  const netAny = this as any;
  if (!netAny._evoInitialConnCount)
    netAny._evoInitialConnCount = this.connections.length; // capture baseline only once
  /** Connection count baseline at first evolutionary pruning invocation. */
  const evolutionaryBaseline = netAny._evoInitialConnCount;
  /** Desired number of connections to retain. */
  const desiredRemainingConnections = Math.max(
    1,
    Math.floor(evolutionaryBaseline * (1 - targetSparsity))
  );
  /** Excess relative to desired number. */
  const excessConnectionCount =
    this.connections.length - desiredRemainingConnections;
  if (excessConnectionCount <= 0) return; // already at or below target
  /** Ranked connections ascending by removal priority. */
  const rankedConnections = rankConnections(this.connections, method);
  /** Slice of ranked connections to remove to reach target sparsity. */
  const connectionsToRemove = rankedConnections.slice(0, excessConnectionCount);
  connectionsToRemove.forEach((c) => this.disconnect(c.from, c.to));
  netAny._topoDirty = true;
}

/** Current sparsity fraction relative to the training-time pruning baseline. */
export function getCurrentSparsity(this: Network): number {
  /** Baseline connection count used for scheduled pruning sparsity measurement. */
  const initialBaseline = (this as any)._initialConnectionCount;
  if (!initialBaseline) return 0;
  return 1 - this.connections.length / initialBaseline;
}

// Explicit export object to keep module side-effects clear (tree-shaking friendliness)
export {};
