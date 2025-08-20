/**
 * NodePool (Phase 2 – COMPLETE)
 * =============================
 * Lightweight object pool for `Node` instances mirroring (future) connection pooling patterns.
 *
 * Objectives:
 * 1. Reduce GC pressure during topology mutation / morphogenesis (frequent add/remove of nodes).
 * 2. Provide deterministic, fully-reset instances on `acquire()` so algorithms can assume a fresh state.
 * 3. Provide instrumentation (reused vs fresh, highWaterMark, recycledRatio) consumed by benchmarks.
 * 4. Serve as a future anchor for slab-backed / SoA node state (Phase 3) without altering the public API.
 *
 * Phase 2 Deliverables Implemented Here:
 * - acquire / release with thorough reset and defensive scrub on release.
 * - highWaterMark updated ONLY on release (tracks retained capacity not transient demand).
 * - Counters reusedCount & freshCount powering recycledRatio assertions.
 * - resetNodePool() for deterministic test harness setup.
 *
 * Deferred (Phase 3+): preWarm(count), adaptive trim(), leak pattern heuristics, slab field hydration.
 */
import Node from './node';

/** Shape describing minimal mutable fields we explicitly reset (used internally). */
interface ResettableNodeFields {
  activation: number;
  state: number;
  old: number;
  mask: number;
  previousDeltaBias: number;
  totalDeltaBias: number;
  derivative?: number;
  connections: Node['connections'];
  error: Node['error'];
  bias: number;
  index?: number;
  geneId: number;
  type: string;
  squash: Node['squash'];
}

/** Internal free list (stack) storing recycled Node instances. */
const pool: Node[] = [];
/** High-water mark statistic (observability aid; may feed future leak detection tooling). */
let highWaterMark = 0;

/** Incrementing counter to allocate fresh stable geneIds when resetting pooled nodes. */
let nextGeneId = 1;

/** Counters for recycling efficiency instrumentation (Phase 2 stress harness). */
let reusedCount = 0;
let freshCount = 0;

/**
 * Reset all mutable / dynamic fields of a node to a pristine post-construction state.
 * This mirrors logic in the constructor & `clear()` while also clearing arrays & error objects.
 *
 * We intentionally do NOT reset the `type` or `squash` function unless explicitly provided so callers
 * can optionally request a different type on acquire. Bias is reinitialized consistent with constructor semantics.
 */
function resetNode(node: Node, type?: string, rng: () => number = Math.random) {
  // Preserve or update type
  if (type) (node as any).type = type;
  const t = (node as any).type;
  // Reinitialize bias identical to constructor semantics
  (node as any).bias = t === 'input' ? 0 : rng() * 0.2 - 0.1;
  // Core dynamic state
  (node as any).activation = 0;
  (node as any).state = 0;
  (node as any).old = 0;
  (node as any).mask = 1;
  (node as any).previousDeltaBias = 0;
  (node as any).totalDeltaBias = 0;
  (node as any).derivative = undefined;
  // Reset connections arrays in-place to retain original array identities (helps hidden class stability)
  node.connections.in.length = 0;
  node.connections.out.length = 0;
  node.connections.gated.length = 0;
  node.connections.self.length = 0;
  // Error object (replace wholesale)
  (node as any).error = { responsibility: 0, projected: 0, gated: 0 };
  // Assign new stable gene id (distinct from original run usage)
  (node as any).geneId = nextGeneId++;
  // Index is preserved; we do NOT recycle indices here (network rebuild logic may reassign in future phase)
}

/** Options bag for acquiring a node. */
export interface AcquireNodeOptions {
  /** Node type (input|hidden|output|constant). Defaults to 'hidden'. */
  type?: string;
  /** Optional custom activation function. */
  activationFn?: (x: number, derivate?: boolean) => number;
  /** Optional rng (seeded) for deterministic bias initialization. */
  rng?: () => number;
}

/**
 * Acquire (obtain) a node instance from the pool (or construct a new one if empty).
 * The node is guaranteed to have fully reset dynamic state (activation, gradients, error, connections).
 */
export function acquireNode(opts: AcquireNodeOptions = {}): Node {
  const { type = 'hidden', activationFn, rng } = opts;
  let node: Node;
  if (pool.length) {
    node = pool.pop()!;
    reusedCount++;
    resetNode(node, type, rng);
    if (activationFn) (node as any).squash = activationFn;
  } else {
    node = new Node(type, activationFn, rng);
    (node as any).geneId = nextGeneId++;
    freshCount++;
  }
  // NOTE: highWaterMark reflects MAX retained pool size; updated only on release().
  return node;
}

/**
 * Release (recycle) a node back into the pool. The caller MUST ensure the node is fully detached
 * from any network (connections arrays pruned, no external references maintained) to prevent leaks.
 * After release, the node must be considered invalid until re-acquired.
 *
 * Phase 2: Automatically invoked by Network.remove() when pooling is enabled to recycle pruned nodes.
 */
export function releaseNode(node: Node) {
  // Proactively scrub large arrays / references to help GC of graphs containing this node.
  node.connections.in.length = 0;
  node.connections.out.length = 0;
  node.connections.gated.length = 0;
  node.connections.self.length = 0;
  (node as any).error = { responsibility: 0, projected: 0, gated: 0 };
  pool.push(node);
  if (pool.length > highWaterMark) highWaterMark = pool.length;
}

/**
 * Get current pool statistics (for debugging / future leak detection).
 */
export function nodePoolStats() {
  // recycledRatio expresses long-run reuse efficiency; 0 => all fresh, 1 => full reuse.
  return {
    size: pool.length,
    highWaterMark,
    reused: reusedCount,
    fresh: freshCount,
    recycledRatio:
      reusedCount + freshCount > 0
        ? reusedCount / (reusedCount + freshCount)
        : 0,
  };
}

/**
 * Reset the pool (drops all retained nodes). Intended for test harness cleanup.
 */
export function resetNodePool() {
  pool.length = 0; // drop all retained instances
  highWaterMark = 0; // reset leak tracking baseline
  reusedCount = 0; // reset instrumentation counters
  freshCount = 0;
}

// Future (Phase 2+): preWarm(count), trim(predicate), integrate with network pruning events.

export default { acquireNode, releaseNode, nodePoolStats, resetNodePool };
