/**
 * Core node-pool chapter for the architecture surface.
 *
 * This folder owns the reusable pool that recycles `Node` instances after
 * topology edits. It sits beside the main node chapter because it is not a new
 * neuron type; it is the lifecycle policy that decides when detached nodes can
 * be scrubbed, retained, and reused.
 *
 * Read this chapter in three passes:
 *
 * 1. start with `acquireNode()` to see how callers obtain a fully reset node,
 * 2. continue to `releaseNode()` when you need the detach-and-recycle rules,
 * 3. finish with `nodePoolStats()` and `resetNodePool()` for observability and
 *    deterministic test harness cleanup.
 */
import Node from '../node/node';

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
 * Reset all mutable / dynamic fields of a node to a pristine post-construction
 * state.
 *
 * The pool intentionally reuses object identity while reinitializing runtime
 * state so topology mutation can avoid needless allocation churn. This helper
 * mirrors the constructor and `clear()` semantics, but also scrubs connection
 * arrays, optimizer scratch space, and error bookkeeping that would otherwise
 * leak state across reuse cycles.
 *
 * @param node Recycled node instance to scrub.
 * @param type Optional node type override for the next acquisition.
 * @param rng Optional RNG used for deterministic bias initialization.
 * @returns Nothing.
 */
const resetNode = (
  node: Node,
  type?: string,
  rng: () => number = Math.random,
): void => {
  if (type) {
    node.type = type;
  }

  const nodeType = node.type;
  node.bias = nodeType === 'input' ? 0 : rng() * 0.2 - 0.1;
  node.activation = 0;
  node.state = 0;
  node.old = 0;
  node.mask = 1;
  node.previousDeltaBias = 0;
  node.totalDeltaBias = 0;
  node.derivative = undefined;
  node.connections.in.length = 0;
  node.connections.out.length = 0;
  node.connections.gated.length = 0;
  node.connections.self.length = 0;
  node.error = { responsibility: 0, projected: 0, gated: 0 };
  node.geneId = nextGeneId++;
};

/** Options bag for acquiring a node. */
export interface AcquireNodeOptions {
  /** Node type (`input` | `hidden` | `output` | `constant`). Defaults to `hidden`. */
  type?: string;
  /** Optional custom activation function. */
  activationFn?: (x: number, derivate?: boolean) => number;
  /** Optional RNG for deterministic bias initialization. */
  rng?: () => number;
}

/**
 * Acquire a node instance from the pool, or construct a fresh one when the
 * pool is empty.
 *
 * The returned node is guaranteed to have detached connections, cleared error
 * state, and a fresh gene id for its next lifecycle.
 *
 * @param opts Optional acquisition settings.
 * @returns A ready-to-use node instance.
 */
export const acquireNode = (opts: AcquireNodeOptions = {}): Node => {
  const { type = 'hidden', activationFn, rng } = opts;
  let node: Node;

  if (pool.length) {
    node = pool.pop()!;
    reusedCount++;
    resetNode(node, type, rng);
    if (activationFn) {
      node.squash = activationFn;
    }
  } else {
    node = new Node(type, activationFn, rng);
    node.geneId = nextGeneId++;
    freshCount++;
  }

  return node;
};

/**
 * Release a detached node back into the pool.
 *
 * Callers must ensure the node is no longer part of a live graph. The pool
 * keeps the object shell, not the prior topology membership.
 *
 * @param node Detached node instance to recycle.
 * @returns Nothing.
 */
export const releaseNode = (node: Node): void => {
  node.connections.in.length = 0;
  node.connections.out.length = 0;
  node.connections.gated.length = 0;
  node.connections.self.length = 0;
  node.error = { responsibility: 0, projected: 0, gated: 0 };
  pool.push(node);

  if (pool.length > highWaterMark) {
    highWaterMark = pool.length;
  }
};

/**
 * Get current pool statistics for diagnostics and memory reporting.
 *
 * @returns Pool size, reuse counters, and the long-run recycled ratio.
 */
export const nodePoolStats = (): {
  size: number;
  highWaterMark: number;
  reused: number;
  fresh: number;
  recycledRatio: number;
} => {
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
};

/**
 * Drop all retained pooled nodes and reset instrumentation counters.
 *
 * @returns Nothing.
 */
export const resetNodePool = (): void => {
  pool.length = 0;
  highWaterMark = 0;
  reusedCount = 0;
  freshCount = 0;
};

export default { acquireNode, releaseNode, nodePoolStats, resetNodePool };
