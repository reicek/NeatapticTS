# architecture/nodePool

Core node-pool chapter for the architecture surface.

This folder owns the reusable pool that recycles `Node` instances after
topology edits. It sits beside the main node chapter because it is not a new
neuron type; it is the lifecycle policy that decides when detached nodes can
be scrubbed, retained, and reused.

Read this chapter in three passes:

1. start with `acquireNode()` to see how callers obtain a fully reset node,
2. continue to `releaseNode()` when you need the detach-and-recycle rules,
3. finish with `nodePoolStats()` and `resetNodePool()` for observability and
   deterministic test harness cleanup.

## architecture/nodePool/nodePool.ts

### AcquireNodeOptions

Options bag for acquiring a node.

### acquireNode

```ts
acquireNode(
  opts: AcquireNodeOptions,
): default
```

Acquire a node instance from the pool, or construct a fresh one when the
pool is empty.

The returned node is guaranteed to have detached connections, cleared error
state, and a fresh gene id for its next lifecycle.

Parameters:
- `opts` - Optional acquisition settings.

Returns: A ready-to-use node instance.

### releaseNode

```ts
releaseNode(
  node: default,
): void
```

Release a detached node back into the pool.

Callers must ensure the node is no longer part of a live graph. The pool
keeps the object shell, not the prior topology membership.

Parameters:
- `node` - Detached node instance to recycle.

Returns: Nothing.

### nodePoolStats

```ts
nodePoolStats(): { size: number; highWaterMark: number; reused: number; fresh: number; recycledRatio: number; }
```

Get current pool statistics for diagnostics and memory reporting.

Returns: Pool size, reuse counters, and the long-run recycled ratio.

### resetNodePool

```ts
resetNodePool(): void
```

Drop all retained pooled nodes and reset instrumentation counters.

Returns: Nothing.
