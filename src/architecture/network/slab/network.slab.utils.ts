import type Network from '../../network/network';
import { _getSlabAllocationStatsSnapshot } from './network.slab.pool.utils';
import {
  _createSlabBuildContext,
  _shouldSkipSlabRebuild,
  _ensureSlabCapacitySync,
  _ensureSlabCapacityAsync,
  _populateSlabConnectionsSync,
  _populateSlabConnectionsAsync,
  _applyGainOmissionPolicy,
  _applyPlasticPolicySync,
  _applyPlasticPolicyAsync,
  _resolveAsyncChunkSize,
  _finalizeSyncSlabRebuild,
  _finalizeAsyncSlabRebuild,
} from './network.slab.rebuild.helpers.utils';
import { _canUseFastSlab } from './network.slab.fast-path.helpers.utils';
import { _buildAdjacency } from './network.slab.adjacency.helpers.utils';
import { _prepareSlabBuildPreconditions } from './network.slab.setup.utils';
import {
  _createConnectionSlabView,
  _readSlabVersion,
} from './network.slab.view.utils';
import { _activateFastSlab } from './network.slab.activate.utils';
import type {
  ConnectionSlabView,
  NetworkSlabProps,
} from './network.slab.utils.types';
import {
  SLAB_GROWTH_FACTOR_BROWSER,
  SLAB_GROWTH_FACTOR_NODE,
  SLAB_DEFAULT_ASYNC_CHUNK_SIZE,
} from './network.slab.utils.types';

export type { ConnectionSlabView } from './network.slab.utils.types';

/**
 * Slab Packing / Structure‑of‑Arrays Backend (Educational Module)
 * ==============================================================
 * Packs per‑connection data into parallel typed arrays (SoA) to accelerate
 * forward passes and to illustrate memory/layout optimizations.
 *
 * Why SoA?
 *  - Locality & fewer cache misses.
 *  - Predictable tight numeric loops (JIT / SIMD friendly).
 *  - Easy instrumentation (single contiguous blocks to measure & diff).
 *
 * Key Arrays (logical length = `used`): weights | from | to | flags | (optional) gain | (optional) plastic.
 * Adjacency (CSR style): outStart (nodeCount+1), outOrder (per‑source permutation) enabling fast fan‑out.
 *
 * On‑Demand & Omission:
 *  - Gain/plastic slabs allocated only when a non‑neutral value appears; freed if neutrality returns.
 *  - `getConnectionSlab()` synthesizes a neutral gain view if omitted internally (keeps teaching tools simple).
 *
 * Capacity Strategy: geometric growth (1.25x browser / 1.75x Node) amortizes realloc cost.
 * Pooling (config gated) reuses typed arrays (see `getSlabAllocationStats`).
 *
 * Rebuild Steps (sync): reindex nodes → grow/allocate if needed → single pass populate → optional slabs → version++.
 * Async variant slices the population loop into microtasks to reduce long main‑thread blocks.
 *
 * Example (inspection):
 * ```ts
 * const slab = net.getConnectionSlab();
 * console.log('Edges', slab.used, 'Version', slab.version, 'Cap', slab.capacity);
 * console.log('First weight from->to', slab.weights[0], slab.from[0], slab.to[0]);
 * ```
 */

/**
 * Allocation statistics snapshot for slab typed arrays.
 *
 * Includes:
 *  - fresh: number of newly constructed typed arrays since process start / metrics reset.
 *  - pooled: number of arrays served from the pool.
 *  - pool: per‑key metrics (created, reused, maxRetained) for educational inspection.
 *
 * NOTE: Stats are cumulative (not auto‑reset); callers may diff successive snapshots.
 * @returns Plain object copy (safe to serialize) of current allocator counters.
 */
export function getSlabAllocationStats() {
  // Step 1: Read allocator snapshot from slab pool helper module.
  const allocatorSnapshot = _getSlabAllocationStatsSnapshot();

  // Step 2: Return snapshot for diagnostics tooling.
  return allocatorSnapshot;
}

/**
 * Build (or refresh) the packed connection slabs for the network synchronously.
 *
 * ACTIONS
 * -------
 * 1. Optionally reindex nodes if structural mutations invalidated indices.
 * 2. Grow (geometric) or reuse existing typed arrays to ensure capacity >= active connections.
 * 3. Populate the logical slice [0, connectionCount) with weight/from/to/flag data.
 * 4. Lazily allocate gain & plastic slabs only on first non‑neutral / plastic encounter; omit otherwise.
 * 5. Release previously allocated optional slabs when they revert to neutral / unused (omission optimization).
 * 6. Update internal bookkeeping: logical count, dirty flags, version counter.
 *
 * PERFORMANCE
 * -----------
 * O(C) over active connections with amortized allocation cost due to geometric growth.
 *
 * @param force When true forces rebuild even if network not marked dirty (useful for timing tests).
 */
export function rebuildConnectionSlab(this: Network, force = false): void {
  const buildContext = _createSlabBuildContext(
    this,
    typeof window === 'undefined'
      ? SLAB_GROWTH_FACTOR_NODE
      : SLAB_GROWTH_FACTOR_BROWSER,
  );

  // Step 1: Exit early when slabs are already clean unless forced.
  if (_shouldSkipSlabRebuild(buildContext.internalNet, force)) {
    return;
  }

  // Step 2: Prepare structural prerequisites.
  _prepareSlabBuildPreconditions(buildContext);

  // Step 3: Ensure slab arrays have sufficient capacity.
  _ensureSlabCapacitySync(buildContext);

  // Step 4: Populate core arrays and collect optional-slab signals.
  const populateResult = _populateSlabConnectionsSync(buildContext);

  // Step 5: Apply optional slab policies.
  _applyGainOmissionPolicy(buildContext, populateResult);
  _applyPlasticPolicySync(buildContext, populateResult);

  // Step 6: Finalize rebuild bookkeeping.
  _finalizeSyncSlabRebuild(buildContext);
}

/**
 * Cooperative asynchronous slab rebuild (Browser only).
 *
 * Strategy:
 *  - Perform capacity decision + allocation up front (mirrors sync path).
 *  - Populate connection data in timer-backed macrotask slices so the browser can service other queued work between chunks.
 *  - Adaptive slice sizing for very large graphs if `config.browserSlabChunkTargetMs` set.
 *
 * Metrics: Increments `_slabAsyncBuilds` for observability.
 * Fallback: On Node (no `window`) defers to synchronous rebuild for simplicity.
 *
 * @param chunkSize Initial maximum connections per slice (may be reduced adaptively for huge graphs).
 * @returns Promise resolving once rebuild completes.
 */
export async function rebuildConnectionSlabAsync(
  this: Network,
  chunkSize = SLAB_DEFAULT_ASYNC_CHUNK_SIZE,
): Promise<void> {
  if (typeof window === 'undefined') {
    return rebuildConnectionSlab.call(this, true);
  }

  const buildContext = _createSlabBuildContext(
    this,
    SLAB_GROWTH_FACTOR_BROWSER,
  );

  // Step 1: Exit early when slabs are already current.
  if (_shouldSkipSlabRebuild(buildContext.internalNet, false)) {
    return;
  }

  // Step 2: Prepare structural prerequisites.
  _prepareSlabBuildPreconditions(buildContext);

  // Step 3: Ensure slab arrays and async gain slab are available.
  await _ensureSlabCapacityAsync(buildContext);

  // Step 4: Resolve chunk size and populate slabs in cooperative chunks.
  const resolvedChunkSize = _resolveAsyncChunkSize(
    buildContext.connectionCount,
    chunkSize,
  );
  const populateResult = await _populateSlabConnectionsAsync(
    buildContext,
    resolvedChunkSize,
  );

  // Step 5: Apply optional slab policies.
  _applyGainOmissionPolicy(buildContext, populateResult);
  _applyPlasticPolicyAsync(buildContext, populateResult);

  // Step 6: Finalize async rebuild bookkeeping.
  _finalizeAsyncSlabRebuild(buildContext);
}

/**
 * Obtain (and lazily rebuild if dirty) the current packed SoA view of connections.
 *
 * Gain Omission: If the internal gain slab is absent (all gains neutral) a synthetic
 * neutral array is created and returned (NOT retained) to keep external educational
 * tooling branch‑free while preserving omission memory savings internally.
 *
 * @returns Read‑only style view (do not mutate) containing typed arrays + metadata.
 */
export function getConnectionSlab(this: Network): ConnectionSlabView {
  // Step 1: Ensure slab view reflects current network structure.
  rebuildConnectionSlab.call(this);

  // Step 2: Build and return packed slab view.
  return _createConnectionSlabView(this);
}

/**
 * High‑performance forward pass using packed slabs + CSR adjacency.
 *
 * Fallback Conditions (auto‑detected):
 *  - Missing slabs / adjacency structures.
 *  - Topology/gating/stochastic predicates fail (see `_canUseFastSlab`).
 *  - Gating present, when applicable (explicit guard).
 *
 * Implementation Notes:
 *  - Reuses internal activation/state buffers to reduce per‑step allocation churn.
 *  - Applies gain multiplication if optional gain slab exists.
 *  - Assumes acyclic graph; topological order recomputed on demand if marked dirty.
 *
 * @param input Input vector (length must equal `network.input`).
 * @returns Output activations (detached plain array) of length `network.output`.
 */
export function fastSlabActivate(this: Network, input: number[]): number[] {
  // Step 1: Ensure connection slab arrays are current.
  rebuildConnectionSlab.call(this);

  // Step 2: Ensure adjacency slabs are current.
  const internalNet = this as unknown as NetworkSlabProps;
  if (internalNet._adjDirty) {
    _buildAdjacency(this);
  }

  // Step 3: Delegate fast activation execution.
  return _activateFastSlab(this, input);
}

/**
 * Public convenience wrapper exposing fast path eligibility.
 * Mirrors `_canUseFastSlab` internal predicate.
 * @param training Whether caller is performing training (disables fast path if true).
 * @returns True when slab fast path predicates hold.
 */
export function canUseFastSlab(this: Network, training: boolean): boolean {
  // Step 1: Delegate to fast-path predicate helper.
  return _canUseFastSlab.call(this, training);
}

/**
 * Retrieve current monotonic slab version (increments on each successful rebuild).
 * @returns Non‑negative integer (0 if slab never built yet).
 */
export function getSlabVersion(this: Network): number {
  // Step 1: Delegate slab version read to view helper.
  return _readSlabVersion(this);
}
