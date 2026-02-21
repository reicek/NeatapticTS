import type Network from '../../network';
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
import {
  _canUseFastSlab,
  _tryFastSlabFallbackForGating,
  _tryFastSlabFallbackForMissingPrerequisites,
  _prepareFastSlabRuntime,
  _resolveFastTopoOrder,
  _ensureFastSlabBuffers,
  _seedFastInputLayer,
  _propagateFastSlabActivations,
  _collectFastSlabOutput,
} from './network.slab.fast-path.helpers.utils';
import { _buildAdjacency } from './network.slab.adjacency.helpers.utils';
import { _reindexNodes } from './network.slab.shared.helpers.utils';
import type {
  NetworkSlabProps,
  SlabBuildContext,
  ConnectionSlabView,
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
 * const slab = (net as any).getConnectionSlab();
 * console.log('Edges', slab.used, 'Version', slab.version, 'Cap', slab.capacity);
 * console.log('First weight from->to', slab.weights[0], slab.from[0], slab.to[0]);
 * ```
 */

const ZERO = 0;
const ONE = 1;
const GROWTH_FACTOR_NODE = 1.75;
const GROWTH_FACTOR_BROWSER = 1.25;
const DEFAULT_ASYNC_CHUNK_SIZE = 50_000;

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
 * Applies prerequisite normalization for slab rebuild passes.
 *
 * @param buildContext - Slab build context.
 * @returns Nothing.
 */
function _prepareSlabBuildPreconditions(buildContext: SlabBuildContext): void {
  // Step 1: Rebuild node indices when structural mutations invalidated ordering.
  if (buildContext.internalNet._nodeIndexDirty) {
    _reindexNodes(buildContext.network);
  }
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
    typeof window === 'undefined' ? GROWTH_FACTOR_NODE : GROWTH_FACTOR_BROWSER,
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
 *  - Populate connection data in microtask slices (yield via resolved Promise) to avoid long main‑thread stalls.
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
  chunkSize = DEFAULT_ASYNC_CHUNK_SIZE,
): Promise<void> {
  if (typeof window === 'undefined') {
    return rebuildConnectionSlab.call(this, true);
  }

  const buildContext = _createSlabBuildContext(this, GROWTH_FACTOR_BROWSER);

  // Step 1: Exit early when slabs are already current.
  if (_shouldSkipSlabRebuild(buildContext.internalNet, false)) {
    return;
  }

  // Step 2: Prepare structural prerequisites.
  _prepareSlabBuildPreconditions(buildContext);

  // Step 3: Ensure slab arrays and async gain slab are available.
  _ensureSlabCapacityAsync(buildContext);

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
  rebuildConnectionSlab.call(this); // Lazy rebuild if needed.
  const internalNet = this as unknown as NetworkSlabProps;
  let gain: Float32Array | Float64Array | null = internalNet._connGain || null;
  if (!gain) {
    // Provide a synthetic neutral gain view for educational/tests expecting parity while preserving omission semantics.
    const cap =
      internalNet._connCapacity ||
      (internalNet._connWeights && internalNet._connWeights.length) ||
      ZERO;
    gain = internalNet._useFloat32Weights
      ? new Float32Array(cap)
      : new Float64Array(cap);
    for (
      let connectionIndex = ZERO;
      connectionIndex < (internalNet._connCount || ZERO);
      connectionIndex++
    ) {
      gain[connectionIndex] = ONE;
    }
  }
  return {
    weights: internalNet._connWeights!,
    from: internalNet._connFrom!,
    to: internalNet._connTo!,
    flags: internalNet._connFlags!,
    gain,
    plastic: internalNet._connPlastic || null,
    version: internalNet._slabVersion || ZERO,
    used: internalNet._connCount || ZERO,
    capacity:
      internalNet._connCapacity ||
      (internalNet._connWeights && internalNet._connWeights.length) ||
      ZERO,
  };
}

/**
 * High‑performance forward pass using packed slabs + CSR adjacency.
 *
 * Fallback Conditions (auto‑detected):
 *  - Missing slabs / adjacency structures.
 *  - Topology/gating/stochastic predicates fail (see `_canUseFastSlab`).
 *  - Any gating present (explicit guard).
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
  const internalNet = this as unknown as NetworkSlabProps;
  rebuildConnectionSlab.call(this);
  if (internalNet._adjDirty) {
    _buildAdjacency(this);
  }

  const gatedFallback = _tryFastSlabFallbackForGating(this, input);
  if (gatedFallback) {
    return gatedFallback;
  }

  const missingPrerequisiteFallback =
    _tryFastSlabFallbackForMissingPrerequisites(this, internalNet, input);
  if (missingPrerequisiteFallback) {
    return missingPrerequisiteFallback;
  }

  _prepareFastSlabRuntime(this, internalNet, (network) => {
    _reindexNodes(network);
  });
  const topoOrder = _resolveFastTopoOrder(this, internalNet);
  const nodeCount = this.nodes.length;

  _ensureFastSlabBuffers(internalNet, nodeCount);
  const activationBuffer = internalNet._fastA as Float32Array | Float64Array;
  const stateBuffer = internalNet._fastS as Float32Array | Float64Array;
  stateBuffer.fill(ZERO);

  _seedFastInputLayer(this, input, activationBuffer);
  _propagateFastSlabActivations(
    this,
    internalNet,
    topoOrder,
    activationBuffer,
    stateBuffer,
  );
  return _collectFastSlabOutput(this, activationBuffer, nodeCount);
}

/**
 * Public convenience wrapper exposing fast path eligibility.
 * Mirrors `_canUseFastSlab` internal predicate.
 * @param training Whether caller is performing training (disables fast path if true).
 * @returns True when slab fast path predicates hold.
 */
export function canUseFastSlab(this: Network, training: boolean) {
  // Step 1: Delegate to fast-path predicate helper.
  return _canUseFastSlab.call(this, training);
}

/**
 * Retrieve current monotonic slab version (increments on each successful rebuild).
 * @returns Non‑negative integer (0 if slab never built yet).
 */
export function getSlabVersion(this: Network): number {
  // Step 1: Read monotonic slab version with zero fallback.
  const internalNet = this as unknown as NetworkSlabProps;
  return internalNet._slabVersion || 0;
}
