/**
 * GPU buffer-set pool for recycling topology buffers across variant evaluations.
 *
 * The pool keeps GPU buffer sets alive across `evaluateWeightVariants` calls so
 * that repeated evaluation of the same network topology does not pay per-iteration
 * allocation and destruction overhead. Buffer sets are keyed by a compound key of
 * topology key and variant count, because different variant counts require
 * independent output state even when the topology is identical.
 *
 * The pool follows the same pattern as TensorFlow.js `BufferManager` and burn's
 * `burn-wgpu` compute server: acquire/release lifecycles with size-keyed handle
 * pools so inference never pays allocation or destruction overhead on the hot
 * path. The default `maxPooledBytes` cap is resolved dynamically by
 * {@link resolveBufferPoolMaxPooledBytes} from the workload node count,
 * average degree, buffer count, and a safety factor, and can be overridden by
 * the caller. This prevents unbounded memory growth when many distinct
 * topologies are evaluated while still leaving enough headroom for typical
 * NEAT networks.
 *
 * @see [TensorFlow.js BufferManager](https://github.com/tensorflow/tfjs/blob/7f5309fef0a47545e34049903dbdae0f97285f7e/tfjs-backend-webgpu/src/buffer_manager.ts)
 * @see [burn-wgpu compute server](https://github.com/tracel-ai/burn/blob/v0.12.1/burn-wgpu/src/compute/server.rs)
 *
 * @module
 */

import type Network from '../network';
import type { GPUBufferSet } from './network.gpu.types';
import { uploadNetworkToGPU } from './network.gpu.buffer';
import {
  DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
  resolveBufferPoolMaxPooledBytes,
} from '../../../acceleration/acceleration.constants';

/**
 * Configuration options for {@link GPUBufferSetPool}.
 *
 * All fields are optional and have sensible defaults. Callers override only the
 * knobs they need to tune.
 */
export interface GPUBufferSetPoolOptions {
  /**
   * Maximum total bytes the pool will retain across all cached buffer sets.
   * When a new acquisition would exceed this cap, free entries are evicted
   * first. Defaults to a dynamic cap resolved by
   * {@link resolveBufferPoolMaxPooledBytes} using the GPU node threshold.
   */
  maxPooledBytes?: number;
}

/**
 * Internal pool entry tracking a buffer set and its in-use status.
 */
interface PoolEntry {
  /** The uploaded GPU buffer set. */
  bufferSet: GPUBufferSet;
  /** Whether this entry is currently acquired (in use) or released (free). */
  inUse: boolean;
  /** Estimated total bytes occupied by this buffer set. */
  bytes: number;
}

/**
 * Compute the compound pool key from a topology key and variant count.
 *
 * Different variant counts produce distinct keys so that independent output
 * state is maintained even when the topology is identical.
 *
 * @param topologyKey - Deterministic string identifying the network topology.
 * @param variantCount - Number of weight variants being evaluated.
 * @returns Compound key string.
 */
function makePoolKey(topologyKey: string, variantCount: number): string {
  return `${topologyKey}:${variantCount}`;
}

/**
 * Estimate the total GPU byte footprint of a buffer set.
 *
 * Sums the `size` property of every buffer in the set. This is used for
 * the `maxPooledBytes` cap enforcement.
 *
 * @param set - The buffer set to measure.
 * @returns Total bytes across all six buffers.
 */
function computeBufferSetBytes(set: GPUBufferSet): number {
  return (
    set.connections.size +
    set.nodes.size +
    set.outputs.size +
    set.params.size +
    set.topoLevels.size +
    set.inStart.size
  );
}

/**
 * Recycle pool for GPU buffer sets used during NGE weight-variant evaluation.
 *
 * The pool caches {@link GPUBufferSet} instances keyed by a compound key of
 * topology key and variant count. When a buffer set is released, it is kept
 * in the pool for reuse by the next acquisition with the same key, avoiding
 * the cost of destroying and re-creating six GPU buffers per evaluation
 * iteration.
 *
 * The pool enforces a configurable `maxPooledBytes` cap to prevent unbounded
 * memory growth. When a new acquisition would exceed the cap, free entries
 * are evicted (their buffers destroyed) until the new set fits.
 *
 * @example
 * ```ts
 * const pool = new GPUBufferSetPool({ maxPooledBytes: 32 * 1024 * 1024 });
 * const bufferSet = pool.acquire(device, 'topo-2-3-1', network, 16);
 * // ... use bufferSet for variant evaluation ...
 * pool.release('topo-2-3-1');
 * // Next acquire with the same key reuses the same buffer set
 * const reused = pool.acquire(device, 'topo-2-3-1', network, 16);
 * pool.destroy();
 * ```
 */
export class GPUBufferSetPool {
  /** Maximum total bytes the pool will retain across all cached buffer sets. */
  readonly maxPooledBytes: number;

  /** Internal storage mapping compound keys to pool entries. */
  private readonly pool = new Map<string, PoolEntry>();

  /** Running total of bytes occupied by all pooled buffer sets. */
  private totalPooledBytes = 0;

  /**
   * Create a new GPU buffer-set pool.
   *
   * @param options - Optional configuration overrides. When omitted, all
   *   defaults are used. The default `maxPooledBytes` is resolved dynamically
   *   from {@link resolveBufferPoolMaxPooledBytes}.
   */
  constructor(options?: GPUBufferSetPoolOptions) {
    this.maxPooledBytes =
      options?.maxPooledBytes ??
      resolveBufferPoolMaxPooledBytes({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
        variantCount: 1,
      });
  }

  /**
   * Acquire a GPU buffer set for the given topology and variant count.
   *
   * If a free (released) buffer set exists for the same compound key, it is
   * reused without reallocation. Otherwise, a new buffer set is uploaded
   * from the network via `uploadNetworkToGPU`. When the new set would
   * exceed `maxPooledBytes`, free entries are evicted first.
   *
   * @param device - WebGPU device used to allocate buffers when a new set
   *   is needed.
   * @param topologyKey - Deterministic string identifying the network
   *   topology.
   * @param network - Network whose slab will be uploaded when a new buffer
   *   set is required.
   * @param variantCount - Number of weight variants being evaluated. Used
   *   as part of the compound key so different variant counts get distinct
   *   buffer sets.
   * @returns A GPU buffer set with `nodeCount` matching the network.
   * @throws Error when the network is not eligible for GPU upload.
   */
  acquire(
    device: GPUDevice,
    topologyKey: string,
    network: Network,
    variantCount: number,
  ): GPUBufferSet {
    const key = makePoolKey(topologyKey, variantCount);
    const existing = this.pool.get(key);

    if (existing !== undefined) {
      existing.inUse = true;
      return existing.bufferSet;
    }

    // Step 1: Evict free entries if the new set would exceed the byte cap.
    const projectedBytes = this.estimateBufferSetBytes(network);
    if (this.totalPooledBytes + projectedBytes > this.maxPooledBytes) {
      this.evictFreeEntries(projectedBytes);
    }

    // Step 2: Upload the network to GPU buffers.
    const bufferSet = uploadNetworkToGPU(device, network);
    const bytes = computeBufferSetBytes(bufferSet);

    // Step 3: Store the new entry in the pool.
    this.pool.set(key, { bufferSet, inUse: true, bytes });
    this.totalPooledBytes += bytes;

    return bufferSet;
  }

  /**
   * Release all buffer sets matching the given topology key back to the pool.
   *
   * Released sets remain in the pool and are available for reuse by a
   * subsequent `acquire` with the same compound key. The variant count is
   * not needed because all variant counts for the given topology are
   * released together.
   *
   * @param topologyKey - Topology key prefix to match.
   */
  release(topologyKey: string): void {
    const prefix = `${topologyKey}:`;
    for (const [key, entry] of this.pool) {
      if (key.startsWith(prefix)) {
        entry.inUse = false;
      }
    }
  }

  /**
   * Destroy all pooled buffer sets and clear the pool.
   *
   * After `destroy`, the pool is empty (`size` is 0) and all GPU buffers
   * have been released. The pool can continue to be used for new acquisitions
   * after destruction.
   */
  destroy(): void {
    for (const [, entry] of this.pool) {
      destroyBufferSet(entry.bufferSet);
    }
    this.pool.clear();
    this.totalPooledBytes = 0;
  }

  /**
   * Number of buffer sets currently held in the pool (both in-use and free).
   */
  get size(): number {
    return this.pool.size;
  }

  /**
   * Estimate the byte footprint of a buffer set for the given network.
   *
   * Uses the struct-packed layout: nodes are 16 bytes each, connections are
   * 16 bytes each, outputs are 4 bytes per node, the params uniform is 16
   * bytes, topo levels are 4 bytes per node, and in-start offsets are 4 bytes
   * per node plus one.
   *
   * @param network - Network to estimate buffer sizes for.
   * @returns Estimated total bytes for the six-buffer set.
   */
  private estimateBufferSetBytes(network: Network): number {
    const nodeCount = network.nodes.length;
    const connectionCount = network.connections.length;
    const connectionsBytes = connectionCount * 16;
    const nodesBytes = nodeCount * 16;
    const outputsBytes = nodeCount * 4;
    const paramsBytes = 16;
    const topoLevelsBytes = nodeCount * 4;
    const inStartBytes = (nodeCount + 1) * 4;
    return (
      connectionsBytes +
      nodesBytes +
      outputsBytes +
      paramsBytes +
      topoLevelsBytes +
      inStartBytes
    );
  }

  /**
   * Evict free entries from the pool until the projected new set fits.
   *
   * Only entries that are not currently in use are evicted. Their GPU buffers
   * are destroyed and their bytes are reclaimed from the running total.
   * Eviction stops as soon as the projected set fits under the cap, so only
   * the minimum number of free entries are reclaimed.
   *
   * This is a single-pass eviction: entries are destroyed and removed from the
   * pool inline so that `totalPooledBytes` is updated during iteration. This
   * ensures the break condition becomes reachable after evicting one entry,
   * rather than collecting all candidates first and evicting them all
   * regardless of whether fewer would suffice.
   *
   * Map deletion during iteration is safe per the ECMAScript spec: deleting
   * the current entry does not skip subsequent entries.
   *
   * @param projectedBytes - Bytes needed for the new buffer set.
   */
  private evictFreeEntries(projectedBytes: number): void {
    for (const [key, entry] of this.pool) {
      if (this.totalPooledBytes + projectedBytes <= this.maxPooledBytes) {
        break;
      }
      if (!entry.inUse) {
        destroyBufferSet(entry.bufferSet);
        this.totalPooledBytes -= entry.bytes;
        this.pool.delete(key);
      }
    }
  }
}

/**
 * Destroy every GPU buffer in a buffer set by calling `.destroy()` on each.
 *
 * This is a local helper that mirrors `destroyGPUBufferSet` from
 * `network.gpu.buffer` but does not require a device reference, since the
 * WebGPU `GPUBuffer.destroy()` method is self-contained.
 *
 * @param set - The buffer set whose buffers should be destroyed.
 */
function destroyBufferSet(set: GPUBufferSet): void {
  set.connections.destroy();
  set.nodes.destroy();
  set.outputs.destroy();
  set.params.destroy();
  set.topoLevels.destroy();
  set.inStart.destroy();
}
