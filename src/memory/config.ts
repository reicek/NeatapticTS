import type { NeatapticConfig } from '../config';

/**
 * Default retained prewarm count for activation-pool warmup when callers leave
 * the global memory config unset.
 */
export const MEMORY_DEFAULT_ACTIVATION_POOL_PREWARM_COUNT = 2;

/**
 * Default retained slab-buffer count per `(kind, bytes, length)` pool key.
 */
export const MEMORY_DEFAULT_SLAB_POOL_MAX_PER_KEY = 4;

/**
 * Browser slab-growth factor used to trade smaller reallocations for lower
 * retained memory pressure in constrained heaps.
 */
export const MEMORY_SLAB_GROWTH_FACTOR_BROWSER = 1.25;

/**
 * Node slab-growth factor used to reduce rebuild churn on larger server heaps.
 */
export const MEMORY_SLAB_GROWTH_FACTOR_NODE = 1.75;

/**
 * Memory-focused config flags owned by the Phase 4 manager surface.
 */
export const MEMORY_MANAGER_FLAG_NAMES = [
  'warnings',
  'float32Mode',
  'deterministicChainMode',
  'enableGatingTraces',
  'poolMaxPerBucket',
  'poolPrewarmCount',
  'enableNodePooling',
  'enableSlabArrayPooling',
  'browserSlabChunkTargetMs',
  'nodeHeapSoftLimitMB',
  'browserMemoryBudgetMB',
  'slabPoolMaxPerKey',
] as const;

/**
 * Narrow flag names controlled by the memory manager.
 */
export type MemoryManagerFlagName = (typeof MEMORY_MANAGER_FLAG_NAMES)[number];

/**
 * Subset of the shared config object that materially changes memory behavior.
 */
export type MemoryManagerFlagMap = Pick<NeatapticConfig, MemoryManagerFlagName>;

/**
 * Runtime environment labels used by the manager when resolving defaults.
 */
export type MemoryManagerEnvironment = 'browser' | 'node';

/**
 * Stable snapshot returned by `MemoryManager.getConfig()`.
 */
export interface MemoryManagerConfigSnapshot extends Omit<
  MemoryManagerFlagMap,
  'browserMemoryBudgetMB' | 'nodeHeapSoftLimitMB' | 'slabPoolMaxPerKey'
> {
  /** Active runtime environment used to resolve environment-sensitive defaults. */
  environment: MemoryManagerEnvironment;
  /** Effective prewarm count after default resolution. */
  activationPoolPrewarmCount: number;
  /** Effective browser used-heap soft target in megabytes. */
  browserMemoryBudgetMB?: number;
  /** Effective Node heap-used soft target in megabytes. */
  nodeHeapSoftLimitMB?: number;
  /** Effective slab-pool cap after truncation and clamping. */
  slabPoolMaxPerKey: number;
  /** Effective slab growth factor for the current runtime. */
  slabGrowthFactor: number;
}

/**
 * Registry shape used by the manager for resettable memory-owned pools.
 */
export interface RegisteredMemoryPool {
  /** Reset hook invoked during manager teardown. */
  reset?: () => void;
  /** Snapshot hook used by diagnostics paths. */
  stats?: () => unknown;
}
