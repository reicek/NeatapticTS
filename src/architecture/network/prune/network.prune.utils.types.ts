import type { NetworkPruningProps } from '../network.types';

/**
 * Pruning method identifier for absolute-weight ranking (`|w|`) used by
 * default scheduled and evolutionary pruning selection helpers.
 */
export const PRUNING_METHOD_MAGNITUDE = 'magnitude' as const;
/**
 * Pruning method identifier for SNIP-like saliency ranking (`|w * g|` proxy).
 */
export const PRUNING_METHOD_SNIP = 'snip' as const;
/**
 * Lower bound that guarantees at least one connection remains after pruning.
 */
export const MIN_REMAINING_CONNECTION_COUNT = 1;
/**
 * Default prune cadence used when schedule frequency is absent or invalid.
 */
export const DEFAULT_PRUNE_FREQUENCY = 1;
/**
 * Minimum normalized schedule progress value used by clamp logic before
 * deriving sparsity targets inside schedule helpers.
 */
export const MIN_PROGRESS_FRACTION = 0;
/**
 * Maximum normalized schedule progress value used by clamp logic before
 * deriving sparsity targets inside schedule helpers.
 */
export const MAX_PROGRESS_FRACTION = 1;
/**
 * Safety cap below full sparsity to avoid degenerate zero-connection networks.
 */
export const MAX_EVOLUTIONARY_TARGET_SPARSITY = 0.999;
/**
 * Retry multiplier used to translate desired regrowth count into max attempts.
 */
export const REGROW_ATTEMPT_MULTIPLIER = 10;

/**
 * Normalized pruning-config shape consumed by internal helpers after boundary
 * guards confirm that runtime options are present.
 */
export type ActivePruningConfig = NonNullable<
  NetworkPruningProps['_pruningConfig']
>;
