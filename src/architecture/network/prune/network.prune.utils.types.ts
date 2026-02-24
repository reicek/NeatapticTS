import type { NetworkPruningProps } from '../network.types';

/** Pruning method identifier for absolute-weight ranking. */
export const PRUNING_METHOD_MAGNITUDE = 'magnitude' as const;
/** Pruning method identifier for SNIP-like saliency ranking. */
export const PRUNING_METHOD_SNIP = 'snip' as const;
/** Lower bound to ensure at least one connection remains after pruning. */
export const MIN_REMAINING_CONNECTION_COUNT = 1;
/** Fallback prune cadence when schedule frequency is omitted or invalid. */
export const DEFAULT_PRUNE_FREQUENCY = 1;
/** Minimum normalized schedule progress value. */
export const MIN_PROGRESS_FRACTION = 0;
/** Maximum normalized schedule progress value. */
export const MAX_PROGRESS_FRACTION = 1;
/** Safety cap below full sparsity to avoid degenerate zero-connection networks. */
export const MAX_EVOLUTIONARY_TARGET_SPARSITY = 0.999;
/** Retry multiplier to convert intended regrowth count into max attempts. */
export const REGROW_ATTEMPT_MULTIPLIER = 10;

/** Non-nullable pruning schedule configuration shape used by helpers. */
export type ActivePruningConfig = NonNullable<
  NetworkPruningProps['_pruningConfig']
>;
