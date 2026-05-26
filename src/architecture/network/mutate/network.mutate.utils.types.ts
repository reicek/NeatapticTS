import type Node from '../../node';

/**
 * Canonical node-type literal for input nodes used as a discriminant in mutation eligibility guards.
 */
export const NODE_TYPE_INPUT: Node['type'] = 'input';

/**
 * Canonical node-type literal for output nodes used as a discriminant in mutation eligibility guards.
 */
export const NODE_TYPE_OUTPUT: Node['type'] = 'output';

/**
 * Canonical node-type literal for hidden nodes used as a discriminant in mutation eligibility guards.
 */
export const NODE_TYPE_HIDDEN: Node['type'] = 'hidden';

/**
 * Canonical recurrent block literal for LSTM expansion used when inserting a long short-term memory block.
 */
export const RECURRENT_BLOCK_LSTM = 'lstm' as const;

/**
 * Canonical recurrent block literal for GRU expansion used when inserting a gated recurrent unit block.
 */
export const RECURRENT_BLOCK_GRU = 'gru' as const;

/**
 * Width used when creating a minimal single-unit recurrent block during LSTM or GRU insertion mutations.
 */
export const SINGLE_UNIT_RECURRENT_BLOCK_WIDTH = 1;

/**
 * Small weight delta applied to sub-node connections to keep mutation side effects numerically observable and non-degenerate.
 */
export const SUB_NODE_STABILITY_WEIGHT_DELTA = 1e-4;

/**
 * Probability threshold used for random 50/50 gating decisions during gate-reassignment mutation passes.
 */
export const GATE_REASSIGN_THRESHOLD = 0.5;

/**
 * Default minimum mutation perturbation value applied when the active mutation method provides no explicit range override.
 */
export const DEFAULT_MUTATION_MIN = -1;

/**
 * Default maximum mutation perturbation value applied when the active mutation method provides no explicit range override.
 */
export const DEFAULT_MUTATION_MAX = 1;

/**
 * Minimum redundant in-degree or out-degree required on both endpoints before a connection may be safely removed.
 */
export const MIN_REDUNDANT_CONNECTION_COUNT = 1;

/**
 * Minimum number of nodes that must exist in the network before swap-node mutation can safely select two distinct candidates.
 */
export const MIN_SWAPPABLE_NODE_COUNT = 2;

/**
 * Warning message emitted when remove-node mutation finds no hidden nodes eligible for removal in the current topology.
 */
export const WARNING_NO_HIDDEN_NODES_TO_REMOVE =
  'No hidden nodes left to remove!';

/**
 * Warning message emitted when activation mutation finds no nodes eligible for squash-function replacement based on current config.
 */
export const WARNING_NO_ACTIVATION_MUTATION_TARGETS =
  'No nodes available for activation function mutation based on config.';

/**
 * Warning message emitted when all self-connection candidates are already occupied and no new self-connection can be added.
 */
export const WARNING_SELF_CONNECTIONS_ALREADY_PRESENT =
  'All eligible nodes already have self-connections.';

/**
 * Warning message emitted when remove-self-connection mutation finds no eligible self-connections to remove from the network.
 */
export const WARNING_NO_SELF_CONNECTIONS_TO_REMOVE =
  'No self-connections exist to remove.';

/**
 * Warning message emitted when add-gate mutation cannot proceed because all eligible connections are already gated.
 */
export const WARNING_ALL_CONNECTIONS_GATED =
  'All connections are already gated.';

/**
 * Warning message emitted when remove-gate mutation finds no gated connections eligible to ungate in the current network.
 */
export const WARNING_NO_GATED_CONNECTIONS_TO_REMOVE =
  'No gated connections to ungate.';

/**
 * Log message prefix used when the mutate dispatcher encounters and discards an unrecognized mutation method identifier.
 */
export const UNKNOWN_MUTATION_WARNING_PREFIX =
  '[mutate] Unknown mutation method ignored:';

/**
 * Error message thrown when the mutate dispatcher is called without a recognized or valid mutation method argument.
 */
export const ERROR_NO_MUTATE_METHOD = 'No (correct) mutate method given!';

/**
 * Internal node property key used to flag batch normalization participation during forward-pass activation computation.
 */
export const BATCH_NORM_FLAG_KEY = '_batchNorm';
