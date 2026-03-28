import type Node from '../../node';

/**
 * Canonical node-type literal for input nodes.
 */
export const NODE_TYPE_INPUT: Node['type'] = 'input';

/**
 * Canonical node-type literal for output nodes.
 */
export const NODE_TYPE_OUTPUT: Node['type'] = 'output';

/**
 * Canonical node-type literal for hidden nodes.
 */
export const NODE_TYPE_HIDDEN: Node['type'] = 'hidden';

/**
 * Canonical recurrent block literal for LSTM expansion.
 */
export const RECURRENT_BLOCK_LSTM = 'lstm' as const;

/**
 * Canonical recurrent block literal for GRU expansion.
 */
export const RECURRENT_BLOCK_GRU = 'gru' as const;

/**
 * Width used when creating a minimal recurrent block.
 */
export const SINGLE_UNIT_RECURRENT_BLOCK_WIDTH = 1;

/**
 * Weight delta used to keep mutation side effects numerically observable.
 */
export const SUB_NODE_STABILITY_WEIGHT_DELTA = 1e-4;

/**
 * Threshold used for random 50/50 gating decisions.
 */
export const GATE_REASSIGN_THRESHOLD = 0.5;

/**
 * Default minimum mutation value when no method override is provided.
 */
export const DEFAULT_MUTATION_MIN = -1;

/**
 * Default maximum mutation value when no method override is provided.
 */
export const DEFAULT_MUTATION_MAX = 1;

/**
 * Minimum redundant in/out degree required before removing a connection.
 */
export const MIN_REDUNDANT_CONNECTION_COUNT = 1;

/**
 * Minimum node count required to perform swap-node mutation.
 */
export const MIN_SWAPPABLE_NODE_COUNT = 2;

/**
 * Message emitted when no hidden node can be removed.
 */
export const WARNING_NO_HIDDEN_NODES_TO_REMOVE =
  'No hidden nodes left to remove!';

/**
 * Message emitted when activation mutation has no eligible nodes.
 */
export const WARNING_NO_ACTIVATION_MUTATION_TARGETS =
  'No nodes available for activation function mutation based on config.';

/**
 * Message emitted when all self-connection candidates are already occupied.
 */
export const WARNING_SELF_CONNECTIONS_ALREADY_PRESENT =
  'All eligible nodes already have self-connections.';

/**
 * Message emitted when no self-connections are available to remove.
 */
export const WARNING_NO_SELF_CONNECTIONS_TO_REMOVE =
  'No self-connections exist to remove.';

/**
 * Message emitted when gating cannot be added because all are already gated.
 */
export const WARNING_ALL_CONNECTIONS_GATED =
  'All connections are already gated.';

/**
 * Message emitted when no gate exists to remove.
 */
export const WARNING_NO_GATED_CONNECTIONS_TO_REMOVE =
  'No gated connections to ungate.';

/**
 * Prefix for unknown-mutation warning logs.
 */
export const UNKNOWN_MUTATION_WARNING_PREFIX =
  '[mutate] Unknown mutation method ignored:';

/**
 * Error emitted when mutate is called without a valid method.
 */
export const ERROR_NO_MUTATE_METHOD = 'No (correct) mutate method given!';

/**
 * Internal node field used to enable batch normalization.
 */
export const BATCH_NORM_FLAG_KEY = '_batchNorm';
