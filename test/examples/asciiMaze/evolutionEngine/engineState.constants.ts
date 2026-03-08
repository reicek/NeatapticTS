/**
 * Constant values shared by the ASCII maze engine-state facade and its helper utilities.
 */

/** Default logits ring length used when allocating pooled softmax buffers. */
export const DEFAULT_LOGITS_RING_CAPACITY = 512;
/** Number of action outputs (N, E, S, W) represented in each logits row. */
export const ACTION_OUTPUT_DIMENSION = 4;
/** Default load factor target for the visited coordinate hash table. */
export const DEFAULT_VISITED_HASH_LOAD_FACTOR = 0.7;
/** Minimum safe load factor applied when normalising visited-hash configuration. */
export const MIN_VISITED_HASH_LOAD_FACTOR = 0.1;
/** Maximum safe load factor applied when normalising visited-hash configuration. */
export const MAX_VISITED_HASH_LOAD_FACTOR = 0.95;
/** Default RNG cache batch size mirroring the façade constant. */
export const DEFAULT_RNG_CACHE_BATCH_SIZE = 4;
/** Default capacity reserved for species identifier scratch arrays. */
export const DEFAULT_SPECIES_SCRATCH_CAPACITY = 64;
/** Default capacity reserved for connection flag buffers. */
export const DEFAULT_CONNECTION_FLAG_CAPACITY = 128;
/** Default capacity reused by history and sampling scratch arrays. */
export const DEFAULT_HISTORY_BUFFER_CAPACITY = 64;
/** Default capacity reserved for sorted index scratch arrays. */
export const DEFAULT_SORTED_INDEX_CAPACITY = 512;
/** Default stack depth reserved for quicksort range storage. */
export const DEFAULT_QUICKSORT_STACK_CAPACITY = 128;
/** Default pool size for telemetry sampling helpers. */
export const DEFAULT_SAMPLE_POOL_SIZE = 40;
/** Default capacity for telemetry string assembly buffers. */
export const DEFAULT_STRING_BUFFER_CAPACITY = 64;
/** Default capacity for the small exploration table scratch. */
export const DEFAULT_SMALL_EXPLORE_TABLE_CAPACITY = 64;
/** Default capacity for the node index buffer used during inspection. */
export const DEFAULT_NODE_INDEX_BUFFER_CAPACITY = 64;
/** Knuth-derived 32-bit constant used when seeding the RNG state. */
export const RNG_GOLDEN_RATIO_SEED = 0x9e3779b9;
