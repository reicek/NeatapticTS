/** Default Node-side SQLite file used for durable NeatChat memory. */
export const DEFAULT_DB_PATH = 'data/neatchat-memory.sqlite';

/** Hard cap for durable entries retained per session before LRU pruning. */
export const MAX_DURABLE_ENTRIES = 1_000;

/** Default maximum number of ranked durable memories returned per query. */
export const DEFAULT_MEMORY_RESULT_LIMIT = 5;

/** Default time-to-live window for durable memory entries. */
export const DEFAULT_MEMORY_TTL_MS = 30 * 24 * 60 * 60 * 1_000;

/** Okapi BM25 saturation parameter used by local ranking. */
export const MEMORY_BM25_K1 = 1.2;

/** Okapi BM25 length-normalization parameter used by local ranking. */
export const MEMORY_BM25_B = 0.75;

/** Default IndexedDB database name for browser durable memory. */
export const IDB_MEMORY_DATABASE_NAME = 'neatchat-memory';

/** Default IndexedDB object store for browser durable memory records. */
export const IDB_MEMORY_OBJECT_STORE_NAME = 'neatchat_memory';
