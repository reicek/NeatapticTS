/** Supported durable memory entry categories for the NeatChat demo. */
export type StoredMemoryEntryType = 'exchange' | 'association' | 'correction';

/**
 * One durable memory entry persisted outside the in-process W3 memory bank.
 *
 * Durable entries survive across sessions and act as the retrieval surface for
 * the local SQLite or IndexedDB adapters.
 */
export interface StoredMemoryEntry {
  /** Stable durable identifier for one stored entry. */
  readonly entryId: string;
  /** Session owner for the durable entry. */
  readonly sessionId: string;
  /** Durable memory category used for downstream filtering. */
  readonly entryType: StoredMemoryEntryType;
  /** Persisted text payload used for retrieval and export. */
  readonly content: string;
  /** Pre-tokenized normalized terms used by local ranking. */
  readonly tokens: readonly string[];
  /** Optional reinforcement weight carried by the durable store. */
  readonly score: number;
  /** Epoch milliseconds when the entry was first stored. */
  readonly createdAt: number;
  /** Epoch milliseconds of the most recent retrieval or write touch. */
  readonly lastUsed: number;
}

/** Input payload required to persist one durable memory entry. */
export type CreateStoredMemoryEntry = Omit<StoredMemoryEntry, 'entryId'>;

/** Query contract for ranked durable-memory retrieval. */
export interface MemoryQuery {
  /** Session owner whose durable entries should be searched. */
  readonly sessionId: string;
  /** Raw user query used to retrieve relevant memory context. */
  readonly query: string;
  /** Maximum number of ranked results to return. */
  readonly maxResults?: number;
}

/** Query contract with an explicit result limit for adapter-level search. */
export type MemorySearchQuery = Required<MemoryQuery>;

/**
 * Ranked durable-memory result returned to the session enrichment layer.
 *
 * `bm25Score` and `overlapScore` are surfaced separately so tests and later
 * diagnostics can explain why one entry outranked another.
 */
export interface MemoryResult extends StoredMemoryEntry {
  /** Local BM25 score for the query against this entry. */
  readonly bm25Score: number;
  /** Count of overlapping normalized query tokens. */
  readonly overlapScore: number;
  /** Combined relevance score used for final ranking. */
  readonly relevanceScore: number;
}

/**
 * Storage adapter contract for durable NeatChat memory.
 *
 * Tests can satisfy this interface with a tiny in-memory adapter, while the
 * real runtime uses SQLite in Node and IndexedDB in the browser.
 */
export interface MemoryAdapter {
  /** Persists one durable memory entry and resolves the generated identifier. */
  store(entry: CreateStoredMemoryEntry): Promise<string>;
  /** Lists all durable entries for one session. */
  list(sessionId: string): Promise<readonly StoredMemoryEntry[]>;
  /** Removes durable entries by identifier and returns the deleted count. */
  remove(entryIds: readonly string[]): Promise<number>;
  /** Optional adapter-native ranked retrieval path. */
  search?(query: MemorySearchQuery): Promise<readonly MemoryResult[]>;
  /** Optional cleanup hook for adapters that own external resources. */
  close?(): Promise<void>;
}
