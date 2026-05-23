/**
 * A single episodic memory record keyed by a short text label.
 *
 * Each record captures one user-specific fact or context fragment so it can be
 * retrieved later by prompt-token overlap and injected into reply generation.
 * Records are immutable once written; the bank itself is updated by returning
 * a new bank value rather than mutating in place.
 *
 * @remarks
 * - `key` — short label or prompt fragment used as the retrieval surface.
 * - `value` — associated content or response fragment stored alongside the key.
 * - `addedAt` — epoch millisecond timestamp set at insertion time.
 * - `hitCount` — incremented each time the record appears in a retrieval result;
 *   low-hit records are pruned first when the bank reaches capacity.
 */
export type NeatChatMemoryRecord = {
  /** Short label or prompt fragment used to retrieve the memory. */
  readonly key: string;
  /** Associated content or response fragment stored with the label. */
  readonly value: string;
  /** Epoch milliseconds when the record was first added. */
  readonly addedAt: number;
  /** Number of retrievals that have matched this record. */
  readonly hitCount: number;
};

/**
 * The episodic memory bank attached to a live NEATchat session.
 *
 * An episodic memory bank holds a bounded list of user-specific fact records
 * that survive across exchanges within the same session and can be
 * checkpointed as part of a session snapshot. Unlike the recurrent hidden
 * state (short-term memory), episodic records are explicitly keyed,
 * retrievable by prompt-token overlap, and persist through export and
 * re-import.
 *
 * @remarks
 * - `maxRecords` — hard cap on stored records; when the cap is reached, the
 *   record with the lowest `hitCount` (oldest first on ties) is dropped
 *   before a new record is appended.
 * - `lastConsolidatedAt` — epoch milliseconds of the most recent
 *   consolidation pass; `null` when the bank has never been consolidated.
 */
export type NeatChatEpisodicMemoryBank = {
  /** Stored episodic memory records in insertion order after pruning. */
  readonly records: readonly NeatChatMemoryRecord[];
  /** Maximum number of records kept before pruning is required. */
  readonly maxRecords: number;
} &
  (
    | {
        /** Epoch milliseconds of the most recent consolidation pass, if any. */
        readonly lastConsolidatedAt: number | null;
      }
    | {
        /** Optional compatibility form used by older owner-local fixtures. */
        readonly lastConsolidatedAt?: number | null;
      }
  );

/**
 * Options for creating an empty NEATchat episodic memory bank.
 *
 * When `maxRecords` is omitted the bank defaults to
 * `NEATCHAT_DEFAULT_MEMORY_BANK_MAX_RECORDS` (100). Provide an explicit value
 * when the session has strict memory-pressure requirements or needs a larger
 * working set for long-running personalized interactions.
 */
export type CreateNeatChatEpisodicMemoryBankOptions = {
  /** Optional explicit cap for stored episodic memory records. */
  readonly maxRecords?: number;
};

/**
 * Options for ranked episodic-memory retrieval.
 *
 * When `maxResults` is omitted the retrieval function returns every record
 * that has at least one overlapping token with the prompt, ordered by overlap
 * count descending and insertion time ascending. Setting `maxResults` limits
 * the result slice to the top-ranked records, which is the recommended
 * default for response-generation use cases.
 */
export type RetrieveNeatChatMemoriesOptions = {
  /** Optional maximum number of ranked memories to return. */
  readonly maxResults?: number;
};

/**
 * Ranked retrieval result pairing one episodic memory record with its
 * prompt-token overlap score.
 *
 * Results are returned from `retrieveNeatChatMemories` sorted by
 * `overlapCount` descending, with ties broken by insertion time ascending
 * (oldest record first). The `record` field is a direct reference to the
 * stored bank record, and the top-level spread fields duplicate its
 * properties for convenient destructuring.
 */
export type NeatChatMemoryRetrievalResult = NeatChatMemoryRecord & {
  /** Retrieved memory record. */
  readonly record: NeatChatMemoryRecord;
  /** Number of overlapping prompt tokens matched by the record. */
  readonly overlapCount: number;
};