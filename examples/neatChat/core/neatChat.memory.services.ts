import { tokenizeNeatChatText } from './neatChat.tokenization.utils';
import type { NeatChatSession } from './neatChat.types';
import type {
  CreateNeatChatEpisodicMemoryBankOptions,
  NeatChatEpisodicMemoryBank,
  NeatChatMemoryRecord,
  NeatChatMemoryRetrievalResult,
  RetrieveNeatChatMemoriesOptions,
} from './neatChat.memory.types';

/**
 * Default maximum number of episodic memory records kept in one NEATchat
 * session's memory bank before older low-hit records are pruned.
 *
 * 100 records balances a useful working-set size against the memory pressure
 * of carrying episodic facts through a live exchange loop. Sessions with
 * narrow domain focus or strict memory budgets should reduce this cap;
 * sessions intended for long-running personalized interactions can safely
 * increase it.
 */
export const NEATCHAT_DEFAULT_MEMORY_BANK_MAX_RECORDS = 100;

/**
 * Creates an empty episodic memory bank for a live NEATchat session.
 *
 * @param options - Optional memory-bank cap overrides.
 * @returns Empty memory bank with the resolved record cap and no consolidation timestamp.
 *
 * @example
 * ```ts
 * import { createNeatChatEpisodicMemoryBank } from '../index';
 *
 * const bank = createNeatChatEpisodicMemoryBank();
 * console.log(bank.records.length); // 0
 * console.log(bank.maxRecords);     // 100
 *
 * const smallBank = createNeatChatEpisodicMemoryBank({ maxRecords: 20 });
 * console.log(smallBank.maxRecords); // 20
 * ```
 */
export function createNeatChatEpisodicMemoryBank(
  options: CreateNeatChatEpisodicMemoryBankOptions = {},
): NeatChatEpisodicMemoryBank {
  // Step 1: Resolve the effective record cap for the new bank.
  const maxRecords =
    options.maxRecords ?? NEATCHAT_DEFAULT_MEMORY_BANK_MAX_RECORDS;

  // Step 2: Return a fresh empty bank with no consolidation timestamp.
  return {
    records: [],
    maxRecords,
    lastConsolidatedAt: null,
  };
}

/**
 * Adds one episodic memory record to a bank without mutating the original bank.
 *
 * @param bank - Existing memory bank to extend.
 * @param record - Incoming record payload without runtime metadata.
 * @returns New memory bank with the appended record and pre-append pruning applied.
 *
 * @remarks
 * When the bank is already at `maxRecords` capacity, the record with the
 * lowest `hitCount` (oldest `addedAt` first on ties) is dropped before
 * appending the new record. The original bank is never mutated; a new bank
 * object is always returned.
 *
 * @example
 * ```ts
 * import {
 *   createNeatChatEpisodicMemoryBank,
 *   addNeatChatMemoryRecord,
 * } from '../index';
 *
 * const emptyBank = createNeatChatEpisodicMemoryBank();
 * const bank = addNeatChatMemoryRecord(emptyBank, {
 *   key: 'user name',
 *   value: 'Alice',
 * });
 * console.log(bank.records.length);      // 1
 * console.log(emptyBank.records.length); // 0 — original is unchanged
 * ```
 */
export function addNeatChatMemoryRecord(
  bank: NeatChatEpisodicMemoryBank,
  record: Omit<NeatChatMemoryRecord, 'addedAt' | 'hitCount'>,
): NeatChatEpisodicMemoryBank {
  // Step 1: Make room for one new record before appending.
  const bankWithAvailableSlot = createBankWithAvailableSlot(bank);

  // Step 2: Append the new runtime-stamped record immutably.
  return {
    ...bankWithAvailableSlot,
    records: [
      ...bankWithAvailableSlot.records,
      createStoredMemoryRecord(record),
    ],
  };

  /**
   * Creates a bank view with at least one available append slot.
   *
   * @param memoryBank - Existing bank that may already be at capacity.
   * @returns Bank with enough retained records to append one more item.
   */
  function createBankWithAvailableSlot(
    memoryBank: NeatChatEpisodicMemoryBank,
  ): NeatChatEpisodicMemoryBank {
    if (memoryBank.records.length < memoryBank.maxRecords) {
      return memoryBank;
    }

    return {
      ...memoryBank,
      records: selectRetainedRecords(
        memoryBank.records,
        Math.max(memoryBank.maxRecords - 1, 0),
      ),
    };
  }
}

/**
 * Retrieves episodic memories ranked by prompt-token overlap.
 *
 * Each stored record is scored by counting how many tokens from the prompt
 * appear in the combined key and value text of that record. Results are
 * returned sorted by overlap count descending; ties are broken by insertion
 * time ascending so older records surface first. The session is never mutated.
 *
 * @param session - Session exposing the live vocabulary and episodic memory bank.
 * @param promptText - Prompt text used for overlap ranking.
 * @param options - Optional retrieval limit.
 * @returns Ranked retrieval results ordered by overlap count and insertion time.
 *
 * @example
 * ```ts
 * import {
 *   createNeatChatSession,
 *   addNeatChatMemoryRecord,
 *   retrieveNeatChatMemories,
 * } from '../index';
 *
 * let session = createNeatChatSession({
 *   corpusRetainedTerms: ['user', 'name'],
 * });
 * session = {
 *   ...session,
 *   memoryBank: addNeatChatMemoryRecord(session.memoryBank, {
 *     key: 'user name',
 *     value: 'Alice',
 *   }),
 * };
 * const results = retrieveNeatChatMemories(session, 'user name', {
 *   maxResults: 3,
 * });
 * console.log(results[0]?.overlapCount); // >= 1
 * ```
 */
export function retrieveNeatChatMemories(
  session: Pick<NeatChatSession, 'memoryBank' | 'vocabulary'>,
  promptText: string,
  options: RetrieveNeatChatMemoriesOptions = {},
): NeatChatMemoryRetrievalResult[] {
  // Step 1: Exit early when there is nothing to rank.
  if (session.memoryBank.records.length === 0) {
    return [];
  }

  // Step 2: Build ranked overlap results for every stored record.
  const promptTokens = tokenizeNeatChatText(
    promptText,
    Number.MAX_SAFE_INTEGER,
  );
  const maxResults = options.maxResults ?? Number.MAX_SAFE_INTEGER;
  const rankedResults = session.memoryBank.records
    .map((memoryRecord) =>
      createRetrievalResult(
        memoryRecord,
        countTokenOverlap(
          promptTokens,
          tokenizeNeatChatText(
            `${memoryRecord.key} ${memoryRecord.value}`,
            Number.MAX_SAFE_INTEGER,
          ),
        ),
      ),
    )
    .toSorted(compareRetrievalResults)
    .slice(0, maxResults);

  // Step 3: Return a read-only ranking snapshot without mutating the session.
  return rankedResults;
}

/**
 * Trims a memory bank down to its configured record cap.
 *
 * When the bank already holds fewer records than `maxRecords`, the original
 * object is returned unchanged. Otherwise records are ranked by `hitCount`
 * ascending (oldest `addedAt` first on ties) and the lowest-ranked excess
 * records are removed, preserving the relative insertion order of retained
 * items.
 *
 * @param bank - Existing bank that may exceed its configured cap.
 * @returns Unchanged bank when already within the cap, otherwise a pruned copy.
 *
 * @example
 * ```ts
 * import {
 *   createNeatChatEpisodicMemoryBank,
 *   addNeatChatMemoryRecord,
 *   pruneNeatChatMemoryBank,
 * } from '../index';
 *
 * let bank = createNeatChatEpisodicMemoryBank({ maxRecords: 2 });
 * bank = addNeatChatMemoryRecord(bank, { key: 'a', value: 'first' });
 * bank = addNeatChatMemoryRecord(bank, { key: 'b', value: 'second' });
 *
 * // Lowering the cap on an existing bank and then pruning explicitly:
 * const shrunk = { ...bank, maxRecords: 1 };
 * const pruned = pruneNeatChatMemoryBank(shrunk);
 * console.log(pruned.records.length); // 1
 * ```
 */
export function pruneNeatChatMemoryBank(
  bank: NeatChatEpisodicMemoryBank,
): NeatChatEpisodicMemoryBank {
  // Step 1: Keep the original object when no pruning is needed.
  if (bank.records.length <= bank.maxRecords) {
    return bank;
  }

  // Step 2: Return a pruned bank that preserves record order for retained items.
  return {
    ...bank,
    records: selectRetainedRecords(bank.records, bank.maxRecords),
  };
}

function createStoredMemoryRecord(
  record: Omit<NeatChatMemoryRecord, 'addedAt' | 'hitCount'>,
): NeatChatMemoryRecord {
  return {
    key: record.key,
    value: record.value,
    addedAt: Date.now(),
    hitCount: 0,
  };
}

function countTokenOverlap(
  promptTokens: readonly string[],
  recordTokens: readonly string[],
): number {
  const recordTokenSet = new Set(recordTokens);

  return promptTokens.reduce(
    (overlapCount, promptToken) =>
      overlapCount + (recordTokenSet.has(promptToken) ? 1 : 0),
    0,
  );
}

function compareRetrievalResults(
  leftResult: NeatChatMemoryRetrievalResult,
  rightResult: NeatChatMemoryRetrievalResult,
): number {
  const overlapDelta = rightResult.overlapCount - leftResult.overlapCount;

  if (overlapDelta !== 0) {
    return overlapDelta;
  }

  return leftResult.record.addedAt - rightResult.record.addedAt;
}

function createRetrievalResult(
  memoryRecord: NeatChatMemoryRecord,
  overlapCount: number,
): NeatChatMemoryRetrievalResult {
  const retrievalResult = {
    ...memoryRecord,
    record: memoryRecord,
    overlapCount,
  };

  return retrievalResult;
}

function selectRetainedRecords(
  records: readonly NeatChatMemoryRecord[],
  retainedCount: number,
): NeatChatMemoryRecord[] {
  const removalCount = Math.max(records.length - retainedCount, 0);

  if (removalCount === 0) {
    return [...records];
  }

  const removalIndexSet = new Set(
    records
      .map((memoryRecord, recordIndex) => ({
        memoryRecord,
        recordIndex,
      }))
      .toSorted(compareRemovalCandidates)
      .slice(0, removalCount)
      .map(({ recordIndex }) => recordIndex),
  );

  return records.filter((_, recordIndex) => !removalIndexSet.has(recordIndex));
}

function compareRemovalCandidates(
  leftCandidate: {
    readonly memoryRecord: NeatChatMemoryRecord;
    readonly recordIndex: number;
  },
  rightCandidate: {
    readonly memoryRecord: NeatChatMemoryRecord;
    readonly recordIndex: number;
  },
): number {
  const removalPriorityDelta = compareRemovalPriority(
    leftCandidate.memoryRecord,
    rightCandidate.memoryRecord,
  );

  if (removalPriorityDelta !== 0) {
    return removalPriorityDelta;
  }

  return leftCandidate.recordIndex - rightCandidate.recordIndex;
}

function compareRemovalPriority(
  leftRecord: NeatChatMemoryRecord,
  rightRecord: NeatChatMemoryRecord,
): number {
  const hitCountDelta = leftRecord.hitCount - rightRecord.hitCount;

  if (hitCountDelta !== 0) {
    return hitCountDelta;
  }

  return leftRecord.addedAt - rightRecord.addedAt;
}