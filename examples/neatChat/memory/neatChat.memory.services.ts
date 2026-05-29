import { tokenizeNeatChatText } from '../core/neatChat.tokenization.utils';
import {
  DEFAULT_MEMORY_RESULT_LIMIT,
  MAX_DURABLE_ENTRIES,
} from './neatChat.memory.constants';
import { retrieveStoredMemoryResults } from './neatChat.retrieval';
import type {
  MemoryAdapter,
  MemoryResult,
  StoredMemoryEntry,
} from './neatChat.memory.types';

type ListableMemoryEntry = Pick<
  StoredMemoryEntry,
  'entryId' | 'sessionId' | 'content' | 'tokens' | 'createdAt' | 'lastUsed'
> &
  Partial<Pick<StoredMemoryEntry, 'entryType' | 'score'>>;

/** Options for storing one completed exchange in the durable memory layer. */
export interface StoreExchangeMemoryOptions {
  /** Adapter that persists the durable entry. */
  readonly adapter: Pick<MemoryAdapter, 'store'>;
  /** Session owner for the stored exchange. */
  readonly sessionId: string;
  /** User message text captured from the completed exchange. */
  readonly userMessage: string;
  /** Assistant response text captured from the completed exchange. */
  readonly response: string;
  /** Optional timestamp override used by deterministic tests. */
  readonly now?: number;
}

/** Options for retrieving ranked durable memory context. */
export interface RetrieveMemoryContextOptions {
  /** Adapter that supplies durable entries or a native ranked search path. */
  readonly adapter: {
    list(sessionId: string): Promise<readonly ListableMemoryEntry[]>;
    search?: Pick<MemoryAdapter, 'search'>['search'];
  };
  /** Session owner whose durable entries should be retrieved. */
  readonly sessionId: string;
  /** Raw query text used to rank durable entries. */
  readonly query: string;
  /** Maximum number of ranked results to return. */
  readonly maxResults?: number;
}

/** Options for pruning the durable memory layer back to its cap. */
export interface PruneMemoryOptions {
  /** Adapter that lists and removes durable entries. */
  readonly adapter: {
    list(sessionId: string): Promise<readonly ListableMemoryEntry[]>;
    remove(entryIds: readonly string[]): Promise<number>;
  };
  /** Session owner whose durable entries should be pruned. */
  readonly sessionId: string;
  /** Maximum retained durable entries after pruning completes. */
  readonly maxEntries?: number;
}

/** Options for exporting a session-scoped durable memory snapshot. */
export interface ExportMemoryOptions {
  /** Adapter that lists durable entries for export. */
  readonly adapter: {
    list(sessionId: string): Promise<readonly ListableMemoryEntry[]>;
  };
  /** Session owner whose durable entries should be exported. */
  readonly sessionId: string;
}

/**
 * Stores one completed exchange as a durable memory entry.
 *
 * @param options - Adapter and exchange details to persist.
 * @returns Generated durable entry identifier.
 */
export async function storeExchangeMemory(
  options: StoreExchangeMemoryOptions,
): Promise<string> {
  const storedAt = options.now ?? Date.now();
  const content = `${options.userMessage} ${options.response}`.trim();

  return options.adapter.store({
    sessionId: options.sessionId,
    entryType: 'exchange',
    content,
    tokens: tokenizeNeatChatText(content, Number.MAX_SAFE_INTEGER),
    score: 1,
    createdAt: storedAt,
    lastUsed: storedAt,
  });
}

/**
 * Retrieves ranked durable memory context for one session-scoped query.
 *
 * @param options - Adapter and query inputs for retrieval.
 * @returns Ranked durable memory results.
 */
export async function retrieveMemoryContext(
  options: RetrieveMemoryContextOptions,
): Promise<readonly MemoryResult[]> {
  return retrieveStoredMemoryResults({
    adapter: options.adapter,
    sessionId: options.sessionId,
    query: options.query,
    maxResults: options.maxResults ?? DEFAULT_MEMORY_RESULT_LIMIT,
  });
}

/**
 * Prunes a session-scoped durable corpus back to its configured cap.
 *
 * @param options - Adapter and session inputs for pruning.
 * @returns Number of removed durable entries.
 */
export async function pruneMemory(
  options: PruneMemoryOptions,
): Promise<number> {
  const maxEntries = options.maxEntries ?? MAX_DURABLE_ENTRIES;
  const sessionEntries = await options.adapter.list(options.sessionId);

  if (sessionEntries.length <= maxEntries) {
    return 0;
  }

  const removableEntryIds = sessionEntries
    .toSorted((leftEntry, rightEntry) => {
      if (leftEntry.lastUsed !== rightEntry.lastUsed) {
        return leftEntry.lastUsed - rightEntry.lastUsed;
      }

      if (leftEntry.createdAt !== rightEntry.createdAt) {
        return leftEntry.createdAt - rightEntry.createdAt;
      }

      return leftEntry.entryId.localeCompare(rightEntry.entryId);
    })
    .slice(0, sessionEntries.length - maxEntries)
    .map((memoryEntry) => memoryEntry.entryId);

  return options.adapter.remove(removableEntryIds);
}

/**
 * Exports the durable entries currently stored for one session.
 *
 * @param options - Adapter and session inputs for export.
 * @returns Stable created-at ordered durable entry snapshot.
 */
export async function exportMemory(
  options: ExportMemoryOptions,
): Promise<readonly StoredMemoryEntry[]> {
  const sessionEntries = (await options.adapter.list(options.sessionId)).map(
    normalizeStoredMemoryEntry,
  );

  return sessionEntries.toSorted((leftEntry, rightEntry) => {
    if (leftEntry.createdAt !== rightEntry.createdAt) {
      return leftEntry.createdAt - rightEntry.createdAt;
    }

    return leftEntry.entryId.localeCompare(rightEntry.entryId);
  });
}

function normalizeStoredMemoryEntry(
  memoryEntry: ListableMemoryEntry,
): StoredMemoryEntry {
  return {
    ...memoryEntry,
    entryType: memoryEntry.entryType ?? 'exchange',
    score: memoryEntry.score ?? 1,
  };
}
