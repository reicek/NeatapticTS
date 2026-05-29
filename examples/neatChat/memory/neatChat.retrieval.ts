import { tokenizeNeatChatText } from '../core/neatChat.tokenization.utils';
import {
  DEFAULT_MEMORY_RESULT_LIMIT,
  MEMORY_BM25_B,
  MEMORY_BM25_K1,
} from './neatChat.memory.constants';
import { sanitizeFtsQuery } from './neatChat.memory.fts';
import type {
  MemoryAdapter,
  MemoryResult,
  StoredMemoryEntry,
} from './neatChat.memory.types';

type RankableStoredMemoryEntry = Pick<
  StoredMemoryEntry,
  'entryId' | 'sessionId' | 'content' | 'tokens' | 'createdAt' | 'lastUsed'
> &
  Partial<Pick<StoredMemoryEntry, 'entryType' | 'score'>>;

/** Retrieval options for ranking durable memory entries. */
export interface RetrieveStoredMemoryResultsOptions {
  /** Adapter supplying durable entries or native ranked search. */
  readonly adapter: {
    list(sessionId: string): Promise<readonly RankableStoredMemoryEntry[]>;
    search?: Pick<MemoryAdapter, 'search'>['search'];
  };
  /** Session owner whose durable entries should be retrieved. */
  readonly sessionId: string;
  /** Raw user query used to build ranking terms. */
  readonly query: string;
  /** Maximum number of ranked results to return. */
  readonly maxResults?: number;
}

/** Local ranking options for a session-scoped durable-memory corpus. */
export interface RankMemoryEntriesOptions {
  /** Session-scoped durable entries to rank. */
  readonly entries: readonly RankableStoredMemoryEntry[];
  /** Raw or sanitized query text used to build ranking terms. */
  readonly query: string;
  /** Maximum number of ranked results to return. */
  readonly maxResults?: number;
}

/**
 * Retrieves ranked durable memories from an adapter.
 *
 * The function prefers an adapter-native search path when present and falls
 * back to the local token-overlap plus BM25 scorer for lightweight adapters.
 *
 * @param options - Adapter, session, and query inputs for retrieval.
 * @returns Ranked durable memory results.
 */
export async function retrieveStoredMemoryResults(
  options: RetrieveStoredMemoryResultsOptions,
): Promise<readonly MemoryResult[]> {
  const maxResults = options.maxResults ?? DEFAULT_MEMORY_RESULT_LIMIT;
  const sanitizedQuery = sanitizeFtsQuery(options.query);

  if (sanitizedQuery.length === 0) {
    return [];
  }

  if (options.adapter.search) {
    return options.adapter.search({
      sessionId: options.sessionId,
      query: sanitizedQuery,
      maxResults,
    });
  }

  const sessionEntries = await options.adapter.list(options.sessionId);

  return rankMemoryEntries({
    entries: sessionEntries,
    query: sanitizedQuery,
    maxResults,
  });
}

/**
 * Ranks a session-scoped durable-memory corpus with token overlap and BM25.
 *
 * @param options - Corpus and query inputs for local ranking.
 * @returns Ranked durable memory results ordered by descending relevance.
 */
export function rankMemoryEntries(
  options: RankMemoryEntriesOptions,
): readonly MemoryResult[] {
  const queryTokens = tokenizeNeatChatText(
    sanitizeFtsQuery(options.query),
    Number.MAX_SAFE_INTEGER,
  );

  if (queryTokens.length === 0 || options.entries.length === 0) {
    return [];
  }

  const distinctQueryTokens = [...new Set(queryTokens)];
  const averageDocumentLength =
    options.entries.reduce(
      (totalLength, memoryEntry) =>
        totalLength + resolveEntryTokens(memoryEntry).length,
      0,
    ) / options.entries.length;
  const documentFrequencyByToken = new Map(
    distinctQueryTokens.map((queryToken) => [
      queryToken,
      countDocumentFrequency(options.entries, queryToken),
    ]),
  );

  return options.entries
    .map((memoryEntry) =>
      createRankedResult(
        memoryEntry,
        distinctQueryTokens,
        documentFrequencyByToken,
        averageDocumentLength,
        options.entries.length,
      ),
    )
    .toSorted(compareMemoryResults)
    .slice(0, options.maxResults ?? DEFAULT_MEMORY_RESULT_LIMIT);
}

function createRankedResult(
  memoryEntry: RankableStoredMemoryEntry,
  distinctQueryTokens: readonly string[],
  documentFrequencyByToken: ReadonlyMap<string, number>,
  averageDocumentLength: number,
  documentCount: number,
): MemoryResult {
  const entryTokens = resolveEntryTokens(memoryEntry);
  const overlapScore = distinctQueryTokens.reduce(
    (matchedTokenCount, queryToken) =>
      matchedTokenCount + (entryTokens.includes(queryToken) ? 1 : 0),
    0,
  );
  const bm25Score = distinctQueryTokens.reduce(
    (totalScore, queryToken) =>
      totalScore +
      calculateBm25Contribution({
        entryTokens,
        queryToken,
        documentFrequency: documentFrequencyByToken.get(queryToken) ?? 0,
        documentCount,
        averageDocumentLength,
      }),
    0,
  );
  const reinforcementScore = Math.max(memoryEntry.score ?? 1, 1);
  const relevanceScore = overlapScore + bm25Score * reinforcementScore;

  return {
    ...memoryEntry,
    entryType: memoryEntry.entryType ?? 'exchange',
    score: memoryEntry.score ?? 1,
    overlapScore,
    bm25Score: Number(bm25Score.toFixed(6)),
    relevanceScore: Number(relevanceScore.toFixed(6)),
  };
}

function compareMemoryResults(
  leftResult: MemoryResult,
  rightResult: MemoryResult,
): number {
  if (rightResult.relevanceScore !== leftResult.relevanceScore) {
    return rightResult.relevanceScore - leftResult.relevanceScore;
  }

  if (rightResult.overlapScore !== leftResult.overlapScore) {
    return rightResult.overlapScore - leftResult.overlapScore;
  }

  if (rightResult.lastUsed !== leftResult.lastUsed) {
    return rightResult.lastUsed - leftResult.lastUsed;
  }

  if (rightResult.createdAt !== leftResult.createdAt) {
    return rightResult.createdAt - leftResult.createdAt;
  }

  return leftResult.entryId.localeCompare(rightResult.entryId);
}

function countDocumentFrequency(
  entries: readonly RankableStoredMemoryEntry[],
  queryToken: string,
): number {
  return entries.reduce((matchingDocumentCount, memoryEntry) => {
    const entryTokenSet = new Set(resolveEntryTokens(memoryEntry));

    return matchingDocumentCount + (entryTokenSet.has(queryToken) ? 1 : 0);
  }, 0);
}

function calculateBm25Contribution(options: {
  readonly entryTokens: readonly string[];
  readonly queryToken: string;
  readonly documentFrequency: number;
  readonly documentCount: number;
  readonly averageDocumentLength: number;
}): number {
  const termFrequency = options.entryTokens.filter(
    (entryToken) => entryToken === options.queryToken,
  ).length;

  if (termFrequency === 0) {
    return 0;
  }

  const inverseDocumentFrequency = Math.log(
    1 +
      (options.documentCount - options.documentFrequency + 0.5) /
        (options.documentFrequency + 0.5),
  );
  const normalizedDocumentLength =
    options.averageDocumentLength === 0
      ? 1
      : options.entryTokens.length / options.averageDocumentLength;
  const denominator =
    termFrequency +
    MEMORY_BM25_K1 *
      (1 - MEMORY_BM25_B + MEMORY_BM25_B * normalizedDocumentLength);

  return (
    inverseDocumentFrequency *
    ((termFrequency * (MEMORY_BM25_K1 + 1)) / denominator)
  );
}

function resolveEntryTokens(
  memoryEntry: RankableStoredMemoryEntry,
): readonly string[] {
  if (memoryEntry.tokens.length > 0) {
    return memoryEntry.tokens;
  }

  return tokenizeNeatChatText(memoryEntry.content, Number.MAX_SAFE_INTEGER);
}
