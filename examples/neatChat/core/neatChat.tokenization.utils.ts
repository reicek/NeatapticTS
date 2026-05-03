import {
  NEATCHAT_DEFAULT_CHUNK_TOKEN_COUNT,
  NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
  NEATCHAT_DEFAULT_TOP_WORD_LIMIT,
  NEATCHAT_SPECIAL_TOKENS,
} from './neatChat.constants';
import { NeatChatPositiveIntegerValidationError } from './neatChat.errors';
import type {
  CreateNeatChatPretrainingPreviewOptions,
  EstimateNeatChatRuntimeOptions,
  NeatChatCorpusReport,
  NeatChatPretrainingPreview,
  NeatChatRuntimeDurationBucket,
  NeatChatRuntimeEstimate,
} from './neatChat.types';

/**
 * Builds a lightweight estimate for the bounded pretraining budget.
 *
 * The estimate is intentionally simple. It exists to make the published
 * browser page and the Node preview agree on the same vocabulary cap, short
 * context window, and expected relative runtime bucket before any real corpus
 * is ingested.
 *
 * @param options - Optional overrides for the top-word cap or token window.
 * @returns Lightweight runtime estimate for the chosen budget.
 *
 * @example
 * ```ts
 * import { estimateNeatChatRuntime } from '../index';
 *
 * const estimate = estimateNeatChatRuntime({ topWordLimit: 1200 });
 * console.log(estimate.expectedPretrainingDurationBucket); // short
 * ```
 */
export function estimateNeatChatRuntime(
  options: EstimateNeatChatRuntimeOptions = {},
): NeatChatRuntimeEstimate {
  const topWordLimit = resolvePositiveInteger(
    options.topWordLimit ?? NEATCHAT_DEFAULT_TOP_WORD_LIMIT,
    'topWordLimit',
  );
  const contextWindowTokenCount = resolvePositiveInteger(
    options.contextWindowTokenCount ??
      NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
    'contextWindowTokenCount',
  );
  const estimatedRetainedVocabularySize =
    topWordLimit + NEATCHAT_SPECIAL_TOKENS.length;
  const expectedPretrainingDurationBucket = resolveRuntimeDurationBucket(
    topWordLimit,
    contextWindowTokenCount,
  );

  return {
    topWordLimit,
    specialTokenCount: NEATCHAT_SPECIAL_TOKENS.length,
    estimatedRetainedVocabularySize,
    contextWindowTokenCount,
    expectedPretrainingDurationBucket,
    summary: `Retain about ${estimatedRetainedVocabularySize} tokens including ${NEATCHAT_SPECIAL_TOKENS.join(', ')}, keep each prompt or reply slice to ${contextWindowTokenCount} tokens, and expect a ${expectedPretrainingDurationBucket} bounded pretraining pass at this scale.`,
  };
}

/**
 * Tokenizes one prompt or reply slice for the NEATchat example.
 *
 * The tokenizer stays intentionally language-agnostic and cheap. It keeps
 * Unicode letter and number runs plus apostrophes, lowercases the result, and
 * truncates to the short context window used by the Step 2 contract.
 *
 * @param text - Prompt or reply text to tokenize.
 * @param maxTokenCount - Maximum number of tokens to keep.
 * @returns Normalized token list truncated to the requested window.
 */
export function tokenizeNeatChatText(
  text: string,
  maxTokenCount = NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
): string[] {
  const resolvedMaxTokenCount = resolvePositiveInteger(
    maxTokenCount,
    'maxTokenCount',
  );

  return [...iterateNormalizedNeatChatTokens(text)].slice(
    0,
    resolvedMaxTokenCount,
  );
}

/**
 * Extracts conversation lines from corpus text for optional session seeding.
 *
 * The parser accepts either:
 * - a JSON string array (`["line 1", "line 2"]`), or
 * - plain text with one interaction per line.
 *
 * Empty lines are removed and remaining lines are trimmed.
 *
 * @param corpusText - Raw corpus text from the browser textarea.
 * @returns Ordered non-empty conversation lines.
 */
export function extractNeatChatConversationLines(corpusText: string): string[] {
  const normalizedCorpusText = corpusText.trim();

  if (normalizedCorpusText.length === 0) {
    return [];
  }

  const strippedFenceText = stripJsonCodeFence(normalizedCorpusText);
  const jsonParsedLines = parseConversationJsonArray(strippedFenceText);

  if (jsonParsedLines !== null) {
    return jsonParsedLines;
  }

  return strippedFenceText
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

/**
 * Builds a bounded optional pretraining preview from pasted corpus text.
 *
 * The preview does not perform real network training yet. It prepares the same
 * retained-vocabulary slice that a later training pass can reuse by counting
 * normalized terms in bounded chunks and keeping only the top-frequency terms.
 *
 * @param options - Corpus text plus bounded retention options.
 * @returns Shared pretraining preview used by browser and Node surfaces.
 *
 * @example
 * ```ts
 * import { createNeatChatPretrainingPreview } from '../index';
 *
 * const preview = createNeatChatPretrainingPreview({
 *   corpusText: 'star sun star moon star sun',
 *   topWordLimit: 2,
 * });
 * console.log(preview.previewTerms); // ['star', 'sun']
 * ```
 */
export function createNeatChatPretrainingPreview(
  options: CreateNeatChatPretrainingPreviewOptions,
): NeatChatPretrainingPreview {
  const characterCount = options.corpusText.length;
  const topWordLimit = resolvePositiveInteger(
    options.topWordLimit ?? NEATCHAT_DEFAULT_TOP_WORD_LIMIT,
    'topWordLimit',
  );
  const chunkTokenCount = resolvePositiveInteger(
    options.chunkTokenCount ?? NEATCHAT_DEFAULT_CHUNK_TOKEN_COUNT,
    'chunkTokenCount',
  );
  const previewTermCount = resolvePositiveInteger(
    options.previewTermCount ?? 8,
    'previewTermCount',
  );
  const termCounts = new Map<string, number>();
  const pendingChunkTokens: string[] = [];
  let totalTokenCount = 0;
  let processedChunkCount = 0;

  // Step 1: Count normalized terms in bounded chunks so large pasted corpora
  // stay on a stable memory profile while the frequency table is built.
  for (const normalizedToken of iterateNormalizedNeatChatTokens(
    options.corpusText,
  )) {
    pendingChunkTokens.push(normalizedToken);
    totalTokenCount += 1;

    if (pendingChunkTokens.length >= chunkTokenCount) {
      flushPendingChunk();
    }
  }

  if (pendingChunkTokens.length > 0) {
    flushPendingChunk();
  }

  // Step 2: Keep only the top-frequency retained terms for the preview.
  const retainedEntries = [...termCounts.entries()]
    .toSorted((leftEntry, rightEntry) => {
      const countDelta = rightEntry[1] - leftEntry[1];

      if (countDelta !== 0) {
        return countDelta;
      }

      return leftEntry[0].localeCompare(rightEntry[0]);
    })
    .slice(0, topWordLimit);
  const retainedTokenCount = retainedEntries.reduce(
    (total, [, count]) => total + count,
    0,
  );
  const retainedTokenCoveragePercent =
    totalTokenCount === 0
      ? 0
      : Number(((retainedTokenCount / totalTokenCount) * 100).toFixed(2));
  const corpusReport = {
    characterCount,
    tokenCount: totalTokenCount,
    uniqueTermCount: termCounts.size,
    retainedTermCount: retainedEntries.length,
    retainedTokenCoveragePercent,
  } satisfies NeatChatCorpusReport;

  // Step 3: Return the compact preview summary used by browser and tests.
  return {
    hasCorpus: totalTokenCount > 0,
    characterCount,
    topWordLimit,
    chunkTokenCount,
    totalTokenCount,
    uniqueTermCount: termCounts.size,
    retainedTermCount: retainedEntries.length,
    retainedTokenCount,
    retainedTokenCoveragePercent,
    corpusReport,
    processedChunkCount,
    previewTerms: retainedEntries
      .slice(0, previewTermCount)
      .map(([term]) => term),
    retainedTerms: retainedEntries.map(([term]) => term),
  };

  function flushPendingChunk(): void {
    processedChunkCount += 1;

    for (const chunkToken of pendingChunkTokens) {
      termCounts.set(chunkToken, (termCounts.get(chunkToken) ?? 0) + 1);
    }

    pendingChunkTokens.length = 0;
  }
}

/**
 * Resolves a positive integer option value and throws for invalid input.
 *
 * This helper is shared across the NEATchat tokenization, session, and
 * snapshot chapters so all positive-integer validation failures surface one
 * consistent typed error family.
 *
 * @param value - Candidate numeric value.
 * @param optionName - Name used in the thrown error.
 * @returns Validated positive integer.
 * @throws {NeatChatPositiveIntegerValidationError} When the value is not a positive integer.
 *
 * @example
 * ```ts
 * const chunkTokenCount = resolvePositiveInteger(256, 'chunkTokenCount');
 * ```
 */
export function resolvePositiveInteger(
  value: number,
  optionName: string,
): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new NeatChatPositiveIntegerValidationError(
      `${optionName} must be a positive integer.`,
    );
  }

  return value;
}

function resolveRuntimeDurationBucket(
  topWordLimit: number,
  contextWindowTokenCount: number,
): NeatChatRuntimeDurationBucket {
  if (topWordLimit <= 750 && contextWindowTokenCount <= 16) {
    return 'very-short';
  }

  if (topWordLimit <= 1_500 && contextWindowTokenCount <= 24) {
    return 'short';
  }

  if (topWordLimit <= 3_000 && contextWindowTokenCount <= 32) {
    return 'moderate';
  }

  return 'heavy';
}

function* iterateNormalizedNeatChatTokens(text: string): Generator<string> {
  for (const tokenMatch of text.toLowerCase().matchAll(/[\p{L}\p{N}'’-]+/gu)) {
    const normalizedToken = tokenMatch[0];

    if (normalizedToken.length > 0) {
      yield normalizedToken;
    }
  }
}

function stripJsonCodeFence(value: string): string {
  const fenceMatch = value.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/u);

  if (!fenceMatch) {
    return value;
  }

  return fenceMatch[1] ?? value;
}

function parseConversationJsonArray(value: string): string[] | null {
  if (!value.startsWith('[')) {
    return null;
  }

  try {
    const parsedValue: unknown = JSON.parse(value);

    if (!Array.isArray(parsedValue)) {
      return null;
    }

    return parsedValue
      .filter((line): line is string => typeof line === 'string')
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
  } catch {
    return null;
  }
}
