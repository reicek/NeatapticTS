/**
 * @module assemble-context
 * @description Five-stage context-window assembly pipeline for Cortex search results.
 * Stateless, composable pure functions: enrich → deduplicate → order → budget → stitch.
 *
 * @example
 * ```js
 * import { assembleContext } from './assemble-context.mjs';
 *
 * const result = await assembleContext(chunks, { budget: 4096, context_format: 'markdown' });
 * console.log(result.context, result.tokenCount, result.tierCounts);
 * ```
 */

import { createHash } from 'node:crypto';
import { computeCosineSimilarity } from './hybrid-rank.mjs';

/**
 * Approximate characters per token for the default rough tokenizer.
 * Conservative 4:1 ratio keeps the context slightly under-filled.
 * @type {number}
 */
export const DEFAULT_CHARS_PER_TOKEN = 4;

/**
 * Default token budget for an unclassified query.
 * @type {number}
 */
export const DEFAULT_BUDGET = 4096;

/**
 * Default cosine similarity threshold for near-duplicate collapse.
 * @type {number}
 */
export const DEFAULT_COSINE_THRESHOLD = 0.95;

/**
 * Default relevance tier thresholds.
 * @type {{ essential: number, supporting: number }}
 */
export const DEFAULT_TIER_THRESHOLDS = { essential: 0.7, supporting: 0.4 };

/**
 * Known document families ordered from most to least authoritative.
 * Used as the default family priority tiebreaker.
 * @type {string[]}
 */
export const DEFAULT_FAMILY_PRIORITY = [
  'ts-source',
  'readme',
  'plan',
  'agent',
  'skill',
  'demo',
];

/**
 * @typedef {Object} InputChunk
 * @property {number} chunk_id
 * @property {string} file_path
 * @property {string} [heading_path]
 * @property {number} char_start
 * @property {number} char_end
 * @property {string} [body_text]
 * @property {string} [content]
 * @property {number} score
 * @property {string} [family]
 * @property {string} [doc_family]
 * @property {number} [depth]
 * @property {number|null} [parent_chunk_id]
 * @property {number[]|null} [embedding]
 * @property {string} [context_header]
 * @property {string} [sha256]
 * @property {string} [tier]
 */

/**
 * @typedef {Object} EnrichedChunk
 * @property {string} sha256
 * @property {string} context_header
 * @property {string} tier
 * @property {string} body_text
 */

/**
 * @typedef {InputChunk & EnrichedChunk} AssembledChunk
 */

/**
 * @typedef {Object} BudgetResult
 * @property {AssembledChunk[]} selectedChunks
 * @property {number} tokenCount
 * @property {number} budget
 * @property {boolean} truncated
 * @property {Object.<string, number>} tierCounts
 */

/**
 * @typedef {Object} StitchJsonResult
 * @property {string} context
 * @property {Object[]} chunks
 * @property {number} tokenCount
 */

/**
 * Compute the SHA-256 hex digest of a UTF-8 string.
 *
 * @param {string} text
 * @returns {string}
 */
function sha256Hex(text) {
  return createHash('sha256').update(text).digest('hex');
}

/**
 * Extract the textual body from a chunk, preferring `body_text` and falling back
 * to `content` so the pipeline accepts both naming conventions.
 *
 * @param {InputChunk} chunk
 * @returns {string}
 */
function getBodyText(chunk) {
  return String(chunk.body_text ?? chunk.content ?? '');
}

/**
 * Build a provenance context header for a chunk.
 *
 * @param {InputChunk} chunk
 * @returns {string}
 */
function buildContextHeader(chunk) {
  if (chunk.context_header) return chunk.context_header;
  const heading = chunk.heading_path?.trim();
  return heading ? `${chunk.file_path} > ${heading}` : chunk.file_path;
}

/**
 * Estimate the token cost of a text fragment using the configured chars-per-token
 * ratio. Empty text costs zero tokens.
 *
 * @param {string} text
 * @param {number} [charsPerToken]
 * @returns {number}
 */
function estimateTokenCount(text, charsPerToken = DEFAULT_CHARS_PER_TOKEN) {
  if (text.length === 0) return 0;
  return Math.ceil(text.length / Math.max(1, charsPerToken));
}

/**
 * Assign a relevance tier based on a chunk's score and configurable thresholds.
 *
 * @param {number} score
 * @param {{ essential?: number, supporting?: number }} [thresholds]
 * @returns {'essential'|'supporting'|'supplementary'}
 */
function assignTier(score, thresholds = {}) {
  const essentialThreshold =
    thresholds.essential ?? DEFAULT_TIER_THRESHOLDS.essential;
  const supportingThreshold =
    thresholds.supporting ?? DEFAULT_TIER_THRESHOLDS.supporting;
  if (score >= essentialThreshold) return 'essential';
  if (score >= supportingThreshold) return 'supporting';
  return 'supplementary';
}

/**
 * Stage 1 — Enrich every chunk with a content hash and a context header.
 *
 * @param {InputChunk[]} chunks
 * @param {{ query_class?: string }} [options]
 * @returns {Promise<AssembledChunk[]>}
 */
export async function enrichChunks(chunks, options = {}) {
  const queryClass = options.query_class ?? 'default';
  return chunks.map((chunk) => {
    const bodyText = getBodyText(chunk);
    const sha256 = chunk.sha256 ?? sha256Hex(bodyText);
    const contextHeader = chunk.context_header ?? buildContextHeader(chunk);
    return {
      ...chunk,
      body_text: bodyText,
      sha256,
      context_header: contextHeader,
      // Preserve an explicit query_class so later stages can make class-specific
      // decisions without mutating the input object.
      query_class: queryClass,
    };
  });
}

/**
 * Resolve a near-duplicate pair to a single representative.
 *
 * @param {AssembledChunk} leftChunk
 * @param {AssembledChunk} rightChunk
 * @returns {AssembledChunk}
 */
function selectNearDuplicateWinner(leftChunk, rightChunk) {
  const scoreDelta = rightChunk.score - leftChunk.score;
  if (scoreDelta > 0) return rightChunk;
  if (scoreDelta < 0) return leftChunk;
  const leftLength = getBodyText(leftChunk).length;
  const rightLength = getBodyText(rightChunk).length;
  if (leftLength !== rightLength) {
    return leftLength <= rightLength ? leftChunk : rightChunk;
  }
  return leftChunk.chunk_id <= rightChunk.chunk_id ? leftChunk : rightChunk;
}

/**
 * Collapse a group of identical-content chunks into near-duplicate survivors.
 *
 * @param {AssembledChunk[]} group
 * @param {number} cosineThreshold
 * @returns {AssembledChunk[]}
 */
function collapseNearDuplicates(group, cosineThreshold) {
  const removedIds = new Set();
  for (let leftIndex = 0; leftIndex < group.length; leftIndex += 1) {
    const leftChunk = group[leftIndex];
    if (removedIds.has(leftChunk.chunk_id)) continue;
    for (
      let rightIndex = leftIndex + 1;
      rightIndex < group.length;
      rightIndex += 1
    ) {
      const rightChunk = group[rightIndex];
      if (removedIds.has(rightChunk.chunk_id)) continue;
      const similarity = computeCosineSimilarity(
        leftChunk.embedding,
        rightChunk.embedding,
      );
      if (similarity >= cosineThreshold) {
        const winner = selectNearDuplicateWinner(leftChunk, rightChunk);
        const loser =
          winner.chunk_id === leftChunk.chunk_id ? rightChunk : leftChunk;
        removedIds.add(loser.chunk_id);
      }
    }
  }
  return group.filter((chunk) => !removedIds.has(chunk.chunk_id));
}

/**
 * Collapse parent chunks in favor of their children when the child is at least
 * as relevant as the parent.
 *
 * @param {AssembledChunk[]} chunks
 * @returns {AssembledChunk[]}
 */
function collapseParentChild(chunks) {
  const idToChunk = new Map(chunks.map((chunk) => [chunk.chunk_id, chunk]));
  const parentToChildren = new Map();
  for (const chunk of chunks) {
    if (chunk.parent_chunk_id == null) continue;
    const children = parentToChildren.get(chunk.parent_chunk_id) ?? [];
    children.push(chunk);
    parentToChildren.set(chunk.parent_chunk_id, children);
  }

  const removedIds = new Set();
  for (const [parentId, children] of parentToChildren.entries()) {
    const parent = idToChunk.get(parentId);
    if (!parent) continue;
    const maxChildScore = Math.max(...children.map((child) => child.score));
    if (maxChildScore >= parent.score) {
      removedIds.add(parentId);
    } else {
      for (const child of children) {
        removedIds.add(child.chunk_id);
      }
    }
  }

  return chunks.filter((chunk) => !removedIds.has(chunk.chunk_id));
}

/**
 * Stage 2 — Deduplicate chunks by exact hash, near-duplicate embedding cosine,
 * and parent-child collapse.
 *
 * @param {AssembledChunk[]} chunks
 * @param {{ cosineThreshold?: number }} [options]
 * @returns {AssembledChunk[]}
 */
export function deduplicateChunks(chunks, options = {}) {
  const cosineThreshold = options.cosineThreshold ?? DEFAULT_COSINE_THRESHOLD;

  // Group chunks by exact content hash.
  const hashGroups = new Map();
  for (const chunk of chunks) {
    const key = chunk.sha256 ?? sha256Hex(getBodyText(chunk));
    const group = hashGroups.get(key) ?? [];
    group.push(chunk);
    hashGroups.set(key, group);
  }
  const groups = Array.from(hashGroups.values());
  const singleExactFamily = groups.length === 1;

  const afterDeduplication = [];
  for (const group of groups) {
    if (group.length <= 1) {
      afterDeduplication.push(...group);
      continue;
    }

    const embeddingsAvailable = group.every(
      (chunk) => Array.isArray(chunk.embedding) && chunk.embedding.length > 0,
    );

    if (embeddingsAvailable) {
      // Warm embeddings: use cosine near-duplicate collapse within the family.
      afterDeduplication.push(
        ...collapseNearDuplicates(group, cosineThreshold),
      );
      continue;
    }

    if (singleExactFamily) {
      // Cold embeddings and the whole result set is one exact-duplicate family:
      // preserve every member because we have no signal to choose a winner.
      afterDeduplication.push(...group);
      continue;
    }

    // Multiple exact families with cold embeddings: keep the highest-scored
    // representative from this duplicate family.
    const representative = group
      .toSorted((leftChunk, rightChunk) => {
        const scoreDelta = rightChunk.score - leftChunk.score;
        if (scoreDelta !== 0) return scoreDelta;
        return leftChunk.chunk_id - rightChunk.chunk_id;
      })
      .at(0);
    if (representative) afterDeduplication.push(representative);
  }

  return collapseParentChild(afterDeduplication);
}

/**
 * Determine the ordinal priority of a chunk's family.
 *
 * @param {AssembledChunk} chunk
 * @param {string[]} [priorityList]
 * @returns {number}
 */
function familyPriorityOrdinal(chunk, priorityList = DEFAULT_FAMILY_PRIORITY) {
  const family = chunk.family ?? chunk.doc_family;
  if (!family) return Number.MAX_SAFE_INTEGER;
  const index = priorityList.indexOf(family);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

/**
 * Stage 3 — Order chunks by relevance tier, file grouping, character position,
 * and family priority.
 *
 * @param {AssembledChunk[]} chunks
 * @param {{
 *   essential?: number,
 *   supporting?: number,
 *   familyPriority?: string[]
 * }} [options]
 * @returns {AssembledChunk[]}
 */
export function orderChunks(chunks, options = {}) {
  const thresholds = {
    essential: options.essential ?? DEFAULT_TIER_THRESHOLDS.essential,
    supporting: options.supporting ?? DEFAULT_TIER_THRESHOLDS.supporting,
  };
  const priorityList = options.familyPriority ?? DEFAULT_FAMILY_PRIORITY;

  const tiered = chunks.map((chunk) => ({
    ...chunk,
    tier: assignTier(chunk.score, thresholds),
  }));

  const fileMaxScores = new Map();
  for (const chunk of tiered) {
    const current = fileMaxScores.get(chunk.file_path) ?? -Infinity;
    if (chunk.score > current) {
      fileMaxScores.set(chunk.file_path, chunk.score);
    }
  }

  return tiered.toSorted((leftChunk, rightChunk) => {
    const leftTier = assignTier(leftChunk.score, thresholds);
    const rightTier = assignTier(rightChunk.score, thresholds);
    const tierDelta =
      ['essential', 'supporting', 'supplementary'].indexOf(leftTier) -
      ['essential', 'supporting', 'supplementary'].indexOf(rightTier);
    if (tierDelta !== 0) return tierDelta;

    const leftFileMax = fileMaxScores.get(leftChunk.file_path) ?? -Infinity;
    const rightFileMax = fileMaxScores.get(rightChunk.file_path) ?? -Infinity;
    const fileMaxDelta = rightFileMax - leftFileMax;
    if (fileMaxDelta !== 0) return fileMaxDelta;

    const familyDelta =
      familyPriorityOrdinal(leftChunk, priorityList) -
      familyPriorityOrdinal(rightChunk, priorityList);
    if (familyDelta !== 0) return familyDelta;

    if (leftChunk.file_path !== rightChunk.file_path) {
      return leftChunk.file_path.localeCompare(rightChunk.file_path);
    }

    return leftChunk.char_start - rightChunk.char_start;
  });
}

/**
 * Truncate text at the last sentence boundary on or before the target
 * character count. Falls back to a hard character truncation when no boundary
 * is found.
 *
 * @param {string} text
 * @param {number} maxChars
 * @returns {{ text: string, truncated: boolean }}
 */
function truncateAtSentenceBoundary(text, maxChars) {
  if (text.length <= maxChars) return { text, truncated: false };
  const clamped = Math.max(1, maxChars);
  const candidate = text.slice(0, clamped);
  const boundaryRegex = /[.!?]+(?:\s|$)/g;
  let lastBoundaryEnd = -1;
  let match;
  while ((match = boundaryRegex.exec(candidate)) !== null) {
    lastBoundaryEnd = match.index + match[0].length;
  }
  if (lastBoundaryEnd <= 0) {
    return {
      text: candidate.trimEnd(),
      truncated: true,
    };
  }
  return {
    text: candidate.slice(0, lastBoundaryEnd).trimEnd(),
    truncated: true,
  };
}

/**
 * Stage 4 — Enforce a token budget on ordered chunks. Essential chunks are
 * always selected (truncated if necessary), supporting chunks are included
 * only if they fit, and supplementary chunks are included only if they fit with
 * graceful sentence-boundary truncation for the final supplementary chunk.
 *
 * @param {AssembledChunk[]} chunks
 * @param {{
 *   budget?: number,
 *   charsPerToken?: number,
 *   countTokens?: (text: string) => number,
 * }} [options]
 * @returns {BudgetResult}
 */
export function enforceBudget(chunks, options = {}) {
  const budget = Math.max(0, options.budget ?? DEFAULT_BUDGET);
  const charsPerToken = options.charsPerToken ?? DEFAULT_CHARS_PER_TOKEN;
  const countTokens =
    typeof options.countTokens === 'function'
      ? options.countTokens
      : (text) => estimateTokenCount(text, charsPerToken);

  const selectedChunks = [];
  let remaining = budget;
  let truncated = false;
  const tierCounts = { essential: 0, supporting: 0, supplementary: 0 };

  const tierOrder = ['essential', 'supporting', 'supplementary'];
  const chunksByTier = new Map(
    tierOrder.map((tier) => [
      tier,
      chunks.filter((chunk) => chunk.tier === tier),
    ]),
  );

  for (const tier of tierOrder) {
    const tierChunks = chunksByTier.get(tier) ?? [];
    const isLastSupplementary = tier === 'supplementary';
    for (let chunkIndex = 0; chunkIndex < tierChunks.length; chunkIndex += 1) {
      const chunk = tierChunks[chunkIndex];
      const bodyText = getBodyText(chunk);
      const bodyTokens = countTokens(bodyText);

      if (bodyTokens <= remaining) {
        selectedChunks.push(chunk);
        remaining -= bodyTokens;
        tierCounts[tier] += 1;
        continue;
      }

      if (remaining <= 0) break;

      const mustInclude = tier === 'essential';
      const isLastChunk =
        isLastSupplementary && chunkIndex === tierChunks.length - 1;
      if (mustInclude || isLastChunk) {
        const { text: truncatedBody, truncated: wasTruncated } =
          truncateAtSentenceBoundary(bodyText, remaining * charsPerToken);
        const truncatedTokens = countTokens(truncatedBody);
        selectedChunks.push({
          ...chunk,
          body_text: truncatedBody,
          content: truncatedBody,
          truncated: true,
        });
        remaining = Math.max(0, remaining - truncatedTokens);
        tierCounts[tier] += 1;
        truncated = truncated || wasTruncated;
      }
      // Supporting and non-last supplementary chunks that do not fit are dropped.
      if (tier === 'essential') continue;
      if (!isLastChunk) continue;
      break;
    }
  }

  const tokenCount = budget - remaining;
  return {
    selectedChunks,
    tokenCount,
    budget,
    truncated,
    tierCounts,
  };
}

/**
 * Stage 5 — Stitch selected chunks into a context string.
 *
 * @param {AssembledChunk[]} chunks
 * @param {{ context_format?: 'markdown'|'json' }} [options]
 * @returns {string | StitchJsonResult}
 */
export function stitchContext(chunks, options = {}) {
  const format = options.context_format ?? 'markdown';
  if (format === 'json') {
    return stitchJson(chunks, options);
  }
  return stitchMarkdown(chunks, options);
}

/**
 * Produce a human-readable Markdown context string with provenance headers.
 *
 * @param {AssembledChunk[]} chunks
 * @param {{ context_format?: 'markdown'|'json' }} _options
 * @returns {string}
 */
function stitchMarkdown(chunks, _options) {
  const parts = [];
  let currentFile = null;
  for (const chunk of chunks) {
    const header = chunk.context_header ?? buildContextHeader(chunk);
    const body = getBodyText(chunk);
    if (chunk.file_path !== currentFile) {
      parts.push(`[${header}]\n${body}`);
      currentFile = chunk.file_path;
    } else {
      parts.push(body);
    }
  }
  return parts.join('\n\n');
}

/**
 * Produce a JSON context object with chunk metadata and a Markdown context string.
 *
 * @param {AssembledChunk[]} chunks
 * @param {{ context_format?: 'markdown'|'json' }} options
 * @returns {StitchJsonResult}
 */
function stitchJson(chunks, options) {
  const context = stitchMarkdown(chunks, options);
  const tokenCount = estimateTokenCount(context);
  const payloadChunks = chunks.map((chunk) => ({
    file_path: chunk.file_path,
    heading_path: chunk.heading_path ?? null,
    char_start: chunk.char_start,
    char_end: chunk.char_end,
    content: getBodyText(chunk),
    score: chunk.score,
    tier: chunk.tier ?? assignTier(chunk.score),
  }));
  return { context, chunks: payloadChunks, tokenCount };
}

/**
 * Compose the full five-stage pipeline.
 *
 * @param {InputChunk[]} chunks
 * @param {{
 *   budget?: number,
 *   charsPerToken?: number,
 *   countTokens?: (text: string) => number,
 *   context_format?: 'markdown'|'json',
 *   cosineThreshold?: number,
 *   essential?: number,
 *   supporting?: number,
 *   familyPriority?: string[],
 *   query_class?: string,
 * }} [options]
 * @returns {Promise<{ context: string, tokenCount: number, tierCounts: Object.<string, number>, selectedChunks: AssembledChunk[] }>}
 */
export async function assembleContext(chunks, options = {}) {
  const enriched = await enrichChunks(chunks, options);
  const deduped = deduplicateChunks(enriched, options);
  const ordered = orderChunks(deduped, options);
  const budgetResult = enforceBudget(ordered, options);
  const stitched = stitchContext(budgetResult.selectedChunks, options);
  const context = typeof stitched === 'string' ? stitched : stitched.context;

  return {
    context,
    tokenCount: budgetResult.tokenCount,
    tierCounts: budgetResult.tierCounts,
    selectedChunks: budgetResult.selectedChunks,
  };
}
