/**
 * @module assemble-context
 * @description Five-stage context-window assembly pipeline for Cortex search results.
 * Stateless, composable pure functions: enrich → deduplicate → order → budget → stitch.
 * Enrichment, deduplication, and initial ordering use server-side SQL when a database
 * client is provided; budget enforcement and final stitching remain client-side.
 *
 * @example
 * ```js
 * import { assembleContext } from './assemble-context.mjs';
 *
 * const result = await assembleContext(chunks, {
 *   budget: 4096,
 *   context_format: 'markdown',
 *   client: dbClient,
 * });
 * console.log(result.context, result.tokenCount, result.tierCounts);
 * ```
 */

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
 * Stage 1 — Enrich every chunk with document metadata and entity data using
 * server-side SQL JOINs. When a database client is provided, chunks are
 * enriched via a single SQL SELECT that JOINs chunks with documents and
 * entities, GROUP BY chunk_sha256 for dedup, and ORDER BY for initial
 * ordering. When no client is provided, a minimal JS-side fallback is used.
 *
 * @param {InputChunk[]} chunks
 * @param {{ query_class?: string, client?: import('@libsql/client').Client }} [options]
 * @returns {Promise<AssembledChunk[]>}
 */
export async function enrichChunks(chunks, options = {}) {
  const client = options.client;
  const queryClass = options.query_class ?? 'default';

  if (!client || chunks.length === 0) {
    return chunks.map((chunk) => ({
      ...chunk,
      body_text: getBodyText(chunk),
      query_class: queryClass,
    }));
  }

  const chunkIds = chunks
    .map((chunk) => chunk.chunk_id)
    .filter((id) => id != null);
  if (chunkIds.length === 0) {
    return chunks.map((chunk) => ({
      ...chunk,
      body_text: getBodyText(chunk),
      query_class: queryClass,
    }));
  }

  const placeholders = chunkIds.map(() => '?').join(',');
  const result = await client.execute({
    sql: `
      SELECT c.chunk_id, c.chunk_index, c.heading_path, c.body_text,
        c.char_start, c.char_end, c.parent_chunk_id, c.depth,
        c.context_header, c.symbol_name, c.signature_text, c.jsdoc_text,
        c.export_type, c.module_path, c.arch_layer, c.chunk_sha256,
        d.file_path, d.doc_family AS family,
        e.entity_type, e.qualified_name AS entity_name
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      LEFT JOIN entities e ON e.chunk_id = c.chunk_id
      WHERE c.chunk_id IN (${placeholders})
      GROUP BY c.chunk_id
      ORDER BY c.chunk_sha256
    `,
    args: chunkIds,
  });

  const enrichedMap = new Map();
  for (const row of result.rows) {
    enrichedMap.set(row.chunk_id, {
      chunk_id: row.chunk_id,
      chunk_index: row.chunk_index,
      heading_path: row.heading_path,
      body_text: row.body_text,
      char_start: row.char_start,
      char_end: row.char_end,
      parent_chunk_id: row.parent_chunk_id,
      depth: row.depth,
      context_header: row.context_header,
      symbol_name: row.symbol_name,
      signature_text: row.signature_text,
      jsdoc_text: row.jsdoc_text,
      export_type: row.export_type,
      module_path: row.module_path,
      arch_layer: row.arch_layer,
      sha256: row.chunk_sha256,
      file_path: row.file_path,
      family: row.family,
      entity_type: row.entity_type,
      entity_name: row.entity_name,
    });
  }

  return chunks.map((chunk) => {
    const enriched = enrichedMap.get(chunk.chunk_id);
    if (!enriched) {
      return {
        ...chunk,
        body_text: getBodyText(chunk),
        query_class: queryClass,
      };
    }
    return {
      ...chunk,
      ...enriched,
      score: chunk.score,
      query_class: queryClass,
    };
  });
}

/**
 * Stage 2 — Deduplicate chunks by content hash. Server-side SQL dedup
 * (GROUP BY chunk_sha256) is handled in the enrichment query. This
 * client-side pass collapses any remaining exact-hash duplicates.
 *
 * @param {AssembledChunk[]} chunks
 * @param {{ cosineThreshold?: number }} [options]
 * @returns {AssembledChunk[]}
 */
export function deduplicateChunks(chunks, options = {}) {
  const seen = new Set();
  const deduped = [];
  for (const chunk of chunks) {
    const key = chunk.sha256 ?? chunk.body_text ?? chunk.chunk_id;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(chunk);
  }
  return deduped;
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
    const heading = chunk.heading_path?.trim();
    const header =
      chunk.context_header ??
      (heading ? `${chunk.file_path} > ${heading}` : chunk.file_path);
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
 *   client?: import('@libsql/client').Client,
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
