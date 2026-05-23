import type {
  SearchSnapshotOptions,
  SemanticSnapshot,
  SemanticSnapshotDocument,
  SemanticSnapshotSearchResult,
} from './semantic-snapshot.types';

/** Default maximum number of ranked results returned by {@link searchSnapshot}. */
const DEFAULT_RESULT_LIMIT = 10;

/**
 * Score multiplier applied to each term occurrence found in a chunk's `heading_path`.
 * Higher than {@link BODY_MATCH_WEIGHT} so contextually titled sections rank first.
 */
const HEADING_MATCH_WEIGHT = 5;

/** Score multiplier applied to each term occurrence found in a chunk's `body_text`. */
const BODY_MATCH_WEIGHT = 1;

/**
 * Search a generated semantic snapshot with lightweight heading-weighted lexical scoring.
 *
 * Tokenizes {@link query} into unique lowercase Unicode tokens (pattern: `[\p{L}\p{N}_-]+`) and
 * scores each chunk by counting term occurrences in its `heading_path` (×{@link HEADING_MATCH_WEIGHT})
 * and `body_text` (×{@link BODY_MATCH_WEIGHT}). Results are sorted by descending score then
 * stable source order (document index, then chunk index within the document).
 *
 * The algorithm is intentionally simple: it does not stem, expand synonyms, or apply TF-IDF.
 * Heading-path weighting ensures chunks that name a concept in their heading rank above chunks
 * that only mention it in the body.
 *
 * @param snapshot - Browser snapshot produced by `scripts/semantic-index/build-browser-snapshot.mjs`.
 * @param query - User query text to split into Unicode-aware lowercase tokens.
 * @param options - Optional result limit (default: {@link DEFAULT_RESULT_LIMIT}).
 * @returns Ranked chunk matches sorted by descending score and stable source order.
 *
 * @example
 * ```ts
 * import { searchSnapshot } from '../shared/semantic/semantic-snapshot-search';
 *
 * const results = searchSnapshot(snapshot, 'NEAT activation', { limit: 5 });
 * for (const { score, document, chunk } of results) {
 *   console.log(score, document.file_path, chunk.heading_path);
 * }
 * ```
 */
export function searchSnapshot(
  snapshot: SemanticSnapshot,
  query: string,
  options: SearchSnapshotOptions = {},
): SemanticSnapshotSearchResult[] {
  const terms = tokenizeQuery(query);
  if (terms.length === 0) {
    return [];
  }

  return snapshot.documents
    .flatMap((documentRecord, documentOrder) =>
      scoreDocument(documentRecord, documentOrder, terms),
    )
    .filter(({ result }) => result.score > 0)
    .toSorted(compareScoredResults)
    .slice(0, options.limit ?? DEFAULT_RESULT_LIMIT)
    .map(({ result }) => result);
}

function scoreDocument(
  documentRecord: SemanticSnapshotDocument,
  documentOrder: number,
  terms: readonly string[],
) {
  return documentRecord.chunks.map((chunk, chunkOrder) => ({
    documentOrder,
    chunkOrder,
    result: {
      score: scoreChunk(chunk.heading_path, chunk.body_text, terms),
      document: documentRecord,
      chunk,
    },
  }));
}

function scoreChunk(
  headingPath: string,
  bodyText: string,
  terms: readonly string[],
): number {
  const normalizedHeading = headingPath.toLocaleLowerCase();
  const normalizedBody = bodyText.toLocaleLowerCase();

  return terms.reduce(
    (score, term) =>
      score +
      countOccurrences(normalizedHeading, term) * HEADING_MATCH_WEIGHT +
      countOccurrences(normalizedBody, term) * BODY_MATCH_WEIGHT,
    0,
  );
}

function tokenizeQuery(query: string): string[] {
  return [
    ...new Set(query.toLocaleLowerCase().match(/[\p{L}\p{N}_-]+/gu) ?? []),
  ];
}

function countOccurrences(text: string, term: string): number {
  if (!text.includes(term)) {
    return 0;
  }

  return text.split(term).length - 1;
}

function compareScoredResults(
  leftResult: {
    documentOrder: number;
    chunkOrder: number;
    result: SemanticSnapshotSearchResult;
  },
  rightResult: {
    documentOrder: number;
    chunkOrder: number;
    result: SemanticSnapshotSearchResult;
  },
): number {
  return (
    rightResult.result.score - leftResult.result.score ||
    leftResult.documentOrder - rightResult.documentOrder ||
    leftResult.chunkOrder - rightResult.chunkOrder
  );
}
