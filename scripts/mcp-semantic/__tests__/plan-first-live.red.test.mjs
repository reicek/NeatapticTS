/**
 * @module plan-first-live.red.test
 * @description RED contract for slice `P2-S6-1` (Phase 2 / Step 06 of
 * `plans/cortex-plan-first-retrieval.plans.md`): the LIVE plan-first proof.
 * The lane drives PLAN-SHAPED queries over the LIVE corpus index through the
 * EXISTING `search_corpus` MCP tool and asserts that the TOP-ranked results
 * carry the PLAN families, that the response reports NO family fallback and
 * NO classification fallback, and that the query class is the plan-specific
 * class - and it RECORDS the measured response as the archivable evidence for
 * the lane.
 *
 * @remarks
 * ### The contract this lane pins
 *
 * `AC-403-A` requires the LIVE index to prove the plan-first lane:
 *
 * - (a) the TOP-ranked results carry the PLAN families - EVERY returned row's
 *   family is inside the routed plan family set,
 * - (b) the response reports NO family fallback (`family_fallback === false`),
 * - (c) the response reports NO classification fallback
 *   (`classification_fallback === false`),
 * - (d) the query class is the plan-specific class
 *   (`query_class === 'plan_specific'`), and
 * - (e) the MEASURED response (the query class, the confidence, both flags and
 *   the family of EVERY returned row) is RECORDED as the archivable evidence
 *   for the lane.
 *
 * The routed plan family set is READ FROM the routing table
 * (`rag-index/routing-table.mjs` -> `ROUTING.plan_specific.family` is the CSV
 * `'plan,completed-plan'`). NO family name is re-typed in the EXECUTABLE part
 * of this file - every family assertion reads `ROUTED_PLAN_FAMILIES`, which is
 * parsed from that CSV, so the lane FOLLOWS the routing instead of pinning a
 * hand-copied family list.
 *
 * ### The MEASURED baseline (the live corpus index)
 *
 * Measured over the live index (`rag-index/data/turso-replica.sqlite`) with
 * `use_dense: false` and `limit: 10`:
 *
 * | query | query_class | confidence | family_fallback | classification_fallback | row families |
 * |---|---|---|---|---|---|
 * | `design specification for network activation runtime` | `plan_specific` | `0.85` | `false` | `false` | 10 rows: 9 `plan` + 1 `completed-plan` |
 * | `roadmap plan acceptance criteria for the release` | `plan_specific` | `0.85` | `false` | `false` | 10 rows: 1 `completed-plan` + 9 `plan` |
 * | `plan for the corpus reindex and index validation` | `plan_specific` | `0.85` | `false` | `false` | 10 rows: 8 `plan` + 2 `completed-plan` |
 * | `what does the plan say about the routing table family` | `plan_specific` | `0.85` | `false` | `false` | 6 rows, all `plan` |
 * | **`how should the network activation runtime be specified in the plan`** | `plan_specific` | `0.85` | **`true`** | **`true`** | **2 rows, BOTH `ts-source`** (`src/methods/methods.ts`, `src/architecture/network/network.utils.ts`) |
 * | `planned slice for the query classification fallback` | `plan_specific` | `0.85` | `false` | `false` | 10 rows: 9 `plan` + 1 `completed-plan` |
 *
 * The first query is the plan's own canonical plan-shaped query and it is the
 * one the recorded evidence is archived for.
 *
 * ### The MEASURED cause of the failing query (MEASURED, NOT GUESSED)
 *
 * The routed family filter IS applied to that query - it is the ROUTED-FAMILY
 * SEARCH THAT MATCHES NOTHING that lets the non-plan families through:
 *
 * 1. The classifier routes the query to `plan_specific`, so the routed family
 *    set (`'plan,completed-plan'`) is the filter.
 * 2. The routed-family BM25 search matches **ZERO** plan-family chunks. The
 *    corpus-wide FTS match set for that query is exactly TWO chunks and BOTH
 *    are `ts-source` (`src/methods/methods.ts`,
 *    `src/architecture/network/network.utils.ts`); the plan-family subset of
 *    that match set is EMPTY.
 * 3. The broaden-on-empty safety net therefore FIRES
 *    (`scripts/mcp-semantic/tools/search-corpus.mjs:925-947`) and RE-RUNS the
 *    search with `family: null` and `skip_family_classification: true` - the
 *    re-run carries NO family filter at all.
 * 4. The response the caller receives is that broadened response, so it
 *    carries `family_fallback: true` AND `classification_fallback: true`, it
 *    emits NO response-level routed `family` field (the ROWS still carry their
 *    own per-document `family`), and its TOP-ranked rows are `ts-source` -
 *    the plan families are NOT at the top of a plan-shaped query.
 *
 * MEASURED PROBES (the same live index, `use_dense: false`, `limit: 10`):
 *
 * - the failing query WITH the routed set applied -> `0` rows,
 *   `family_fallback: false`, `classification_fallback: false` (the filtered
 *   search really matches nothing);
 * - the failing query WITHOUT any family filter -> `2` rows, both
 *   `ts-source`, `family_fallback: true`, `classification_fallback: true`;
 * - the failing query's CONTENT TERMS (`network activation runtime`) -> `10`
 *   rows, `family_fallback: false`, `classification_fallback: false`.
 *
 * ### Why the failing query is a RETRIEVAL defect and NOT a coverage gap
 *
 * The corpus DOES hold plan-family chunks that cover the failing query's
 * content terms: the routed plan family set, pinned through
 * `classification_hints.family`, returns 10 `completed-plan` rows for
 * `network activation runtime`. So the corpus HOLDS the plan answer and the
 * routed family filter is applied - the failing query is a RETRIEVAL defect in
 * the plan-first lane and NOT a corpus coverage gap. The anti-vacuity lane
 * below pins exactly that, so the RED lanes cannot be satisfied by an empty
 * result set.
 *
 * ### The corpus is NOT rewritten by this lane
 *
 * The live index is READ-ONLY for this test: the lane never re-indexes, never
 * re-chunks, and never mutates a document, a chunk or a family assignment -
 * it only READS the corpus through the EXISTING `search_corpus` tool. (The
 * tool's own existing impression telemetry is the search path's normal
 * behavior and does not touch the corpus content or the family assignments.)
 * The live replica's FILE BYTES can still move - the search path writes that
 * telemetry and the declared `rag-index` sweep can re-index the corpus - so
 * the measured baseline below must be RE-MEASURED if a reindex runs between
 * this lane and the GREEN lane.
 *
 * ### THE LANE SHAPE
 *
 * - `AC-403-A` - FOUR named lanes: the GREEN classification lane (every
 *   fixture query classifies as the plan-specific class with a numeric
 *   confidence), the RED family lane (no returned row outside the routed plan
 *   family set), the RED family-fallback lane (no family fallback reported)
 *   and the RED classification-fallback lane (no classification fallback
 *   reported). The three RED lanes are RED BY ASSERTION and each failure
 *   names the offending query (and, for the family lane, the leaked family).
 * - the anti-vacuity lane - GREEN: the routed plan family set DOES hold
 *   chunks that cover the failing query's content terms.
 * - the recorded-evidence lanes - GREEN: the measured response carries the
 *   classification metadata, both fallback flags and a family on every row,
 *   and the measured response is ARCHIVED to
 *   `artifacts/slice-P2-S6-1-live-response.json` - the canonical plan-shaped
 *   response at the top level AND the full measured matrix over EVERY
 *   plan-shaped fixture query, so the tracker evidence carries the query
 *   class, the confidence, both flags and the family of every returned row
 *   for every measured query (including the query whose live ranking leaks a
 *   non-plan family today).
 *
 * Every lane drives the EXISTING `search_corpus` tool over the EXISTING live
 * index: NO lane mocks the ranking away, NO lane creates an import failure
 * and NO lane uses a fake index.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';

import { createRepoCortexMcpServer } from '../repo-cortex-mcp.mjs';
import { defaultDatabasePath } from '../../../rag-index/init-schema.mjs';
import { ROUTING } from '../../../rag-index/routing-table.mjs';

/** The live corpus index the lanes read. */
const CORPUS_DATABASE_PATH = defaultDatabasePath;

/**
 * The CSV family list the routing table publishes for a plan-shaped query.
 *
 * @type {string}
 */
const ROUTED_PLAN_FAMILY_CSV = ROUTING.plan_specific.family;

/**
 * The routed family SET, parsed from the CSV the routing table publishes.
 *
 * @type {string[]}
 */
const ROUTED_PLAN_FAMILIES = ROUTED_PLAN_FAMILY_CSV.split(',')
  .map((family) => family.trim())
  .filter((family) => family.length > 0);

/** The query class a plan-shaped query is expected to classify as. */
const PLAN_QUERY_CLASS = 'plan_specific';

/**
 * The six plan-shaped queries the lane measures over the LIVE index.
 *
 * The first entry is the plan's own canonical plan-shaped query.
 *
 * @type {string[]}
 */
const PLAN_SHAPED_QUERIES = [
  'design specification for network activation runtime',
  'roadmap plan acceptance criteria for the release',
  'plan for the corpus reindex and index validation',
  'what does the plan say about the routing table family',
  'how should the network activation runtime be specified in the plan',
  'planned slice for the query classification fallback',
];

/** The canonical plan-shaped query - the one the recorded evidence covers. */
const CANONICAL_PLAN_SHAPED_QUERY = PLAN_SHAPED_QUERIES[0];

/**
 * The content query used by the anti-vacuity lane - the content terms of the
 * query whose live ranking leaks a non-plan family today.
 */
const ANTI_VACUITY_CONTENT_QUERY = 'network activation runtime';

/**
 * The archivable evidence path for the recorded measured response.
 *
 * @type {string}
 */
const ARTIFACT_PATH = join(
  process.cwd(),
  'artifacts',
  'slice-P2-S6-1-live-response.json',
);

/**
 * The MCP server the lanes drive - built ONCE and reused by every lane.
 *
 * @type {{ dispatch: (request: Record<string, unknown>) => Promise<any> } | null}
 */
let corpusServer = null;

/**
 * The measured live response per fixture query, keyed by the query text.
 *
 * @type {Map<string, Record<string, any>>}
 */
const MEASURED_RESPONSES = new Map();

/**
 * Call the existing `search_corpus` MCP tool and return its structured result.
 *
 * @param {Record<string, unknown>} args - The tool arguments.
 * @returns {Promise<Record<string, any>>} The tool structured content.
 */
async function callSearchCorpus(args) {
  const response = await corpusServer.dispatch({
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/call',
    params: { name: 'search_corpus', arguments: args },
  });
  if (response?.isError) {
    throw new Error(
      `search_corpus returned an error: ${JSON.stringify(response.content)}`,
    );
  }
  return response.structuredContent ?? {};
}

/**
 * Collect the row families that fall OUTSIDE the routed plan family set.
 *
 * @param {Record<string, any>} response - The tool structured content.
 * @returns {string[]} The unique leaked families.
 */
function leakedFamilies(response) {
  const rows = Array.isArray(response.results) ? response.results : [];
  return [
    ...new Set(
      rows
        .map((row) => row?.family)
        .filter((family) => !ROUTED_PLAN_FAMILIES.includes(family)),
    ),
  ];
}

/**
 * Count the returned rows that carry a family INSIDE the routed family set.
 *
 * @param {Record<string, any>} response - The tool structured content.
 * @returns {number} The number of plan-family rows.
 */
function planFamilyRowCount(response) {
  const rows = Array.isArray(response.results) ? response.results : [];
  return rows.filter((row) => ROUTED_PLAN_FAMILIES.includes(row?.family))
    .length;
}

/**
 * Collect the per-row family violations against the routed plan family set.
 *
 * Every violation names the offending query and the leaked family, so the
 * failure message of the family lane is self-explaining.
 *
 * @param {Record<string, any>} response - The tool structured content.
 * @param {string} query - The query text the response was measured for.
 * @returns {Array<{ query: string, leaked_family: string | null, file_path: string | null }>} The violations.
 */
function familyViolations(response, query) {
  const rows = Array.isArray(response.results) ? response.results : [];
  const leaked = new Set(leakedFamilies(response));
  return rows
    .filter((row) => leaked.has(row?.family))
    .map((row) => ({
      query,
      leaked_family: row?.family ?? null,
      file_path: row?.file_path ?? null,
    }));
}

/**
 * The returned rows that carry NO family at all.
 *
 * @param {Record<string, any>} response - The tool structured content.
 * @returns {string[]} The file paths of the family-less rows.
 */
function rowsWithoutFamily(response) {
  const rows = Array.isArray(response.results) ? response.results : [];
  return rows
    .filter((row) => typeof row?.family !== 'string' || row.family.length === 0)
    .map((row) => row?.file_path ?? null);
}

/**
 * Build the archivable evidence payload from the measured live responses.
 *
 * The payload carries the canonical plan-shaped response at the top level (the
 * positive plan-first proof of `AC-403-A`) AND the full measured matrix over
 * EVERY plan-shaped fixture query, so the tracker evidence shows the query
 * class, the confidence, both fallback flags and the family of every returned
 * row for every measured query - including the query whose live ranking
 * leaks a non-plan family today.
 *
 * @param {Record<string, any>} canonicalResponse - The canonical measured response.
 * @returns {Record<string, unknown>} The archivable evidence payload.
 */
function buildEvidencePayload(canonicalResponse) {
  const measurements = PLAN_SHAPED_QUERIES.map((query) => {
    const response = MEASURED_RESPONSES.get(query) ?? canonicalResponse ?? {};
    const rows = Array.isArray(response.results) ? response.results : [];
    return {
      query,
      query_class: response.query_class ?? null,
      confidence: response.confidence ?? null,
      family_fallback: response.family_fallback ?? null,
      classification_fallback: response.classification_fallback ?? null,
      families: rows.map((row) => row?.family ?? null),
      rows: rows.map((row) => ({
        family: row?.family ?? null,
        file_path: row?.file_path ?? null,
        score: typeof row?.score === 'number' ? row.score : null,
      })),
    };
  });
  const canonical =
    measurements.find(
      (measurement) => measurement.query === CANONICAL_PLAN_SHAPED_QUERY,
    ) ?? {};
  return {
    slice_id: 'P2-S6-1',
    corpus_database_path: relative(process.cwd(), CORPUS_DATABASE_PATH)
      .split(sep)
      .join('/'),
    routed_plan_families: ROUTED_PLAN_FAMILIES,
    query: canonical.query ?? CANONICAL_PLAN_SHAPED_QUERY,
    query_class: canonical.query_class ?? null,
    confidence: canonical.confidence ?? null,
    family_fallback: canonical.family_fallback ?? null,
    classification_fallback: canonical.classification_fallback ?? null,
    families: canonical.families ?? [],
    rows: canonical.rows ?? [],
    measurements,
  };
}

/**
 * Write the measured live response to the archivable evidence artifact.
 *
 * @param {Record<string, any>} response - The measured tool structured content.
 * @returns {Record<string, unknown>} The written evidence payload.
 */
function writeEvidenceArtifact(response) {
  const payload = buildEvidencePayload(response);
  mkdirSync(dirname(ARTIFACT_PATH), { recursive: true });
  writeFileSync(ARTIFACT_PATH, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  return payload;
}

beforeAll(async () => {
  corpusServer = createRepoCortexMcpServer({
    databasePath: CORPUS_DATABASE_PATH,
  });
  for (const query of PLAN_SHAPED_QUERIES) {
    MEASURED_RESPONSES.set(
      query,
      await callSearchCorpus({ query, use_dense: false, limit: 10 }),
    );
  }
});

describe('plan-first-live.red', () => {
  describe('AC-403-A: the live plan-first proof', () => {
    it('classifies every plan-shaped fixture query as the plan-specific class', () => {
      // Arrange / Act: the fixture responses were measured in `beforeAll`.
      const misclassified = PLAN_SHAPED_QUERIES.filter(
        (query) =>
          MEASURED_RESPONSES.get(query)?.query_class !== PLAN_QUERY_CLASS,
      );
      const outOfRange = PLAN_SHAPED_QUERIES.filter((query) => {
        const confidence = MEASURED_RESPONSES.get(query)?.confidence;
        return (
          typeof confidence !== 'number' || confidence <= 0 || confidence > 1
        );
      });

      // Assert
      expect(misclassified).toEqual([]);
      expect(outOfRange).toEqual([]);
    });

    it('returns only rows inside the routed plan family set for every plan-shaped query', () => {
      // Arrange / Act: the fixture responses were measured in `beforeAll`.
      const violations = PLAN_SHAPED_QUERIES.flatMap((query) =>
        familyViolations(MEASURED_RESPONSES.get(query), query),
      );

      // Assert: every violation names the offending query and leaked family.
      expect(violations).toEqual([]);
    });

    it('reports no family fallback for any plan-shaped query', () => {
      // Arrange / Act: the fixture responses were measured in `beforeAll`.
      const offending = PLAN_SHAPED_QUERIES.filter(
        (query) => MEASURED_RESPONSES.get(query)?.family_fallback !== false,
      );

      // Assert
      expect(offending).toEqual([]);
    });

    it('reports no classification fallback for any plan-shaped query', () => {
      // Arrange / Act: the fixture responses were measured in `beforeAll`.
      const offending = PLAN_SHAPED_QUERIES.filter(
        (query) =>
          MEASURED_RESPONSES.get(query)?.classification_fallback !== false,
      );

      // Assert
      expect(offending).toEqual([]);
    });
  });

  describe('anti-vacuity: the corpus DOES hold the plan answer', () => {
    it('returns routed plan-family chunks for the failing query content terms', async () => {
      // Arrange
      const args = {
        query: ANTI_VACUITY_CONTENT_QUERY,
        query_class: PLAN_QUERY_CLASS,
        classification_hints: { family: ROUTED_PLAN_FAMILY_CSV },
        use_dense: false,
        limit: 10,
      };

      // Act
      const response = await callSearchCorpus(args);

      // Assert: the corpus HOLDS the plan answer, so the RED lanes are a
      // retrieval defect and not a corpus coverage gap.
      expect(planFamilyRowCount(response)).toBeGreaterThan(0);
    });
  });

  describe('the recorded measured response (the archivable evidence)', () => {
    /** The measured live response for the canonical plan-shaped query. */
    let measuredResponse = null;

    /** The archived evidence payload written to the artifact. */
    let evidencePayload = null;

    beforeAll(() => {
      // Reuse the canonical measurement taken by the file-level `beforeAll`
      // so the archived evidence is the SAME live response the RED lanes saw.
      measuredResponse = MEASURED_RESPONSES.get(CANONICAL_PLAN_SHAPED_QUERY);
      evidencePayload = writeEvidenceArtifact(measuredResponse);
    });

    it('carries the plan-specific query class on the canonical plan-shaped query', () => {
      // Arrange / Act: the canonical response was measured in `beforeAll`.
      const queryClass = measuredResponse?.query_class;

      // Assert
      expect(queryClass).toBe(PLAN_QUERY_CLASS);
    });

    it('carries a numeric confidence in (0, 1] on the canonical plan-shaped query', () => {
      // Arrange / Act: the canonical response was measured in `beforeAll`.
      const confidence = measuredResponse?.confidence;

      // Assert
      expect(typeof confidence).toBe('number');
      expect(confidence > 0 && confidence <= 1).toBe(true);
    });

    it('carries both fallback flags as booleans on the canonical plan-shaped query', () => {
      // Arrange / Act: the canonical response was measured in `beforeAll`.
      const { family_fallback: familyFallback } = measuredResponse;
      const { classification_fallback: classificationFallback } =
        measuredResponse;

      // Assert
      expect(typeof familyFallback).toBe('boolean');
      expect(typeof classificationFallback).toBe('boolean');
    });

    it('gives every returned row a family on the canonical plan-shaped query', () => {
      // Arrange / Act: the canonical response was measured in `beforeAll`.
      const withoutFamily = rowsWithoutFamily(measuredResponse);

      // Assert
      expect(withoutFamily).toEqual([]);
    });

    it('records the measured response as the archivable evidence', () => {
      // Arrange / Act: the artifact was written in `beforeAll`.
      const recorded = {
        query: evidencePayload.query,
        query_class: evidencePayload.query_class,
        confidence: evidencePayload.confidence,
        family_fallback: evidencePayload.family_fallback,
        classification_fallback: evidencePayload.classification_fallback,
      };

      // Assert
      expect(recorded).toEqual({
        query: CANONICAL_PLAN_SHAPED_QUERY,
        query_class: measuredResponse.query_class,
        confidence: measuredResponse.confidence,
        family_fallback: measuredResponse.family_fallback,
        classification_fallback: measuredResponse.classification_fallback,
      });
      expect(
        evidencePayload.rows.every(
          (row) => typeof row.family === 'string' && row.family.length > 0,
        ),
      ).toBe(true);
    });

    it('records the measured response of every plan-shaped fixture query', () => {
      // Arrange / Act: the artifact was written in `beforeAll`.
      const recordedQueries = (evidencePayload.measurements ?? []).map(
        (measurement) => measurement.query,
      );

      // Assert: the tracker evidence covers EVERY measured query, so the
      // defect proof is archived next to the positive plan-first proof.
      expect(recordedQueries).toEqual(PLAN_SHAPED_QUERIES);
    });

    it('records the query class, the confidence and both fallback flags of every measured query', () => {
      // Arrange / Act: the artifact was written in `beforeAll`.
      const incomplete = (evidencePayload.measurements ?? []).filter(
        (measurement) =>
          typeof measurement.query_class !== 'string' ||
          typeof measurement.confidence !== 'number' ||
          typeof measurement.family_fallback !== 'boolean' ||
          typeof measurement.classification_fallback !== 'boolean',
      );

      // Assert
      expect(incomplete).toEqual([]);
    });
  });
});
