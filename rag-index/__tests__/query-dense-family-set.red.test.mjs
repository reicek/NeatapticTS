/**
 * @module query-dense-family-set.red.test
 * @description RED contract for the routed CSV family set in the DENSE path
 * (Phase 2, Step 4 - AC-401-A2).
 *
 * The home drives the REAL `queryDenseIndex` of `rag-index/query-dense.mjs` over a CONTROLLED
 * DB CLIENT - the owner-local mock pattern of `rag-index/query-dense.test.mjs`. The CLIENT is
 * mocked; the SQL SEMANTICS are NOT: every SQL string and every bound argument the dense path
 * produces is the production one.
 *
 * The `embedText` embedder and the `modelMeta` descriptor are INJECTED, so NO ONNX model is
 * loaded, NO live index is needed, and there is NO RNG, NO clock and NO network.
 *
 * The family value is READ FROM THE ROUTING TABLE (`ROUTING.plan_specific.family`) and is NOT
 * re-typed here, so the home follows the routing.
 *
 * The EXPECTED predicate is compiled by the REAL `rag-index/metadata-filter.mjs`
 * (`compileFamilySetFilter`), which this home does NOT `jest.unstable_mockModule`, so the
 * contract is measured against the real helper and not against a mock.
 *
 * RED PHASE: all three single-family SQL sites of the dense path
 * (`rag-index/query-dense.mjs:199` the BM25 candidate loader, `:309` the ANN candidate path and
 * `:372` the dense top-k / brute-force path) compare the WHOLE CSV for EQUALITY against
 * `d.doc_family` (`AND d.doc_family = ?`). The routed CSV is therefore bound as ONE equality
 * parameter, and no document carries the literal family name `plan,completed-plan`, so the
 * routed filter matches ZERO dense rows while the dense index is warm.
 */

import { describe, expect, it, jest } from '@jest/globals';

import { compileFamilySetFilter } from '../metadata-filter.mjs';
import { queryDenseIndex } from '../query-dense.mjs';
import { ROUTING } from '../routing-table.mjs';

// ---------------------------------------------------------------------------
// The routed family values (READ from the routing table, NEVER re-typed)
// ---------------------------------------------------------------------------

/** The routed plan-family filter: a CSV LIST of family names. */
const ROUTED_FAMILY = ROUTING.plan_specific.family;

/** The routed plan-family members, in routing order. */
const ROUTED_FAMILY_MEMBERS = ROUTED_FAMILY.split(',')
  .map((name) => name.trim())
  .filter((name) => name.length > 0);

/** A routed SINGLE-family filter, used by the anti-vacuity lane. */
const ROUTED_SINGLE_FAMILY = ROUTING.code_specific.family;

/**
 * Every string that may legitimately appear as a BOUND family argument: the routed CSV itself
 * (the value that must NEVER be bound) and each routed family member.
 */
const FAMILY_ARGUMENT_CANDIDATES = [
  ROUTED_FAMILY,
  ...ROUTED_FAMILY_MEMBERS,
  ROUTED_SINGLE_FAMILY,
];

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** A plan-shaped query for the dense path. */
const PLAN_SHAPED_QUERY = 'plan tracker acceptance criteria';

/** The injected model descriptor: no ONNX model is loaded, no metadata file is read. */
const INJECTED_MODEL_META = Object.freeze({
  dimension: 4,
  model_id: 'all-MiniLM-L6-v2',
  model_sha256: 'red-contract-model',
});

/**
 * Build the injected text embedder: a deterministic vector, no model, no network.
 *
 * @returns {Function} A jest mock resolving to a 4-dimensional Float32Array.
 */
function makeEmbedText() {
  return jest.fn().mockResolvedValue(new Float32Array([0.1, 0.2, 0.3, 0.4]));
}

/**
 * Build the controlled DB client: the SQL and the bound arguments are captured, the rows are
 * irrelevant to this contract.
 *
 * @returns {{ execute: Function, close: Function }} The controlled client.
 */
function makeControlledClient() {
  return {
    execute: jest.fn(),
    close: jest.fn().mockResolvedValue(undefined),
  };
}

// ---------------------------------------------------------------------------
// The dense drive
// ---------------------------------------------------------------------------

/**
 * Drive the dense path and capture the SQL of its three single-family SQL sites.
 *
 * The controlled client answers the BM25 candidate loader with an empty row set and REJECTS the
 * ANN candidate call, so the dense path falls through to the brute-force (dense top-k) path.
 * The three captured calls are, in order: the BM25 candidate loader
 * (`rag-index/query-dense.mjs:199`), the ANN candidate path (`:309`) and the dense top-k /
 * brute-force path (`:372`).
 *
 * @param {string | undefined} family - The family filter to route (undefined = no filter).
 * @returns {Promise<{ bm25: object, ann: object, bruteForce: object }>} The captured calls.
 */
async function captureDenseFamilySql(family) {
  const client = makeControlledClient();
  client.execute
    .mockResolvedValueOnce({ rows: [] }) // the BM25 candidate loader
    .mockRejectedValueOnce(new Error('ANN index cold')) // forces the brute-force fallback
    .mockResolvedValueOnce({ rows: [] }); // the dense top-k / brute-force path

  await queryDenseIndex({
    query: PLAN_SHAPED_QUERY,
    client,
    dense: true,
    family,
    embedText: makeEmbedText(),
    modelMeta: INJECTED_MODEL_META,
  });

  const calls = client.execute.mock.calls.map((call) => call[0]);
  if (calls.length !== 3) {
    throw new Error(
      `Expected 3 dense SQL calls (BM25, ANN, brute-force), captured ${calls.length}.`,
    );
  }
  const [bm25, ann, bruteForce] = calls;
  return { bm25, ann, bruteForce };
}

// ---------------------------------------------------------------------------
// The measurement
// ---------------------------------------------------------------------------

/**
 * Normalize a family predicate fragment so the measured and the expected forms are comparable
 * regardless of the whitespace inside the placeholder list.
 *
 * @param {string} fragment - The predicate fragment.
 * @returns {string} The normalized fragment.
 */
function normalizePredicateFragment(fragment) {
  return String(fragment)
    .replace(/\s+/g, ' ')
    .replace(/\(\s*\?(\s*,\s*\?)*\s*\)/g, (placeholders) =>
      placeholders.replace(/\s+/g, ''),
    )
    .trim();
}

/**
 * Extract the WHERE clause of a SQL statement.
 *
 * The `d.doc_family` column also appears in the SELECT list of every dense statement, so the
 * family PREDICATE is only ever read from the WHERE clause.
 *
 * @param {string} sql - The SQL statement.
 * @returns {string} The WHERE clause, or an empty string when there is none.
 */
function extractWhereClause(sql) {
  const normalized = String(sql ?? '').replace(/\s+/g, ' ');
  const whereIndex = normalized.toUpperCase().indexOf(' WHERE ');
  return whereIndex === -1 ? '' : normalized.slice(whereIndex + ' WHERE '.length);
}

/**
 * Extract the document-family PREDICATE from a SQL statement.
 *
 * @param {string} sql - The SQL statement.
 * @returns {{ operator: string, fragment: string, placeholderCount: number } | null} The
 *   predicate, or null when the statement carries no family predicate.
 */
function extractFamilyPredicate(sql) {
  const match = extractWhereClause(sql).match(
    /d\.doc_family\s*(=|\bIN\b)\s*(\([^)]*\)|\?)/i,
  );
  if (!match) return null;
  return {
    operator: match[1].toUpperCase(),
    fragment: normalizePredicateFragment(
      `d.doc_family ${match[1].toUpperCase()} ${match[2]}`,
    ),
    placeholderCount: (match[2].match(/\?/g) ?? []).length,
  };
}

/**
 * Collect the bound arguments that carry a document-family value.
 *
 * @param {Array<unknown>} args - The bound arguments of the SQL call.
 * @returns {string[]} The bound family values, sorted.
 */
function collectBoundFamilyValues(args) {
  return (args ?? [])
    .filter(
      (argument) =>
        typeof argument === 'string' &&
        FAMILY_ARGUMENT_CANDIDATES.includes(argument),
    )
    .toSorted();
}

/**
 * Measure the family contract of one SQL call: the predicate it carries and the family values
 * it binds.
 *
 * @param {{ sql: string, args: Array<unknown> }} call - The captured SQL call.
 * @returns {{ predicate: string | null, placeholderCount: number, boundFamilyValues: string[] }}
 *   The measured contract.
 */
function measureFamilyContract(call) {
  const predicate = extractFamilyPredicate(call.sql);
  return {
    predicate: predicate?.fragment ?? null,
    placeholderCount: predicate?.placeholderCount ?? 0,
    boundFamilyValues: collectBoundFamilyValues(call.args),
  };
}

/**
 * The family contract the REAL helper publishes for a routed family value.
 *
 * @param {string | undefined} family - The routed family value.
 * @returns {{ predicate: string | null, placeholderCount: number, boundFamilyValues: string[] }}
 *   The expected contract.
 */
function expectedFamilyContract(family) {
  const compiled = compileFamilySetFilter(family);
  return {
    predicate: compiled.sql ? normalizePredicateFragment(compiled.sql) : null,
    placeholderCount: compiled.params.length,
    boundFamilyValues: compiled.params.toSorted(),
  };
}

// ---------------------------------------------------------------------------
// AC-401-A2 - the routed CSV reaches the three dense SQL sites
// ---------------------------------------------------------------------------

describe('query-dense-family-set: the routed family set reaches the dense SQL', () => {
  describe('AC-401-A2 the routed CSV reaches the three dense SQL sites', () => {
    it('binds the routed plan-family set as separate parameters in the BM25 candidate SQL', async () => {
      const { bm25 } = await captureDenseFamilySql(ROUTED_FAMILY);

      expect(measureFamilyContract(bm25)).toEqual(
        expectedFamilyContract(ROUTED_FAMILY),
      );
    });

    it('binds the routed plan-family set as separate parameters in the ANN candidate SQL', async () => {
      const { ann } = await captureDenseFamilySql(ROUTED_FAMILY);

      expect(measureFamilyContract(ann)).toEqual(
        expectedFamilyContract(ROUTED_FAMILY),
      );
    });

    it('binds the routed plan-family set as separate parameters in the dense top-k / brute-force SQL', async () => {
      const { bruteForce } = await captureDenseFamilySql(ROUTED_FAMILY);

      expect(measureFamilyContract(bruteForce)).toEqual(
        expectedFamilyContract(ROUTED_FAMILY),
      );
    });
  });

  // -------------------------------------------------------------------------
  // AC-401-A2 - the anti-vacuity lanes (these are GREEN today and must stay GREEN)
  // -------------------------------------------------------------------------

  describe('AC-401-A2 the anti-vacuity lanes', () => {
    it('carries no document-family predicate when no family is routed', async () => {
      const { bm25, ann, bruteForce } = await captureDenseFamilySql(undefined);

      expect({
        bm25: extractFamilyPredicate(bm25.sql),
        ann: extractFamilyPredicate(ann.sql),
        bruteForce: extractFamilyPredicate(bruteForce.sql),
      }).toEqual({ bm25: null, ann: null, bruteForce: null });
    });

    it('keeps the single-family equality predicate with exactly one bound parameter', async () => {
      const { bm25, ann, bruteForce } = await captureDenseFamilySql(
        ROUTED_SINGLE_FAMILY,
      );
      const expected = expectedFamilyContract(ROUTED_SINGLE_FAMILY);

      expect({
        bm25: measureFamilyContract(bm25),
        ann: measureFamilyContract(ann),
        bruteForce: measureFamilyContract(bruteForce),
      }).toEqual({ bm25: expected, ann: expected, bruteForce: expected });
    });
  });
});
