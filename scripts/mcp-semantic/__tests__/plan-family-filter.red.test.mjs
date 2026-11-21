/**
 * @module plan-family-filter.red.test
 * @description RED contract for slice `P2-S4-1` (Phase 2 / Step 04 of
 * `plans/cortex-plan-first-retrieval.plans.md`): the CSV family filter the
 * routing publishes MUST reach the corpus SQL as a SET filter, and the
 * broaden-on-empty safety net must be REPORTED only when it really fired.
 *
 * @remarks
 * ### The defect this lane pins
 *
 * The routing publishes the plan family as a CSV LIST
 * (`rag-index/routing-table.mjs` -> `ROUTING.plan_specific.family` is
 * `'plan,completed-plan'`), and the query classifier publishes the same CSV
 * (`rag-index/classify-query.mjs` -> `EMBEDDED_FAMILY_DEFAULTS.plan_specific`).
 * `scripts/mcp-semantic/tools/search-corpus.mjs` compares that whole CSV
 * string for EQUALITY against a single `d.doc_family` column (at both SQL
 * sites - the exact-symbol lookup and the BM25 full-text search), so a routed
 * filtered query can NEVER match a row.
 *
 * The broaden-on-empty safety net then silently re-runs the search with NO
 * family filter and reports the broaden (`family_fallback: true` AND
 * `classification_fallback: true`) while returning rows from EVERY family.
 * The response therefore LOOKS like a successful plan-first search while
 * leaking non-plan families.
 *
 * ### The MEASURED current response (the baseline the green lane compares to)
 *
 * Measured by this lane over the live corpus index
 * (`rag-index/data/turso-replica.sqlite`) with the query
 * `'design specification for network activation runtime'` and
 * `use_dense: false`:
 *
 * - `query_class`: `plan_specific`
 * - `confidence`: `0.85`
 * - `family_fallback`: `true` - the routed family set was silently dropped
 * - `classification_fallback`: `true`
 * - returned row families: `ts-source`, `plan`, `completed-plan`
 *
 * `ts-source` is the LEAK: it is OUTSIDE the routed family set (`plan`,
 * `completed-plan`). The GREEN lane must return plan-family rows with
 * `family_fallback: false` AND `classification_fallback: false`.
 *
 * ### The fixtures (measured over the same live index)
 *
 * - `PLAN_SHAPED_QUERY` - the BROADENED response today is 6 rows: 3 `plan`,
 *   1 `completed-plan` and 2 `ts-source`. The 4 in-set rows (3 `plan` + 1
 *   `completed-plan`) are the rows the set filter MUST return, and the 2
 *   `ts-source` rows are the LEAK.
 * - `PLAN_MATCHING_QUERY` - the BROADENED response today is 10 rows: 8
 *   `completed-plan` and 2 `plan`. EVERY row is inside the routed set, so
 *   this fixture isolates the fallback flag from the leak: it fails ONLY
 *   because the routed family set was dropped and the search was broadened.
 * - `NOTHING_MATCHING_QUERY` - matches NOTHING anywhere in the corpus (0 rows
 *   with and without the family filter), so the safety net must fire and
 *   MUST BE REPORTED. EVERY token of the query is MEASURED against the live
 *   index to match ZERO rows in the prefix form the sanitizer emits, so the
 *   OR-joined expression (5 tokens, above the 3-token implicit-AND limit)
 *   cannot hit a row either. Its routed family is PINNED through
 *   `classification_hints.family` (the routed CSV read from the routing
 *   table), so the fixture does NOT depend on how the query text routes.
 *
 * Every lane drives the EXISTING `search_corpus` tool over the EXISTING index:
 * NO lane mocks the SQL away and NO lane creates an import failure.
 *
 * ### THE LANE SHAPE
 *
 * - `AC-401-A` - THREE lanes: the RED leak lane (no row outside the routed
 *   set AND no family fallback), the RED classification lane (no
 *   classification fallback) and the GREEN GUARD lane (plan-family chunks
 *   ARE returned). The guard lane is what makes the RED lanes NON-VACUOUS: an
 *   empty result set would satisfy "no row outside the routed set" on its
 *   own, so the contract also pins that plan-family chunks come back.
 * - `AC-405-A` - TWO separate named lanes (the plan requires the two cases
 *   to stay separate so the safety net is never removed by the fix): the
 *   genuinely-empty case (empty result set AND the broaden reported) and the
 *   matches-inside-the-set case (no broaden reported).
 */

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

/** A plan-shaped query whose routed search leaks a non-plan family today. */
const PLAN_SHAPED_QUERY = 'design specification for network activation runtime';

/** A plan-shaped query that matches INSIDE the routed plan family set. */
const PLAN_MATCHING_QUERY = 'plan roadmap decision criteria';

/**
 * A query that matches nothing anywhere in the corpus.
 *
 * The literal is NOT copied into `plans/**`: the plan tree is indexed and the
 * post-write reindex would re-introduce the tokens into the very corpus this
 * fixture claims to miss. Only the token COUNT and the MEASURED zero-match
 * counts are recorded in the plan tracker.
 */
const NOTHING_MATCHING_QUERY =
  'zqe82770df8ac3 zq2e18cdd114f5 zqe0eac33767f7 zq6cfb720d4f1b zq8327ba26b521';

/**
 * Call the existing `search_corpus` MCP tool and return its structured result.
 *
 * @param {Record<string, unknown>} args - The tool arguments.
 * @returns {Promise<Record<string, any>>} The tool structured content.
 */
async function callSearchCorpus(args) {
  const server = createRepoCortexMcpServer({
    databasePath: CORPUS_DATABASE_PATH,
  });
  const response = await server.dispatch({
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
 * Collect the row families that fall OUTSIDE the routed family set.
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
 * Whether the response metadata REPORTS that the family filter was broadened.
 *
 * The broaden is reported through the response metadata; the tool carries
 * `family_fallback` and `classification_fallback` for that purpose, so the
 * lane accepts EITHER of them and pins the REPORTING, not the field name.
 *
 * @param {Record<string, any>} response - The tool structured content.
 * @returns {boolean} True when the metadata reports the broaden.
 */
function broadenReported(response) {
  return (
    response.family_fallback === true ||
    response.classification_fallback === true
  );
}

describe('plan-family-filter.red', () => {
  describe('AC-401-A: the routed family set reaches the corpus search', () => {
    it('returns only rows inside the routed plan family set for a plan-shaped query', async () => {
      const response = await callSearchCorpus({
        query: PLAN_SHAPED_QUERY,
        use_dense: false,
        limit: 10,
      });

      expect(leakedFamilies(response)).toEqual([]);
      expect(response.family_fallback).toBe(false);
    });

    it('reports no classification fallback for a plan-shaped query', async () => {
      const response = await callSearchCorpus({
        query: PLAN_SHAPED_QUERY,
        use_dense: false,
        limit: 10,
      });

      expect(response.classification_fallback).toBe(false);
    });

    it('returns plan-family chunks for a plan-shaped query', async () => {
      const response = await callSearchCorpus({
        query: PLAN_SHAPED_QUERY,
        use_dense: false,
        limit: 10,
      });

      expect(planFamilyRowCount(response)).toBeGreaterThan(0);
    });
  });

  describe('AC-405-A: the broaden-on-empty safety net is distinguished', () => {
    it('reports the broaden fallback for a query that matches nothing anywhere', async () => {
      const response = await callSearchCorpus({
        query: NOTHING_MATCHING_QUERY,
        query_class: 'plan_specific',
        classification_hints: { family: ROUTED_PLAN_FAMILY_CSV },
        use_dense: false,
        limit: 10,
      });

      expect(response.results).toEqual([]);
      expect(broadenReported(response)).toBe(true);
    });

    it('reports no broaden fallback for a query that matches inside the routed plan family set', async () => {
      const response = await callSearchCorpus({
        query: PLAN_MATCHING_QUERY,
        use_dense: false,
        limit: 10,
      });

      expect(broadenReported(response)).toBe(false);
    });
  });
});
