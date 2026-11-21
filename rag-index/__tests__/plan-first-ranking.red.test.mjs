/**
 * @module plan-first-ranking.red.test
 * @description RED contract for the plan-first family priority and the retired generated-README
 * suppression (Phase 2, Step 5 - `AC-402-A` and `AC-406-A`).
 *
 * THE CONTRACT, IN TWO PARTS.
 *
 * (1) `AC-402-A` - the PUBLISHED family priority of `rag-index/assemble-context.mjs`
 *     (`DEFAULT_FAMILY_PRIORITY`, published at `:53-60`) must place the PLAN families AT THE HEAD of the
 *     list and AHEAD of the SOURCE family, and the ORDERING SEAM (`orderChunks`, `:394`) must HONOUR that
 *     priority: at an equal score a plan-family chunk sorts above a source-family chunk. The ordering stays
 *     a published priority list - this home never pins the literal list, so the implementing lane
 *     (`P2-S5-2`) owns the list and this home only measures the RANKING it produces.
 *
 * (2) `AC-406-A` - the advanced-search path must carry NO suppression for a family the corpus can
 *     no longer publish. The corpus publishes NINE families (`CORPUS_FAMILY_NAMES`, published by
 *     `rag-index/corpus-sources.mjs:202`); the retired `readme` and `generated-readme` families are
 *     NOT among them, so the `include_code_only` filter of the advanced-search module (the two
 *     constants at `:84-85`, the filter at `:857-863`) suppresses families that can never appear - a
 *     DEAD branch that the corpus change made dead.
 *
 * THE MEASURED BASELINE (measured on the working tree and recorded here so the GREEN lane compares
 * against a MEASURED baseline - it is NOT re-derived by this home and NOT pinned as an assertion):
 *
 * - the published priority is `ts-source`, `readme`, `plan`, `agent`, `skill`, `demo` - so the source
 *   family is FIRST, the retired `readme` family is SECOND and the plan families are THIRD;
 * - `completed-plan` is ABSENT from the published priority, so its production ordinal is
 *   `Number.MAX_SAFE_INTEGER`: a family absent from the list sorts LAST, it does NOT sort first - a
 *   raw `indexOf` would return `-1` and make the comparison vacuously green, so this home mirrors the
 *   production ordinal semantics of `familyPriorityOrdinal` (`rag-index/assemble-context.mjs:375`)
 *   in a small documented local helper instead of re-typing the rule differently;
 * - `orderChunks` on a plan-family chunk and a source-family chunk at an EQUAL score and DIFFERENT
 *   file paths returns the SOURCE chunk FIRST (the family delta is a TIE-BREAK after the tier delta
 *   and the file-max delta, and it outranks the file-path fallback);
 * - the advanced-search module still declares `README_FAMILY = 'readme'` and
 *   `GENERATED_README_FAMILY = 'generated-readme'` and filters on both.
 *
 * WHY THE FILE SCAN IS HERMETIC. `scripts/mcp-semantic/tools/search-advanced.mjs` is NEVER IMPORTED by
 * this home. The `rag-index-mjs` jest project has NO setup file, and the `mcp-semantic-mjs` setup file
 * (`testing/jest-setup-mcp-semantic.mjs`) exists precisely to force the dense and reranker states to `cold`
 * "without loading onnxruntime-node native bindings, which can crash during Jest teardown". Importing the
 * advanced-search module into a project WITHOUT that guard would carry that ONNX native-binding risk into this
 * home, and the slice forbids a lane that fails by IMPORT FAILURE. The module is therefore read as TEXT through
 * `node:fs`. THE SCAN IS ALSO THE STRONGER PROBE FOR `AC-406-A`: because the corpus can no longer publish the
 * retired families, the suppression filters NOTHING out, so a behavioural assertion over search results would be
 * VACUOUS - byte-identical before and after the fix - while the source-level scan measures the suppression itself.
 *
 * THE ANTI-VACUITY LANE. The invariants in the last describe block are GREEN today and MUST stay GREEN
 * after the fix. They deliberately do NOT pin the current literal priority list (the new list is the
 * implementing lane's choice) and do NOT pin the current family order (that IS the red contract above).
 */

import { readFileSync } from 'node:fs';
import { URL, fileURLToPath } from 'node:url';

import { describe, expect, it } from '@jest/globals';

import { DEFAULT_FAMILY_PRIORITY, orderChunks } from '../assemble-context.mjs';
import { CORPUS_FAMILY_NAMES } from '../corpus-sources.mjs';
import { ROUTING } from '../routing-table.mjs';

// ---------------------------------------------------------------------------
// The families the contract ranks (READ from the routing table, never re-typed)
// ---------------------------------------------------------------------------

/** The SOURCE family: the family the plan families must rank AHEAD of. */
const SOURCE_FAMILY = ROUTING.code_specific.family;

/** The PLAN families the published priority must place at the head, in routing order. */
const PLAN_FAMILIES = ROUTING.plan_specific.family
  .split(',')
  .map((family) => family.trim())
  .filter((family) => family.length > 0);

/** The retired README family names the corpus can no longer publish. */
const RETIRED_README_FAMILIES = ['readme', 'generated-readme'];

// ---------------------------------------------------------------------------
// The advanced-search module (read as TEXT - NEVER imported)
// ---------------------------------------------------------------------------

/** The absolute path of the advanced-search module, resolved from THIS file's URL. */
const SEARCH_ADVANCED_PATH = fileURLToPath(
  new URL('../../scripts/mcp-semantic/tools/search-advanced.mjs', import.meta.url),
);

/** The RAW source of the advanced-search module. */
const SEARCH_ADVANCED_SOURCE = readFileSync(SEARCH_ADVANCED_PATH, 'utf8');

// ---------------------------------------------------------------------------
// The measurement helpers
// ---------------------------------------------------------------------------

/**
 * The production ordinal semantics of `familyPriorityOrdinal` (`rag-index/assemble-context.mjs:375`):
 * a family ABSENT from the priority list sorts LAST, not first.
 *
 * The helper MIRRORS the production rule instead of re-typing it, so the `completed-plan` comparison is a
 * genuine measurement rather than a vacuously green `indexOf` (which returns `-1` for an absent family and would
 * make the absent family look like the FIRST one).
 *
 * @param {string} family - The family name.
 * @param {string[]} [priorityList] - The published priority list.
 * @returns {number} The ordinal; `Number.MAX_SAFE_INTEGER` for a family absent from the list.
 */
function priorityOrdinalOf(family, priorityList = DEFAULT_FAMILY_PRIORITY) {
  const index = priorityList.indexOf(family);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

/**
 * Strip the comments from a JavaScript source.
 *
 * A COMMENT that mentions a retired family is NOT a live reference to that family, so the comments are removed
 * before the literal scan. The line-comment pattern skips a `//` that follows a colon, so a URL inside a string
 * literal is never mistaken for a comment.
 *
 * @param {string} source - The JavaScript source.
 * @returns {string} The source without its comments.
 */
function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|[^:])\/\/[^\n]*/gm, '$1');
}

/**
 * The family string literals the advanced-search module uses as SUPPRESSION TARGETS.
 *
 * Two shapes are read: a family CONSTANT (`const README_FAMILY = 'readme';`) and a direct family COMPARISON
 * (`result.family !== 'readme'`). Both shapes name a family the module filters on.
 *
 * @param {string} source - The module source.
 * @returns {string[]} The suppression family literals, sorted and de-duplicated.
 */
function collectSuppressionFamilyLiterals(source) {
  const code = stripComments(source);
  const literals = new Set();
  const patterns = [
    /const\s+\w*FAMILY\w*\s*=\s*'([^']+)'/g,
    /\.family\s*!==?\s*'([^']+)'/g,
  ];
  for (const pattern of patterns) {
    for (const match of code.matchAll(pattern)) {
      literals.add(match[1]);
    }
  }
  return [...literals].toSorted();
}

/**
 * Every quoted string literal of a source (single quotes, double quotes and template literals).
 *
 * @param {string} source - The module source.
 * @returns {string[]} The literals, sorted and de-duplicated.
 */
function collectQuotedLiterals(source) {
  const code = stripComments(source);
  const literals = new Set();
  const patterns = [/'([^'\n]*)'/g, /"([^"\n]*)"/g, /`([^`]*)`/g];
  for (const pattern of patterns) {
    for (const match of code.matchAll(pattern)) {
      literals.add(match[1]);
    }
  }
  return [...literals].toSorted();
}

// ---------------------------------------------------------------------------
// AC-402-A - the published priority ranks the plan families first
// ---------------------------------------------------------------------------

describe('plan-first-ranking: the plan families rank first', () => {
  describe('AC-402-A the published priority ranks the plan families ahead of the source family', () => {
    it('publishes the plan families at the head of the priority list', () => {
      // Arrange / Act: the LEADING entries of the published priority - one entry per routed plan family.
      const head = DEFAULT_FAMILY_PRIORITY.slice(0, PLAN_FAMILIES.length);

      // Assert: the head of the list IS the plan families. Their INTERNAL order is the implementing lane's
      // choice, so the comparison is set-wise and never pins a literal priority array.
      expect(head.toSorted()).toEqual([...PLAN_FAMILIES].toSorted());
    });

    it('ranks the plan family ahead of the source family in the published priority', () => {
      // Arrange / Act: the ordinal of each family in the PUBLISHED priority list.
      const planOrdinal = priorityOrdinalOf(PLAN_FAMILIES[0]);
      const sourceOrdinal = priorityOrdinalOf(SOURCE_FAMILY);

      // Assert: the plan family is ranked ahead of the source family (a LOWER ordinal ranks first).
      expect(planOrdinal).toBeLessThan(sourceOrdinal);
    });

    it('ranks the completed-plan family ahead of the source family in the published priority', () => {
      // Arrange / Act: the completed-plan family is ABSENT from the published priority today, so its ordinal is
      // the LAST one - the comparison below is a genuine measurement, not a vacuously green `-1` index.
      const completedPlanOrdinal = priorityOrdinalOf(PLAN_FAMILIES[1]);
      const sourceOrdinal = priorityOrdinalOf(SOURCE_FAMILY);

      // Assert: the completed-plan family is ranked ahead of the source family (a LOWER ordinal ranks first).
      expect(completedPlanOrdinal).toBeLessThan(sourceOrdinal);
    });

    it('orders a plan-family chunk ahead of a source-family chunk at an equal score', () => {
      // Arrange: two chunks of the SAME score (so the tier delta and the file-max delta are ties) in DIFFERENT
      // files (so the file-path fallback cannot decide), with the SOURCE path sorting FIRST - the file-path
      // fallback therefore CONTRADICTS the expected order, so only the family priority can produce it.
      const planChunk = { chunk_id: 1, file_path: 'z-plan.md', char_start: 0, score: 0.5, family: PLAN_FAMILIES[0] };
      const sourceChunk = { chunk_id: 2, file_path: 'a-source.ts', char_start: 0, score: 0.5, family: SOURCE_FAMILY };

      // Act
      const ordered = orderChunks([planChunk, sourceChunk]);

      // Assert: the plan-family chunk is ordered first, the source-family chunk second.
      expect(ordered.map((chunk) => chunk.family)).toEqual([PLAN_FAMILIES[0], SOURCE_FAMILY]);
    });

    it('publishes no family the corpus can no longer publish in the priority list', () => {
      // Arrange / Act: the families of the published priority that the corpus can no longer publish.
      const unpublishedFamilies = DEFAULT_FAMILY_PRIORITY.filter(
        (family) => !CORPUS_FAMILY_NAMES.includes(family),
      );

      // Assert: the retired families are gone from the published priority.
      expect(unpublishedFamilies).toEqual([]);
    });
  });

  // -------------------------------------------------------------------------
  // AC-406-A - the advanced-search path carries no dead family suppression
  // -------------------------------------------------------------------------

  describe('AC-406-A the advanced search path carries no dead family suppression', () => {
    it('suppresses no family the corpus can no longer publish', () => {
      // Arrange / Act: the suppression targets of the module that the corpus cannot publish.
      const unpublishedSuppressionTargets = collectSuppressionFamilyLiterals(SEARCH_ADVANCED_SOURCE).filter(
        (family) => !CORPUS_FAMILY_NAMES.includes(family),
      );

      // Assert: every family the module suppresses is a family the corpus can still publish.
      expect(unpublishedSuppressionTargets).toEqual([]);
    });

    it('carries no reference to the retired README families', () => {
      // Arrange / Act: the quoted string literals of the module that name a retired README family.
      const retiredFamilyReferences = collectQuotedLiterals(SEARCH_ADVANCED_SOURCE).filter((literal) =>
        RETIRED_README_FAMILIES.includes(literal),
      );

      // Assert: the retired README families are not named anywhere in the module.
      expect(retiredFamilyReferences).toEqual([]);
    });
  });

  // -------------------------------------------------------------------------
  // The anti-vacuity lanes (GREEN today and GREEN after the fix)
  // -------------------------------------------------------------------------

  describe('the anti-vacuity lanes', () => {
    it('publishes a non-empty priority list without duplicate families', () => {
      // Arrange / Act: the families of the published priority that appear more than once.
      const duplicateFamilies = DEFAULT_FAMILY_PRIORITY.filter(
        (family, index) => DEFAULT_FAMILY_PRIORITY.indexOf(family) !== index,
      );

      // Assert: the published priority is non-empty and lists every family once.
      expect({
        nonEmpty: DEFAULT_FAMILY_PRIORITY.length > 0,
        duplicateFamilies,
      }).toEqual({ nonEmpty: true, duplicateFamilies: [] });
    });

    it('keeps the source family published by the corpus contract', () => {
      // Assert: the corpus still publishes the source family the plan families rank ahead of.
      expect(CORPUS_FAMILY_NAMES).toContain(SOURCE_FAMILY);
    });

    it('publishes every routed plan family in the corpus contract', () => {
      // Arrange / Act: the routed plan families the corpus publishes.
      const publishedPlanFamilies = CORPUS_FAMILY_NAMES.filter((family) =>
        PLAN_FAMILIES.includes(family),
      );

      // Assert: every routed plan family is a published corpus family.
      expect(publishedPlanFamilies.toSorted()).toEqual([...PLAN_FAMILIES].toSorted());
    });

    it('still tie-breaks within one file by character position', () => {
      // Arrange: two chunks of the SAME file and the SAME score, so the family delta is a tie and the character
      // position is the only remaining tie-break.
      const laterChunk = { chunk_id: 1, file_path: 'a-source.ts', char_start: 50, score: 0.5, family: SOURCE_FAMILY };
      const earlierChunk = { chunk_id: 2, file_path: 'a-source.ts', char_start: 10, score: 0.5, family: SOURCE_FAMILY };

      // Act
      const ordered = orderChunks([laterChunk, earlierChunk]);

      // Assert: the earlier character position is ordered first.
      expect(ordered.map((chunk) => chunk.char_start)).toEqual([10, 50]);
    });
  });
});
