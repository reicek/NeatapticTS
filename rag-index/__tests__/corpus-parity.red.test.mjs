/**
 * @module corpus-parity.red.test
 * @description RED contracts for the corpus parity between the published corpus contract, the
 * doc-entity type map, the per-family freshness rules, the eval fixtures, the README family
 * table, the live index and the docs-generator source of truth (Phase 1, Step 2).
 *
 * Every lane in this file measures an EXISTING surface, so a lane is RED BY ASSERTION and NEVER
 * by an import failure, a missing-file error or an environment error:
 *
 * - the published contract (`rag-index/corpus-sources.mjs`),
 * - the doc-entity type map (`rag-index/extract-doc-entities.mjs`),
 * - the per-family freshness rules (`rag-index/validate-index.mjs`),
 * - the eval fixtures (`rag-index/eval-queries.json`, `rag-index/eval-queries-v2.json`),
 * - the family table of `rag-index/README.md`,
 * - the live index replica (`rag-index/data/turso-replica.sqlite`, opened READ-ONLY), and
 * - the docs-generator constants (`scripts/generate-docs/generate-docs.constants.ts`, read
 *   through the `node:module` createRequire + TypeScript transpile bridge, because this Jest
 *   project runs plain Node ESM WITHOUT a TypeScript transform).
 *
 * The live replica is gitignored, so a fresh clone has NO replica. Every lane therefore falls
 * back to the PUBLISHED-CONTRACT census measured against the live tree, and the invariant under
 * test is the same with and without the replica.
 *
 * RED PHASE: the entity-type map still publishes the retired `benchmark` type and no `test`
 * type, the eval fixtures still expect the retired `readme` family, and the README family table
 * still publishes the retired `readme` and `benchmark` labels and no `test` label.
 */

import { describe, expect, it } from '@jest/globals';
import { existsSync, readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import fg from 'fast-glob';

import {
  CORPUS_FAMILY_NAMES,
  CORPUS_SOURCES,
  DOCS_TARGET_ROOTS,
} from '../corpus-sources.mjs';
import { mapFamilyToEntityType } from '../extract-doc-entities.mjs';
import { defaultDatabasePath, repoRoot } from '../init-schema.mjs';
import { discoverFamilyAssignments, resolveGatedPass } from '../validate-index.mjs';

const require = createRequire(import.meta.url);

// ---------------------------------------------------------------------------
// The contract under test
// ---------------------------------------------------------------------------

/** The families the published corpus contract retires. */
const RETIRED_FAMILIES = ['readme', 'benchmark'];

/** The corpus families the published contract publishes. */
const PUBLISHED_FAMILIES = [...CORPUS_FAMILY_NAMES];

/** The eval fixtures the family expectations are measured against. */
export const MEASURED_EVAL_FIXTURE_FILES = [
  'eval-queries.json',
  'eval-queries-v2.json',
];

/**
 * The module-private per-family rule maps of `rag-index/validate-index.mjs`.
 *
 * The maps are NOT exported, so the lanes read the source text instead of importing them.
 */
export const MEASURED_RULE_MAP_DECLARATIONS = [
  'DEFAULT_FAMILY_GATING',
  'DEFAULT_FAMILY_MAX_SYNC_WAIT_MS',
  'DEFAULT_FAMILY_MAX_AGE_MS',
];

// ---------------------------------------------------------------------------
// The MEASURED current values (read from the live tree on the RED lane)
// ---------------------------------------------------------------------------

/**
 * The `node:sqlite` module, or null when the runtime does not expose it.
 *
 * The module is loaded through `createRequire` so that a runtime without `node:sqlite` degrades
 * to the published-contract census instead of failing the whole file at import time.
 *
 * @returns {object | null} The `node:sqlite` module, or null.
 */
function loadSqliteModule() {
  try {
    return require('node:sqlite');
  } catch {
    return null;
  }
}

const SQLITE_MODULE = loadSqliteModule();

/**
 * Read a UTF-8 file, returning an empty string when the file cannot be read.
 *
 * @param {string} filePath - The absolute file path.
 * @returns {string} The file text.
 */
function readTextFileSync(filePath) {
  try {
    return readFileSync(filePath, 'utf8');
  } catch {
    return '';
  }
}

/**
 * Read the per-family document census from the live replica, READ-ONLY.
 *
 * The replica is opened read-only and the handle is closed in a `finally`, so the lane never
 * writes to the index and never leaves a handle open.
 *
 * @returns {Record<string, number> | null} The census, or null when the replica is absent or
 *   cannot be read.
 */
function readReplicaCensus() {
  if (!existsSync(defaultDatabasePath) || !SQLITE_MODULE) return null;

  let database = null;
  try {
    database = new SQLITE_MODULE.DatabaseSync(defaultDatabasePath, {
      readOnly: true,
    });
    const rows = database
      .prepare(
        'SELECT doc_family, COUNT(*) AS n FROM documents GROUP BY doc_family',
      )
      .all();
    return Object.fromEntries(rows.map((row) => [row.doc_family, row.n]));
  } catch {
    return null;
  } finally {
    database?.close();
  }
}

/**
 * Read the repository-relative paths the live replica labels with the `test` family.
 *
 * @returns {string[] | null} The labelled paths, or null when the replica is absent or cannot
 *   be read.
 */
function readReplicaTestFamilyPaths() {
  if (!existsSync(defaultDatabasePath) || !SQLITE_MODULE) return null;

  let database = null;
  try {
    database = new SQLITE_MODULE.DatabaseSync(defaultDatabasePath, {
      readOnly: true,
    });
    const rows = database
      .prepare('SELECT file_path FROM documents WHERE doc_family = ?')
      .all('test');
    return rows.map((row) => row.file_path);
  } catch {
    return null;
  } finally {
    database?.close();
  }
}

/**
 * Measure the published-contract census by globbing the live tree against `CORPUS_SOURCES`.
 *
 * This is the same census `discoverFamilyAssignments()` measures, computed synchronously so the
 * module-level measurement needs no top-level await.
 *
 * @returns {Record<string, number>} The family document counts.
 */
function measurePublishedContractCensus() {
  const census = {};
  for (const source of CORPUS_SOURCES) {
    const paths = fg.sync(source.patterns, {
      cwd: repoRoot,
      onlyFiles: true,
      ignore: source.ignore ?? [],
    });
    census[source.family] = paths.length;
  }
  return census;
}

/**
 * Measure the published-contract census through the exported assignment seam.
 *
 * @returns {Promise<Record<string, number>>} The family document counts.
 */
async function measurePublishedContractAssignments() {
  const assignments = await discoverFamilyAssignments();
  const census = {};
  for (const family of assignments.values()) {
    census[family] = (census[family] ?? 0) + 1;
  }
  return census;
}

const REPLICA_CENSUS = readReplicaCensus();

/**
 * The census the live index publishes today, measured by this lane.
 *
 * Measured from the live replica when the replica file exists, and from the published contract
 * when it does not (the replica is gitignored, so a fresh clone has none).
 *
 * @type {Record<string, number>}
 */
export const MEASURED_LIVE_CENSUS =
  REPLICA_CENSUS ?? measurePublishedContractCensus();

/**
 * Which surface the measured census was taken from.
 *
 * @type {'live-replica' | 'published-contract'}
 */
export const MEASURED_LIVE_CENSUS_SOURCE = REPLICA_CENSUS
  ? 'live-replica'
  : 'published-contract';

// ---------------------------------------------------------------------------
// The source text of the module-private per-family rule maps
// ---------------------------------------------------------------------------

const VALIDATE_INDEX_SOURCE = readTextFileSync(
  path.join(repoRoot, 'rag-index', 'validate-index.mjs'),
);

/**
 * Split an object-literal body into its top-level entries.
 *
 * @param {string} body - The text between the braces of the literal.
 * @returns {string[]} The trimmed, non-empty entries.
 */
function splitTopLevelEntries(body) {
  const entries = [];
  let current = '';
  let depth = 0;

  for (const character of body) {
    if (character === '{' || character === '[' || character === '(') depth += 1;
    else if (character === '}' || character === ']' || character === ')') depth -= 1;

    if (character === ',' && depth === 0) {
      entries.push(current);
      current = '';
      continue;
    }
    current += character;
  }
  entries.push(current);

  return entries.map((entry) => entry.trim()).filter((entry) => entry.length > 0);
}

/**
 * Extract the top-level keys of a `const <name> = { ... };` object literal.
 *
 * @param {string} source - The source text to search.
 * @param {string} declarationName - The declaration to extract.
 * @returns {string[] | null} The key list, or null when the declaration is absent.
 */
function extractObjectLiteralKeys(source, declarationName) {
  const marker = `const ${declarationName} = {`;
  const declarationIndex = source.indexOf(marker);
  if (declarationIndex < 0) return null;

  const bodyStart = declarationIndex + marker.length;
  let depth = 1;
  let bodyEnd = -1;
  for (let index = bodyStart; index < source.length; index += 1) {
    const character = source[index];
    if (character === '{') depth += 1;
    else if (character === '}') {
      depth -= 1;
      if (depth === 0) {
        bodyEnd = index;
        break;
      }
    }
  }
  if (bodyEnd < 0) return null;

  const keys = [];
  for (const entry of splitTopLevelEntries(source.slice(bodyStart, bodyEnd))) {
    const match = entry.match(
      /^(?:'([^']*)'|"([^"]*)"|([A-Za-z_$][\w$]*))\s*:/,
    );
    if (!match) return null;
    keys.push(match[1] ?? match[2] ?? match[3]);
  }
  return keys;
}

/**
 * Read the top-level keys of a module-private per-family rule map.
 *
 * The rule maps are module-private and therefore not importable, so the source text is read
 * instead; the plan explicitly allows measuring against the repository files. A missing
 * declaration FAILS the lane - it never skips it.
 *
 * @param {string} declarationName - The declaration to read.
 * @returns {string[]} The key list.
 */
function readRuleMapKeys(declarationName) {
  const keys = extractObjectLiteralKeys(VALIDATE_INDEX_SOURCE, declarationName);
  if (keys === null) {
    throw new Error(
      `Declaration ${declarationName} is absent from rag-index/validate-index.mjs`,
    );
  }
  return keys;
}

// ---------------------------------------------------------------------------
// The eval-fixture family expectations
// ---------------------------------------------------------------------------

/**
 * Collect the family names an eval fixture expects.
 *
 * The families are read ONLY from `expected_doc_families` and from `relevance_grades[].family`,
 * so prose in `query` or `notes` can never contribute a family name.
 *
 * @param {object[]} queries - The parsed fixture queries.
 * @returns {string[]} The expected family names.
 */
function collectFixtureFamilyNames(queries) {
  const families = [];
  for (const query of queries) {
    for (const family of query.expected_doc_families ?? []) {
      if (typeof family === 'string') families.push(family);
    }
    for (const grade of query.relevance_grades ?? []) {
      if (typeof grade?.family === 'string') families.push(grade.family);
    }
  }
  return families;
}

// ---------------------------------------------------------------------------
// The README family table
// ---------------------------------------------------------------------------

/**
 * Parse the family labels of the `## Corpus families` table of `rag-index/README.md`.
 *
 * Only the table counts: the second column of every data row of the table under the heading.
 *
 * @param {string} readmeText - The README text.
 * @returns {string[]} The family labels, backticks stripped.
 */
function parseCorpusFamilyTableLabels(readmeText) {
  const lines = readmeText.split(/\r?\n/);
  const headingIndex = lines.findIndex((line) =>
    /^##\s+Corpus families\s*$/.test(line),
  );
  if (headingIndex < 0) return [];

  let cursor = headingIndex + 1;
  while (cursor < lines.length && !lines[cursor].trim().startsWith('|')) cursor += 1;

  const labels = [];
  for (
    ;
    cursor < lines.length && lines[cursor].trim().startsWith('|');
    cursor += 1
  ) {
    const cells = lines[cursor]
      .split('|')
      .slice(1, -1)
      .map((cell) => cell.trim());
    if (cells.length < 2) continue;
    // The header row and the separator row carry no priority number.
    if (!/^\d+$/.test(cells[0])) continue;
    labels.push(cells[1].replace(/`/g, ''));
  }
  return labels;
}

// ---------------------------------------------------------------------------
// The docs-generator source of truth
// ---------------------------------------------------------------------------

const DOCS_CONSTANTS_PATH = path.join(
  repoRoot,
  'scripts',
  'generate-docs',
  'generate-docs.constants.ts',
);

/**
 * Read the docs-generator `DOCS_TARGETS` constant through the createRequire bridge.
 *
 * The generator constants are TypeScript and this Jest project runs plain Node ESM WITHOUT a
 * TypeScript transform, so the file cannot be imported directly. The bridge transpiles the
 * source text to CommonJS and evaluates it with the same `require` the module would have used.
 *
 * @returns {Record<string, object>} The `DOCS_TARGETS` constant of the generator.
 */
function readDocsGeneratorTargets() {
  const ts = require('typescript');
  const source = readTextFileSync(DOCS_CONSTANTS_PATH);
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
  });

  const moduleShim = { exports: {} };
  new Function(
    'exports',
    'require',
    'module',
    '__filename',
    '__dirname',
    outputText,
  )(
    moduleShim.exports,
    require,
    moduleShim,
    DOCS_CONSTANTS_PATH,
    path.dirname(DOCS_CONSTANTS_PATH),
  );

  const docsTargets = moduleShim.exports.DOCS_TARGETS;
  if (!docsTargets) {
    throw new Error(
      'DOCS_TARGETS is absent from scripts/generate-docs/generate-docs.constants.ts',
    );
  }
  return docsTargets;
}

// ---------------------------------------------------------------------------
// AC-204-A - the residue is gone
// ---------------------------------------------------------------------------

describe('AC-204-A the residue is gone', () => {
  it('publishes no retired family in the corpus contract', () => {
    // Arrange / Act: the published family names of the corpus contract.
    const retiredFamiliesInContract = PUBLISHED_FAMILIES.filter((family) =>
      RETIRED_FAMILIES.includes(family),
    );

    // Assert.
    expect(retiredFamiliesInContract).toEqual([]);
  });

  it('publishes no retired family in the entity-type map', () => {
    // Arrange / Act: the entity type each retired family maps to.
    const retiredEntityTypes = {
      readme: mapFamilyToEntityType('readme'),
      benchmark: mapFamilyToEntityType('benchmark'),
    };

    // Assert.
    expect(retiredEntityTypes).toEqual({ readme: null, benchmark: null });
  });

  it('publishes a member for the test family in the entity-type map', () => {
    // Arrange / Act: the entity type the test family maps to.
    const testEntityType = mapFamilyToEntityType('test');

    // Assert.
    expect(testEntityType).not.toBeNull();
  });

  it('names no retired family in the per-family freshness rules', () => {
    // Arrange / Act: the family keys of the three module-private rule maps.
    const ruleFamilyKeys = MEASURED_RULE_MAP_DECLARATIONS.flatMap(
      (declarationName) => readRuleMapKeys(declarationName),
    );
    const retiredFamiliesInRules = ruleFamilyKeys.filter((family) =>
      RETIRED_FAMILIES.includes(family),
    );

    // Assert.
    expect(retiredFamiliesInRules).toEqual([]);
  });

  it('carries the test family in the per-family freshness rules', () => {
    // Arrange / Act: whether each rule map carries the test family.
    const testFamilyCoverage = Object.fromEntries(
      MEASURED_RULE_MAP_DECLARATIONS.map((declarationName) => [
        declarationName,
        readRuleMapKeys(declarationName).includes('test'),
      ]),
    );

    // Assert.
    expect(testFamilyCoverage).toEqual({
      DEFAULT_FAMILY_GATING: true,
      DEFAULT_FAMILY_MAX_SYNC_WAIT_MS: true,
      DEFAULT_FAMILY_MAX_AGE_MS: true,
    });
  });

  it('gates the test family and gates no retired family', () => {
    // Arrange / Act: the gated pass of a stale family for each family of interest.
    const gatedPass = {
      test: resolveGatedPass({ test: { fresh: false } }),
      readme: resolveGatedPass({ readme: { fresh: false } }),
      benchmark: resolveGatedPass({ benchmark: { fresh: false } }),
    };

    // Assert.
    expect(gatedPass).toEqual({ test: false, readme: true, benchmark: true });
  });

  it('names no retired family in either eval fixture', async () => {
    // Arrange: both eval fixtures, parsed.
    const retiredFamiliesInFixtures = [];
    for (const fixtureFile of MEASURED_EVAL_FIXTURE_FILES) {
      const queries = JSON.parse(
        await readFile(path.join(repoRoot, 'rag-index', fixtureFile), 'utf8'),
      );

      // Act: the family names the fixture expects.
      for (const family of collectFixtureFamilyNames(queries)) {
        if (RETIRED_FAMILIES.includes(family)) {
          retiredFamiliesInFixtures.push(`${fixtureFile}#${family}`);
        }
      }
    }

    // Assert.
    expect([...new Set(retiredFamiliesInFixtures)].toSorted()).toEqual([]);
  });

  it('describes the published family set in the rag-index README family table', async () => {
    // Arrange: the README text.
    const readmeText = await readFile(
      path.join(repoRoot, 'rag-index', 'README.md'),
      'utf8',
    );

    // Act: the family labels of the `## Corpus families` table.
    const tableFamilies = [
      ...new Set(parseCorpusFamilyTableLabels(readmeText)),
    ].toSorted();

    // Assert.
    expect(tableFamilies).toEqual(PUBLISHED_FAMILIES.toSorted());
  });
});

// ---------------------------------------------------------------------------
// AC-205-A - the index is purged
// ---------------------------------------------------------------------------

describe('AC-205-A the index is purged', () => {
  it('owns zero documents in the retired families in the live index', async () => {
    // Arrange / Act: the census of the live index, read READ-ONLY from the replica when the
    // replica exists and from the published contract when it does not.
    const census =
      readReplicaCensus() ?? (await measurePublishedContractAssignments());
    const retiredFamiliesWithDocuments = RETIRED_FAMILIES.filter(
      (family) => (census[family] ?? 0) > 0,
    );

    // Assert.
    expect(retiredFamiliesWithDocuments).toEqual([]);
  });

  it('gives the test family exactly the reclassified test documents', async () => {
    // Arrange: the paths the live index labels with the test family.
    const labelledTestPaths =
      readReplicaTestFamilyPaths() ??
      [...(await discoverFamilyAssignments()).entries()]
        .filter(([, family]) => family === 'test')
        .map(([filePath]) => filePath);

    // Act: the test documents on disk.
    const onDiskTestPaths = await fg(
      ['examples/**/*.test.ts', 'benchmarks/**/*.test.ts'],
      { cwd: repoRoot, onlyFiles: true },
    );

    // Assert.
    expect(labelledTestPaths.toSorted()).toEqual(onDiskTestPaths.toSorted());
  });

  it('records the measured census as the baseline the green lane compares against', () => {
    // Arrange / Act: the measured census of the live index.
    const census = MEASURED_LIVE_CENSUS;
    const retiredFamiliesInCensus = Object.keys(census).filter((family) =>
      RETIRED_FAMILIES.includes(family),
    );

    // Assert.
    expect({
      hasTestFamily: typeof census.test === 'number',
      retiredFamiliesInCensus,
    }).toEqual({ hasTestFamily: true, retiredFamiliesInCensus: [] });
  });
});

// ---------------------------------------------------------------------------
// AC-202-B - the generator-to-contract link
// ---------------------------------------------------------------------------

describe('AC-202-B the generator-to-contract link', () => {
  it('publishes the same six docs-generator targets in the same declaration order', () => {
    // Arrange: the docs-generator targets, read through the transpile bridge.
    const generatorTargets = readDocsGeneratorTargets();

    // Act: the target names in declaration order.
    const generatorTargetNames = Object.values(generatorTargets).map(
      (target) => target.name,
    );

    // Assert.
    expect(generatorTargetNames).toEqual(
      DOCS_TARGET_ROOTS.map((target) => target.name),
    );
  });

  it('publishes the same repo-relative roots and the same excludeRootSourceFiles flags', () => {
    // Arrange: the docs-generator targets, read through the transpile bridge.
    const generatorTargets = readDocsGeneratorTargets();

    // Act: the repo-relative root and the exclusion flag of each target. The root is computed
    // relative to the working directory, and `path.resolve('src')` and `path.resolve('.')`
    // share that working directory, so the comparison is cwd-independent.
    const publishedTargets = Object.values(generatorTargets).map((target) => ({
      root: path
        .relative(path.resolve('.'), target.sourceDir)
        .split(path.sep)
        .join('/'),
      excludeRootSourceFiles: target.excludeRootSourceFiles ?? false,
    }));

    // Assert.
    expect(publishedTargets).toEqual(
      DOCS_TARGET_ROOTS.map((target) => ({
        root: target.root,
        excludeRootSourceFiles: target.excludeRootSourceFiles,
      })),
    );
  });
});
