/**
 * @module corpus-contract.red.test
 * @description RED contracts for the published corpus-family contract (Phase 1, Step 1).
 *
 * Every lane in this file probes the EXPORTED `discoverFamilyAssignments` seam of
 * `rag-index/validate-index.mjs` - an EXISTING module - so the lane is RED BY ASSERTION and
 * NEVER by an import failure. No lane imports a module that does not exist yet.
 *
 * The MEASURED CURRENT VALUES block records what the live corpus publishes TODAY, so the green
 * lane compares against a measurement instead of a hand-written expectation. Those values were
 * read from the live tree with the same seam this file probes:
 *
 *   - the ten-family census over 2080 assigned paths
 *   - 71 assigned example READMEs (58 generated + 13 handwritten)
 *   - the 20.9 MB session snapshot assigned to `demo`
 *   - 339 test documents split across `demo` (317) and `benchmark` (22)
 *
 * RED PHASE: the corpus still publishes ten families; the target contract publishes nine.
 */

import { describe, expect, it } from '@jest/globals';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fg from 'fast-glob';

import { discoverFamilyAssignments } from '../validate-index.mjs';
import { repoRoot } from '../init-schema.mjs';

const moduleDirectory = path.dirname(fileURLToPath(import.meta.url));
const BASELINE_FIXTURE_PATH = path.join(
  moduleDirectory,
  'fixtures',
  'build-memory-baseline.json',
);
const BASELINE_SCHEMA = 'rag-index/build-memory-baseline/v1';

// ---------------------------------------------------------------------------
// The TARGET contract (Phase 1, Step 1)
// ---------------------------------------------------------------------------

/** The nine families the published corpus contract must publish. */
const TARGET_FAMILIES = [
  'agent',
  'completed-plan',
  'copilot-instructions',
  'demo',
  'plan',
  'root-doc',
  'skill',
  'test',
  'ts-source',
];

/** The CLOSED allow-list of handwritten example READMEs. */
const HANDWRITTEN_EXAMPLE_READMES = [
  'examples/README.md',
  'examples/asciiMaze/README.md',
  'examples/evolveXor/README.md',
  'examples/flappy_bird/README.md',
  'examples/helloNetwork/README.md',
  'examples/neatChat/README.md',
  'examples/neatChat/memory/README.md',
  'examples/neatRacing/README.md',
  'examples/neatenstein/README.md',
  'examples/neatenstein/browser-entry/README.md',
  'examples/racing_curriculum/README.md',
  'examples/sequenceReset/README.md',
  'examples/shared/semantic/README.md',
];

/** The generated runtime artifact that must leave the corpus. */
const SESSION_SNAPSHOT_PATH =
  'examples/neatChat/default-pretrained-session-snapshot.ts';

// ---------------------------------------------------------------------------
// The MEASURED current values (read from the live tree on the RED lane)
// ---------------------------------------------------------------------------

/**
 * The census the live corpus publishes today. Recorded so the green lane compares against a
 * measurement. These are NOT assertions about the target contract - the target contract is
 * asserted above.
 */
export const MEASURED_CURRENT_VALUES = {
  familyCount: 10,
  families: [
    'agent',
    'benchmark',
    'completed-plan',
    'copilot-instructions',
    'demo',
    'plan',
    'readme',
    'root-doc',
    'skill',
    'ts-source',
  ],
  assignedPathCount: 2080,
  exampleReadmeCount: 71,
  snapshotFamily: 'demo',
  testDocumentCount: 339,
  testDocumentFamilies: { demo: 317, benchmark: 22 },
  planDocumentCount: 219,
  planLogTrackerCount: 97,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Read the corpus census the memory baseline was captured against.
 *
 * @returns {Promise<object>} The parsed baseline fixture.
 */
async function readBaselineFixture() {
  return JSON.parse(await readFile(BASELINE_FIXTURE_PATH, 'utf8'));
}

/**
 * Glob the repository for the file paths matching the given patterns.
 *
 * The glob options mirror the seam under test (`discoverFamilyPaths` in `validate-index.mjs`)
 * exactly, so the paths compared here are like-for-like with the published assignments.
 *
 * @param {string[]} patterns - fast-glob patterns, relative to the repository root.
 * @returns {Promise<string[]>} The matching repository-relative paths.
 */
async function globRepoPaths(patterns) {
  return fg(patterns, { cwd: repoRoot, onlyFiles: true });
}

// ---------------------------------------------------------------------------
// AC-201-A - the published family set
// ---------------------------------------------------------------------------

describe('AC-201-A the published corpus family set', () => {
  it('publishes exactly the nine target corpus families', async () => {
    const assignments = await discoverFamilyAssignments();
    const publishedFamilies = [...new Set(assignments.values())].toSorted();
    const surplusFamilies = publishedFamilies.filter(
      (family) => !TARGET_FAMILIES.includes(family),
    );
    const missingFamilies = TARGET_FAMILIES.filter(
      (family) => !publishedFamilies.includes(family),
    );

    expect({ surplusFamilies, missingFamilies }).toEqual({
      surplusFamilies: [],
      missingFamilies: [],
    });
  });
});

// ---------------------------------------------------------------------------
// AC-202-A - the exclusions and the preservation
// ---------------------------------------------------------------------------

describe('AC-202-A the corpus exclusions and the preservation', () => {
  it('assigns no family to the generated session snapshot', async () => {
    const assignments = await discoverFamilyAssignments();

    expect(assignments.has(SESSION_SNAPSHOT_PATH)).toBe(false);
  });

  it('assigns exactly the 13 handwritten example READMEs to the demo family', async () => {
    const assignments = await discoverFamilyAssignments();
    const assignedExampleReadmes = [...assignments.keys()]
      .filter(
        (filePath) =>
          filePath.startsWith('examples/') && filePath.endsWith('/README.md'),
      )
      .toSorted();
    const assignedExampleReadmeFamilies = new Set(
      assignedExampleReadmes.map((filePath) => assignments.get(filePath)),
    );

    expect(assignedExampleReadmes).toEqual(
      HANDWRITTEN_EXAMPLE_READMES.toSorted(),
    );
    expect(assignedExampleReadmeFamilies).toEqual(new Set(['demo']));
  });

  it('assigns every examples and benchmarks test document to the test family', async () => {
    const assignments = await discoverFamilyAssignments();
    const testDocumentPaths = await globRepoPaths([
      'examples/**/*.test.ts',
      'benchmarks/**/*.test.ts',
    ]);
    const misassignedTestDocuments = testDocumentPaths
      .filter((filePath) => assignments.get(filePath) !== 'test')
      .toSorted();
    const assignedTestDocuments = testDocumentPaths.filter((filePath) =>
      assignments.has(filePath),
    );

    expect(misassignedTestDocuments).toEqual([]);
    expect(assignedTestDocuments.length).toBe(testDocumentPaths.length);
  });

  it('keeps every plan tracker in the plan or completed-plan family without introducing a log family', async () => {
    const assignments = await discoverFamilyAssignments();
    const planDocumentPaths = await globRepoPaths(['plans/**/*.md']);
    const planDocumentFamilies = new Set(
      planDocumentPaths.map((filePath) => assignments.get(filePath)),
    );
    const logTrackerPaths = planDocumentPaths.filter((filePath) =>
      filePath.endsWith('.logs.md'),
    );
    const logTrackerFamilies = new Set(
      logTrackerPaths.map((filePath) => assignments.get(filePath)),
    );
    const publishedFamilies = [...new Set(assignments.values())].toSorted();
    const familiesBeyondTheContract = publishedFamilies.filter(
      (family) => !TARGET_FAMILIES.includes(family),
    );

    expect(planDocumentFamilies).toEqual(new Set(['plan', 'completed-plan']));
    expect(logTrackerFamilies).toEqual(new Set(['plan', 'completed-plan']));
    expect(familiesBeyondTheContract).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// AC-203-A - the pre-change build memory baseline
// ---------------------------------------------------------------------------

describe('AC-203-A the pre-change build memory baseline', () => {
  it('records the four measured build numbers in the baseline fixture', async () => {
    const baseline = await readBaselineFixture();
    const measurements = baseline.measurements;

    expect({
      peakRssRecorded: measurements.peakRssBytes > 0,
      replicaRecorded: measurements.replicaBytes > 0,
      documentsRecorded: measurements.documentCount > 0,
      chunksRecorded: measurements.chunkCount > 0,
    }).toEqual({
      peakRssRecorded: true,
      replicaRecorded: true,
      documentsRecorded: true,
      chunksRecorded: true,
    });
  });

  it('records the corpus identity that produced the baseline', async () => {
    const baseline = await readBaselineFixture();
    const censusTotal = baseline.corpus.families.reduce(
      (total, family) => total + family.documentCount,
      0,
    );

    expect(baseline.schema).toBe(BASELINE_SCHEMA);
    expect(baseline.corpus.identity).toMatch(/^[0-9a-f]{64}$/);
    expect(censusTotal).toBe(baseline.measurements.documentCount);
  });

  it('captured the baseline against the pre-change corpus without touching the live index', async () => {
    const baseline = await readBaselineFixture();
    const baselineFamilies = baseline.corpus.families
      .map((family) => family.family)
      .toSorted();

    expect(baselineFamilies).toEqual(MEASURED_CURRENT_VALUES.families);
    expect(baseline.build.liveIndexUntouched).toBe(true);
  });
});
