/**
 * @module rag-update-memory.red.test
 * @description RED contracts for the bounded-memory `rag:update` pipeline proof
 * (Phase 1, Step 3, slice `P1-S3-1`).
 *
 * The file mixes THREE kinds of lane, and the mix is deliberate:
 *
 *   - `AC-203-B` - RED TODAY. The `rag:update` stage report carries the stage name, the status
 *     and the elapsed time ONLY, so NO stage publishes a peak-memory figure. The lane measures
 *     the EXISTING `updateRag` seam of `rag-index/update-rag.mjs` and pins the future figure
 *     against the RECORDED pre-change baseline read from the frozen fixture.
 *   - `AC-206-A` - GREEN TODAY (pinning lanes). The stage order, the per-stage success, the
 *     published family set with every family fresh and the measured per-stage elapsed time are
 *     the chain contract that the peak publication must not break.
 *   - `AC-203-E-RED` - RED TODAY. The corpus identity is not published by the module yet, so the
 *     lane reads the identity composition OFF THE MODULE NAMESPACE through a DYNAMIC
 *     `await import()` of an EXISTING module and fails BY ASSERTION.
 *
 * SEAMS (all of them EXIST today - no lane imports a module or a symbol that does not exist):
 *   - `updateRag` from `rag-index/update-rag.mjs` (injected runner / hash store).
 *   - `buildFamilyFresh` and `resolveGatedPass` from `rag-index/validate-index.mjs`.
 *   - `CORPUS_FAMILY_NAMES` from `rag-index/corpus-sources.mjs`.
 *   - `isFreshDocument` from `rag-index/freshness.mjs`.
 *
 * FIXTURE (READ ONLY - this file never writes it):
 *   `rag-index/__tests__/fixtures/build-memory-baseline.json`
 *   READ PATH:
 *   `path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures', 'build-memory-baseline.json')`
 *   The fixture-recorded baseline figure is read at RUNTIME through the fallback chain
 *   `multiSample.postRemediation.primary.peakRssBytes.median` ->
 *   `multiSample.postRemediation.pooled.peakRssBytes.median` ->
 *   `multiSample.postRemediation.pooled.median` -> `measurements.peakRssBytes`.
 *   The fixture's operative arm is `multiSample.postRemediation`, so the figure this file pins
 *   to is that arm's primary median (563331072 bytes). THE FROZEN CEILING (585711616 bytes) is
 *   NOT asserted anywhere in this file - the ceiling is the GREEN lane `P1-S3-3` obligation.
 *
 * LANE DEVIATION (AC-206-A, "the validate stage reports the published family set"):
 *   The "validate stage reports the published family set" channel has NO owning implementing
 *   slice (`P1-S3-2` owns only the peak publication and the corpus identity) and `updateRag`
 *   does not surface a SUCCESSFUL stage's stdout, so the family-set / freshness contract is
 *   observed on the validator seam the validate stage itself runs (`buildFamilyFresh` /
 *   `resolveGatedPass`). No lane imports a module that does not exist yet.
 *
 * HERMETICITY: every lane uses the injected seams only. No subprocess, no live index, no
 * network, no corpus filesystem access, no `Math.random()`, and no assertion on an exact
 * elapsed time.
 */
/* global console */
import { beforeAll, describe, expect, it } from '@jest/globals';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { updateRag } from '../update-rag.mjs';
import { buildFamilyFresh, resolveGatedPass } from '../validate-index.mjs';
import { CORPUS_FAMILY_NAMES } from '../corpus-sources.mjs';
import { isFreshDocument } from '../freshness.mjs';
import { repoRoot } from '../init-schema.mjs';

const moduleDirectory = path.dirname(fileURLToPath(import.meta.url));

/** The frozen, READ-ONLY baseline fixture this lane reads at runtime. */
const BASELINE_FIXTURE_PATH = path.join(
  moduleDirectory,
  'fixtures',
  'build-memory-baseline.json',
);

/** The canonical stage order of the `rag:update` chain. */
const CANONICAL_STAGE_ORDER = [
  'build',
  'prewarm-embed',
  'build-terms',
  'build-graph',
  'snapshot',
  'validate',
];

/**
 * The candidate key names a stage report may use for its peak-memory figure.
 *
 * The plan does NOT pin the field name, so the detector tolerates a documented candidate set
 * instead of hard-coding a guessed name.
 */
const PEAK_KEY_PATTERN = /peak|rss|memory|resident|highwater|hwm/i;

/** The maximum depth the peak detector descends into nested objects. */
const PEAK_DETECTOR_MAX_DEPTH = 2;

// ---------------------------------------------------------------------------
// The MEASURED current values (read from the live tree on the RED lane)
// ---------------------------------------------------------------------------

/**
 * What the live pipeline publishes TODAY, recorded so the green lane compares against a
 * measurement instead of a hand-written expectation. These are NOT assertions about the target
 * contract - the target contract is asserted in the lanes below.
 *
 * `measuredElapsedMsSample` and `measuredPeakFigures` are the values of the recorded RED run
 * (the elapsed times vary from run to run - the lane asserts only that every stage reports a
 * finite elapsed time >= 0, never an exact time). The per-run values are printed by the
 * `[P1-S3-1] measured stage report:` line the AC-206-A lane emits.
 */
export const MEASURED_CURRENT_VALUES = {
  stageCount: 6,
  stageOrder: CANONICAL_STAGE_ORDER,
  stageStatuses: ['ok', 'ok', 'ok', 'ok', 'ok', 'ok'],
  stageReportFields: ['name', 'status', 'elapsedMs'],
  stagesPublishingAPeakFigure: 0,
  measuredElapsedMsSample: [14, 15, 16, 16, 15, 16],
  measuredPeakFigures: [null, null, null, null, null, null],
  baselinePeakBytes: 563331072,
  baselinePeakBytesSource:
    'multiSample.postRemediation.primary.peakRssBytes.median',
  updateRagModuleNamespace: ['updateRag'],
  identityFunctionPublished: false,
  publishedFamilyCount: CORPUS_FAMILY_NAMES.length,
  publishedFamilies: [...CORPUS_FAMILY_NAMES],
};

// ---------------------------------------------------------------------------
// Helpers - the fixture, the baseline figure and the peak detector
// ---------------------------------------------------------------------------

/**
 * Read the frozen memory baseline fixture.
 *
 * @returns {Promise<object>} The parsed baseline fixture.
 */
async function readBaselineFixture() {
  return JSON.parse(await readFile(BASELINE_FIXTURE_PATH, 'utf8'));
}

/**
 * Resolve the fixture-recorded baseline figure through the documented fallback chain.
 *
 * @param {object} baseline - The parsed baseline fixture.
 * @returns {number | undefined} The recorded peak figure, or undefined when the fixture
 *   records none.
 */
function resolveBaselinePeakBytes(baseline) {
  const candidates = [
    baseline?.multiSample?.postRemediation?.primary?.peakRssBytes?.median,
    baseline?.multiSample?.postRemediation?.pooled?.peakRssBytes?.median,
    baseline?.multiSample?.postRemediation?.pooled?.median,
    baseline?.measurements?.peakRssBytes,
  ];

  for (const candidate of candidates) {
    if (Number.isFinite(candidate) && candidate > 0) return candidate;
  }
  return undefined;
}

/** @type {number | undefined} */
let baselinePeakBytes = undefined;

beforeAll(async () => {
  baselinePeakBytes = resolveBaselinePeakBytes(await readBaselineFixture());
});

/**
 * Coerce a published figure into a positive finite number.
 *
 * A numeric string of digits counts; anything else does not.
 *
 * @param {unknown} entry - The published value.
 * @returns {number | undefined} The positive finite number, or undefined.
 */
function toPositiveFiniteNumber(entry) {
  if (typeof entry === 'number') {
    return Number.isFinite(entry) && entry > 0 ? entry : undefined;
  }
  if (typeof entry === 'string' && /^\d+$/.test(entry.trim())) {
    const parsed = Number(entry.trim());
    return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
  }
  return undefined;
}

/**
 * Collect every peak-memory figure published under a matching key name.
 *
 * Strings are NEVER descended into, so a JSON payload carried in a stage's `stdout` cannot
 * accidentally satisfy the detector.
 *
 * @param {unknown} value - The value to scan.
 * @param {number} [depth=0] - The current recursion depth.
 * @returns {number[]} The positive finite figures found.
 */
function collectPeakFigures(value, depth = 0) {
  if (!value || typeof value !== 'object' || depth > PEAK_DETECTOR_MAX_DEPTH) {
    return [];
  }

  const figures = [];
  for (const [key, entry] of Object.entries(value)) {
    if (PEAK_KEY_PATTERN.test(key)) {
      const numeric = toPositiveFiniteNumber(entry);
      if (numeric !== undefined) {
        figures.push(numeric);
        continue;
      }
    }
    if (entry && typeof entry === 'object') {
      figures.push(...collectPeakFigures(entry, depth + 1));
    }
  }
  return figures;
}

/**
 * Read the peak-memory figure a stage report publishes.
 *
 * @param {object} stage - The stage report entry.
 * @returns {number | undefined} The published figure, or undefined when the stage publishes none.
 */
function readPeakFigure(stage) {
  const found = collectPeakFigures(stage);
  return found.length ? Math.max(...found) : undefined;
}

// ---------------------------------------------------------------------------
// Helpers - the injected pipeline doubles
// ---------------------------------------------------------------------------

/**
 * Block synchronously for a few milliseconds so the measured elapsed time is non-zero.
 *
 * @returns {void}
 */
function waitSynchronously() {
  try {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 3);
  } catch {
    const until = Date.now() + 3;
    while (Date.now() < until) {
      // Busy-wait fallback for a host that refuses to block the main thread.
    }
  }
}

/**
 * Create an injectable stage runner that publishes the peak figure through EVERY plausible
 * channel the implementing slice may read (the runner result object OR the child's stdout), so
 * the lane is satisfiable whatever spelling the implementation chooses.
 *
 * The plan does NOT pin the name of the field the implementing slice reads off the runner
 * result, so the double publishes the figure under every listed alias AND answers any other
 * peak-like property read off that result (a `Proxy` fallback). The stage report therefore
 * carries the figure whatever spelling `P1-S3-2` chooses, and the lane stays RED for the
 * RIGHT reason today - today `updateRag` reads ONLY `status`, `stdout` and `stderr` off the
 * result, so the published figure is DISCARDED and no stage report carries it.
 *
 * @param {number} peakBytes - The peak figure the runner publishes.
 * @returns {function} The injectable stage runner. The seam calls it with
 *   `{ name, script, args }`; the double ignores them because the peak publication must not
 *   depend on the stage identity.
 */
function createPeakPublishingRunner(peakBytes) {
  const payload = {
    peakMemoryBytes: peakBytes,
    peakRssBytes: peakBytes,
    peakBytes: peakBytes,
    peakMemory: peakBytes,
    maxRssBytes: peakBytes,
    maxRSS: peakBytes,
    memory: { peakMemoryBytes: peakBytes, peakRssBytes: peakBytes },
  };

  // `stdout` stays PURE JSON so a runner that parses the child's stdout as one JSON document
  // succeeds; the `key=value` spelling is offered on `stderr` for a runner that scans it.
  const result = {
    status: 0,
    stdout: `${JSON.stringify(payload)}\n`,
    stderr: `PEAK_RSS_BYTES=${peakBytes}\nPEAK_MEMORY_BYTES=${peakBytes}\n`,
    ...payload,
  };

  return () => {
    waitSynchronously();
    return new Proxy(result, {
      get(target, property, receiver) {
        if (property in target) {
          return Reflect.get(target, property, receiver);
        }
        if (
          typeof property === 'string' &&
          PEAK_KEY_PATTERN.test(property)
        ) {
          return peakBytes;
        }
        return undefined;
      },
    });
  };
}

/**
 * Drive the `rag:update` chain with the injected doubles and return the summary.
 *
 * The hash store is injected so the `build-graph` stage is NOT skipped, and the hash writer is
 * a no-op so the lane never touches `rag-index/data/corpus-hash.json`.
 *
 * @returns {Promise<{ summary: object, baselinePeakBytes: number | undefined }>} The pipeline
 *   summary and the baseline figure the runner published.
 */
async function runPeakPublishingPipeline() {
  const peakBytes =
    baselinePeakBytes ?? resolveBaselinePeakBytes(await readBaselineFixture());

  const summary = await updateRag({
    spawnRunner: createPeakPublishingRunner(peakBytes),
    readHashFile: async () => null,
    writeHashFile: async () => {},
    computeCorpusHash: async () => 'stable-corpus-hash',
  });

  return { summary, baselinePeakBytes: peakBytes };
}

/**
 * Render the measured stage report as a stable, parseable line.
 *
 * @param {object} summary - The pipeline summary.
 * @returns {{ stageOrder: string[], stages: object[] }} The measured stage report.
 */
function measureStageReport(summary) {
  const stages = summary.stages.map((stage) => ({
    name: stage.name,
    status: stage.status,
    elapsedMs: stage.elapsedMs,
    peakMemoryBytes: readPeakFigure(stage) ?? null,
  }));

  return { stageOrder: stages.map((stage) => stage.name), stages };
}

// ---------------------------------------------------------------------------
// Helpers - the synthetic corpus observed on the validator seam
// ---------------------------------------------------------------------------

/**
 * Mirror the validator's internal repo-relative normalisation so the synthetic family
 * assignments address the same keys the validator looks up.
 *
 * @param {string} filePath - The document path.
 * @returns {string} The repo-relative, forward-slash path.
 */
function toRepoRelativeKey(filePath) {
  return path.relative(repoRoot, filePath).replaceAll(path.sep, '/');
}

/**
 * Build the synthetic, all-fresh corpus observed on the validator seam: one document row and
 * one matching freshness proof per published family.
 *
 * @returns {{ documents: object[], freshnessChecks: object[], familyAssignments: Map<string, string> }}
 *   The synthetic corpus.
 */
function createSyntheticFreshCorpus() {
  const documents = [];
  const freshnessChecks = [];
  const familyAssignments = new Map();

  CORPUS_FAMILY_NAMES.forEach((family, index) => {
    const documentRow = {
      file_path: `synthetic/${family}/document-${index}.md`,
      sha256: String(index).padStart(2, '0').repeat(32),
      mtime_ms: 1700000000000 + index,
      file_size: 1024 + index,
      doc_family: family,
    };
    const proof = {
      file_path: documentRow.file_path,
      mtime_ms: documentRow.mtime_ms,
      size: documentRow.file_size,
      sha256: documentRow.sha256,
    };

    // The proof matches its document row by construction; the validator's own predicate is the
    // authority, so the synthetic corpus is built against it.
    if (!isFreshDocument(documentRow, proof)) {
      throw new Error(
        'synthetic freshness proof does not match its document row',
      );
    }

    documents.push(documentRow);
    freshnessChecks.push(proof);
    familyAssignments.set(toRepoRelativeKey(documentRow.file_path), family);
  });

  return { documents, freshnessChecks, familyAssignments };
}

// ---------------------------------------------------------------------------
// Helpers - the corpus identity contract
// ---------------------------------------------------------------------------

/** The corpus rows the identity contract is measured on. */
const CORPUS_ROWS = [
  {
    file_path: 'src/example.ts',
    sha256: 'aaaa1111',
    mtime_ms: 1700000000000,
    file_size: 4096,
    doc_family: 'ts-source',
  },
  {
    file_path: 'plans/example.md',
    sha256: 'bbbb2222',
    mtime_ms: 1700000000001,
    file_size: 2048,
    doc_family: 'plan',
  },
];

/** A corpus whose rows all carry the `plan` family. */
const CORPUS_A = CORPUS_ROWS.map((row) => ({ ...row, doc_family: 'plan' }));

/** A corpus whose rows are byte-identical to `CORPUS_A` except for the family. */
const CORPUS_B = CORPUS_ROWS.map((row) => ({ ...row, doc_family: 'demo' }));

/**
 * Read the corpus identity function OFF THE MODULE NAMESPACE.
 *
 * A DYNAMIC import of an EXISTING module never fails at link time, so the lane is RED BY
 * ASSERTION (the symbol is simply absent today) and never by an import failure.
 *
 * @returns {Promise<Function | undefined>} The identity function, or undefined when the module
 *   publishes none.
 */
async function readIdentityFunction() {
  const updateRagModule = await import('../update-rag.mjs');
  return updateRagModule.computeCorpusIdentity;
}

/**
 * Render an identity value so two identities compare BY VALUE instead of by object identity.
 *
 * @param {unknown} value - The identity value.
 * @returns {string} The canonical rendering.
 */
function canonicalizeIdentity(value) {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  if (typeof value === 'string') return `string:${value}`;
  if (typeof value === 'number' || typeof value === 'boolean') {
    return `${typeof value}:${String(value)}`;
  }
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalizeIdentity(entry)).join(',')}]`;
  }
  if (typeof value === 'object') {
    const entries = Object.keys(value)
      .sort()
      .map((key) => `${key}:${canonicalizeIdentity(value[key])}`);
    return `{${entries.join(',')}}`;
  }
  return `${typeof value}:${String(value)}`;
}

/**
 * Call the identity function without ever letting a call-time error escape the lane.
 *
 * A lane that fails BY ASSERTION reports an unpublished identity; it never crashes.
 *
 * @param {Function | undefined} identityFn - The identity function.
 * @param {object[]} corpus - The corpus rows.
 * @returns {Promise<string | null>} The canonical identity, or null when the identity could not
 *   be produced.
 */
async function callIdentity(identityFn, corpus) {
  if (typeof identityFn !== 'function') return null;
  try {
    return canonicalizeIdentity(await Promise.resolve(identityFn(corpus)));
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// AC-203-B - the pipeline reports the peak
// ---------------------------------------------------------------------------

describe('AC-203-B (the pipeline reports the peak)', () => {
  it('publishes a per-stage peak memory figure for every stage of the chain', async () => {
    const { summary } = await runPeakPublishingPipeline();
    const stagesWithPeak = summary.stages.filter((stage) =>
      Number.isFinite(readPeakFigure(stage)),
    ).length;

    expect({
      totalStages: summary.stages.length,
      stagesWithPeak,
    }).toEqual({
      totalStages: 6,
      stagesWithPeak: 6,
    });
  });

  it('pins the published figure against the recorded pre-change baseline', async () => {
    const { summary, baselinePeakBytes: recordedBaseline } =
      await runPeakPublishingPipeline();
    const figures = summary.stages.map((stage) => readPeakFigure(stage));
    const baselineIsPositive =
      Number.isFinite(recordedBaseline) && recordedBaseline > 0;
    const everyStagePinnedToBaseline =
      baselineIsPositive &&
      figures.length > 0 &&
      figures.every((figure) => figure === recordedBaseline);

    expect({ baselineIsPositive, everyStagePinnedToBaseline }).toEqual({
      baselineIsPositive: true,
      everyStagePinnedToBaseline: true,
    });
  });
});

// ---------------------------------------------------------------------------
// AC-206-A - the whole chain completes
// ---------------------------------------------------------------------------

describe('AC-206-A (the whole chain completes)', () => {
  it('runs the six canonical stages in the canonical order', async () => {
    const { summary } = await runPeakPublishingPipeline();
    const stageNames = summary.stages.map((stage) => stage.name);

    expect(stageNames).toEqual(CANONICAL_STAGE_ORDER);
  });

  it('reports success for every stage of the chain', async () => {
    const { summary } = await runPeakPublishingPipeline();
    const statuses = summary.stages.map((stage) => stage.status);

    expect({
      statuses,
      everyStageSucceeded: statuses.every((status) => status === 'ok'),
    }).toEqual({
      statuses: ['ok', 'ok', 'ok', 'ok', 'ok', 'ok'],
      everyStageSucceeded: true,
    });
  });

  it('publishes the family set with every family fresh', () => {
    const { documents, freshnessChecks, familyAssignments } =
      createSyntheticFreshCorpus();
    const familyFresh = buildFamilyFresh(
      documents,
      freshnessChecks,
      familyAssignments,
    );
    const publishedFamilies = Object.keys(familyFresh);

    expect({
      publishedFamilies,
      publishedFamiliesMatchContract:
        publishedFamilies.length === CORPUS_FAMILY_NAMES.length &&
        publishedFamilies.every(
          (family, index) => family === CORPUS_FAMILY_NAMES[index],
        ),
      allFamiliesFresh: publishedFamilies.every(
        (family) => familyFresh[family].fresh === true,
      ),
      gatedPass: resolveGatedPass(familyFresh),
    }).toEqual({
      publishedFamilies: CORPUS_FAMILY_NAMES,
      publishedFamiliesMatchContract: true,
      allFamiliesFresh: true,
      gatedPass: true,
    });
  });

  it('flips a stale ts-source document to not fresh and fails the gated pass', () => {
    const { documents, freshnessChecks, familyAssignments } =
      createSyntheticFreshCorpus();
    const staleDocument = {
      ...documents[0],
      sha256: 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
    };
    const staleProof = {
      ...freshnessChecks[0],
      sha256: 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
    };
    const familyFresh = buildFamilyFresh(
      [staleDocument],
      [staleProof],
      familyAssignments,
    );

    expect({
      tsSourceFresh: familyFresh['ts-source'].fresh,
      stalePathRecorded: familyFresh['ts-source'].stalePaths.includes(
        staleDocument.file_path,
      ),
      gatedPass: resolveGatedPass(familyFresh),
    }).toEqual({
      tsSourceFresh: false,
      stalePathRecorded: true,
      gatedPass: false,
    });
  });

  it('records the measured per-stage elapsed time and peak memory', async () => {
    const { summary } = await runPeakPublishingPipeline();
    const measured = measureStageReport(summary);
    const everyStageReportsAFiniteElapsedTime = measured.stages.every(
      (stage) => Number.isFinite(stage.elapsedMs) && stage.elapsedMs >= 0,
    );

    console.info(
      `[P1-S3-1] measured stage report: ${JSON.stringify(measured)}`,
    );

    expect({
      everyStageReportsAFiniteElapsedTime,
      measuredStageCount: measured.stages.length,
    }).toEqual({
      everyStageReportsAFiniteElapsedTime: true,
      measuredStageCount: 6,
    });
  });
});

// ---------------------------------------------------------------------------
// AC-203-E-RED - the corpus identity carries the family assignment
// ---------------------------------------------------------------------------

describe('AC-203-E-RED (the corpus identity carries the family assignment)', () => {
  it('publishes the corpus identity composition on the module namespace', async () => {
    const identityFn = await readIdentityFunction();

    expect(typeof identityFn).toBe('function');
  });

  it('produces different identities for corpora that differ only in doc_family', async () => {
    const identityFn = await readIdentityFunction();
    const published = typeof identityFn === 'function';
    const identityA = await callIdentity(identityFn, CORPUS_A);
    const identityB = await callIdentity(identityFn, CORPUS_B);

    expect({
      identityFunctionPublished: published,
      identitiesDiffer:
        identityA !== null && identityB !== null && identityA !== identityB,
    }).toEqual({
      identityFunctionPublished: true,
      identitiesDiffer: true,
    });
  });

  it('produces identical identities for two copies of the same corpus', async () => {
    const identityFn = await readIdentityFunction();
    const published = typeof identityFn === 'function';
    const identityA = await callIdentity(identityFn, CORPUS_A);
    const identityB = await callIdentity(
      identityFn,
      CORPUS_A.map((row) => ({ ...row })),
    );

    expect({
      identityFunctionPublished: published,
      identitiesMatch:
        identityA !== null && identityB !== null && identityA === identityB,
    }).toEqual({
      identityFunctionPublished: true,
      identitiesMatch: true,
    });
  });
});
