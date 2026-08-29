/**
 * @module validate-index.max-age-sanity.test
 * @description Red tests for RAG Index Freshness Strategy Phase 3, slice
 *              P3-S1-B: per-family `max_age_ms` deletion-sweep sanity signal.
 *
 * Contract under test:
 * - The freshness manifest gains an optional per-family `max_age_ms` field
 *   (default `null` = disabled), written and preserved across rewrites by
 *   `writeFreshnessManifest` and loaded by `loadFreshnessManifest`.
 * - When a family has indexed files missing from disk AND its manifest
 *   `lastReindex` is older than its configured `max_age_ms`, validation emits
 *   a warning suggesting a deletion sweep. The warning never affects `pass`,
 *   `failures`, or `fixHint`.
 * - Unchanged (existing) files are never flagged by `max_age_ms` — the
 *   Phase 0 removal of the age gate for unchanged files is NOT re-introduced.
 *
 * Determinism note (plan risk R6): tests inject a fixed `now` and fixed
 * manifest timestamps — never Date.now() — so warning output is replay-stable
 * apart from injected clock values.
 *
 * RED PHASE: `validate-index.mjs` does not yet implement the signal. Tests
 * fail on the missing `warnings` result field and the missing
 * `loadFreshnessManifest` export.
 *
 * Pure .mjs test — runs via Jest ESM project `rag-index-mjs`.
 */

import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

const {
  loadFreshnessManifest,
  validateSemanticIndex,
  validateDatabase,
  writeFreshnessManifest,
} = await import('../validate-index.mjs');

// Fixed test clock — never Date.now() in assertions (determinism, plan risk R6).
const NOW = 1_000_000_000_000;
const STALE_LAST_REINDEX = NOW - 5_000;
const RECENT_LAST_REINDEX = NOW - 100;
const SMALL_MAX_AGE_MS = 1_000;

function makeDoc(filePath, overrides = {}) {
  return {
    file_path: filePath,
    mtime_ms: overrides.mtime_ms ?? NOW,
    file_size: overrides.file_size ?? 10,
    sha256: overrides.sha256 ?? 'hash',
    indexed_at: overrides.indexed_at ?? NOW,
  };
}

function makeFreshProof(filePath, overrides = {}) {
  return {
    file_path: filePath,
    mtime_ms: overrides.mtime_ms ?? NOW,
    size: overrides.size ?? 10,
    sha256: overrides.sha256 ?? 'hash',
  };
}

function makeManifest(families) {
  return { lastReindex: NOW, updatedBy: 'max-age-sanity.test', families };
}

function makeStubClient(documents) {
  return {
    async execute({ sql }) {
      if (sql.includes('COUNT(*)')) return { rows: [{ count: 0 }] };
      if (sql.includes('doc_family = ?')) return { rows: [] };
      return { rows: documents };
    },
  };
}

describe('per-family max_age_ms sanity signal (P3-S1-B)', () => {
  describe('validateSemanticIndex — warning-only signal', () => {
    it('does not flag unchanged files even when the manifest entry is far older than max_age_ms', async () => {
      const documents = [makeDoc('plans/kept.md')];
      const freshnessChecks = [makeFreshProof('plans/kept.md')];
      const familyAssignments = new Map([['plans/kept.md', 'plan']]);
      const manifest = makeManifest({
        plan: {
          fresh: true,
          stalePaths: [],
          lastReindex: 0,
          max_age_ms: SMALL_MAX_AGE_MS,
        },
      });

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        manifest,
        now: NOW,
      });

      expect(result.warnings).toEqual([]);
      expect(result.pass).toBe(true);
    });

    it('emits a deletion-sweep warning when a family has missing files and a stale manifest entry', async () => {
      const documents = [makeDoc('plans/gone.md')];
      const freshnessChecks = [{ file_path: 'plans/gone.md', missing: true }];
      const familyAssignments = new Map([['plans/gone.md', 'plan']]);
      const manifest = makeManifest({
        plan: {
          fresh: false,
          stalePaths: ['plans/gone.md'],
          lastReindex: STALE_LAST_REINDEX,
          max_age_ms: SMALL_MAX_AGE_MS,
        },
      });

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        manifest,
        now: NOW,
      });

      expect(result.warnings).toEqual([
        expect.stringMatching(/Deletion sweep suggested for family 'plan'/),
      ]);
      expect(result.pass).toBe(false);
    });

    it('keeps the warning out of the pass verdict for non-gated families', async () => {
      const documents = [makeDoc('completed/gone.md')];
      const freshnessChecks = [{ file_path: 'completed/gone.md', missing: true }];
      const familyAssignments = new Map([['completed/gone.md', 'completed-plan']]);
      const manifest = makeManifest({
        'completed-plan': {
          fresh: false,
          stalePaths: ['completed/gone.md'],
          lastReindex: STALE_LAST_REINDEX,
          max_age_ms: SMALL_MAX_AGE_MS,
        },
      });

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        manifest,
        now: NOW,
      });

      expect(result.pass).toBe(true);
      expect(result.warnings).toHaveLength(1);
    });

    it('does not warn when max_age_ms is null (disabled)', async () => {
      const documents = [makeDoc('plans/gone.md')];
      const freshnessChecks = [{ file_path: 'plans/gone.md', missing: true }];
      const familyAssignments = new Map([['plans/gone.md', 'plan']]);
      const manifest = makeManifest({
        plan: {
          fresh: false,
          stalePaths: ['plans/gone.md'],
          lastReindex: STALE_LAST_REINDEX,
          max_age_ms: null,
        },
      });

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        manifest,
        now: NOW,
      });

      expect(result.warnings).toEqual([]);
    });

    it('does not warn when manifest lastReindex is within max_age_ms', async () => {
      const documents = [makeDoc('plans/gone.md')];
      const freshnessChecks = [{ file_path: 'plans/gone.md', missing: true }];
      const familyAssignments = new Map([['plans/gone.md', 'plan']]);
      const manifest = makeManifest({
        plan: {
          fresh: false,
          stalePaths: ['plans/gone.md'],
          lastReindex: RECENT_LAST_REINDEX,
          max_age_ms: 10_000,
        },
      });

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        manifest,
        now: NOW,
      });

      expect(result.warnings).toEqual([]);
    });

    it('does not warn for families absent from the manifest', async () => {
      const documents = [makeDoc('plans/gone.md')];
      const freshnessChecks = [{ file_path: 'plans/gone.md', missing: true }];
      const familyAssignments = new Map([['plans/gone.md', 'plan']]);
      const manifest = makeManifest({});

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        manifest,
        now: NOW,
      });

      expect(result.warnings).toEqual([]);
    });

    it('does not warn when no manifest is provided', async () => {
      const documents = [makeDoc('plans/gone.md')];
      const freshnessChecks = [{ file_path: 'plans/gone.md', missing: true }];
      const familyAssignments = new Map([['plans/gone.md', 'plan']]);

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        now: NOW,
      });

      expect(result.warnings).toEqual([]);
    });

    it('does not flag content-stale files that still exist on disk', async () => {
      const documents = [makeDoc('src/stale.ts', { mtime_ms: 1, file_size: 1, sha256: 'stale' })];
      const freshnessChecks = [
        { file_path: 'src/stale.ts', mtime_ms: 2, size: 1, sha256: 'fresh' },
      ];
      const familyAssignments = new Map([['src/stale.ts', 'ts-source']]);
      const manifest = makeManifest({
        'ts-source': {
          fresh: false,
          stalePaths: ['src/stale.ts'],
          lastReindex: 0,
          max_age_ms: SMALL_MAX_AGE_MS,
        },
      });

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        manifest,
        now: NOW,
      });

      expect(result.warnings).toEqual([]);
    });

    it('sorts warnings deterministically across families', async () => {
      const documents = [makeDoc('src/gone.ts'), makeDoc('plans/gone.md')];
      const freshnessChecks = [
        { file_path: 'src/gone.ts', missing: true },
        { file_path: 'plans/gone.md', missing: true },
      ];
      const familyAssignments = new Map([
        ['src/gone.ts', 'ts-source'],
        ['plans/gone.md', 'plan'],
      ]);
      const manifest = makeManifest({
        plan: {
          fresh: false,
          stalePaths: ['plans/gone.md'],
          lastReindex: STALE_LAST_REINDEX,
          max_age_ms: SMALL_MAX_AGE_MS,
        },
        'ts-source': {
          fresh: false,
          stalePaths: ['src/gone.ts'],
          lastReindex: STALE_LAST_REINDEX,
          max_age_ms: SMALL_MAX_AGE_MS,
        },
      });

      const result = await validateSemanticIndex({
        documents,
        freshnessChecks,
        familyAssignments,
        manifest,
        now: NOW,
      });

      expect(result.warnings).toEqual([
        expect.stringMatching(/family 'plan'/),
        expect.stringMatching(/family 'ts-source'/),
      ]);
    });
  });

  describe('writeFreshnessManifest — max_age_ms schema', () => {
    let tempDir;

    beforeEach(async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'max-age-sanity-write-'));
    });

    afterEach(async () => {
      await rm(tempDir, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 200,
      });
    });

    it('writes max_age_ms as null for every family when no previous manifest is given', async () => {
      const outputPath = path.join(tempDir, 'freshness-manifest.json');

      const manifest = await writeFreshnessManifest({}, outputPath);

      expect(Object.values(manifest.families).map((entry) => entry.max_age_ms)).toEqual(
        Array(10).fill(null),
      );
    });

    it('preserves configured positive max_age_ms values from the previous manifest', async () => {
      const outputPath = path.join(tempDir, 'freshness-manifest.json');
      const previousManifest = makeManifest({
        plan: { max_age_ms: 86_400_000 },
      });

      const manifest = await writeFreshnessManifest(
        {},
        outputPath,
        'max-age-sanity.test',
        previousManifest,
      );

      expect(manifest.families.plan.max_age_ms).toBe(86_400_000);
      expect(manifest.families.readme.max_age_ms).toBe(null);
    });

    it('normalizes invalid max_age_ms values to null', async () => {
      const outputPath = path.join(tempDir, 'freshness-manifest.json');
      const previousManifest = makeManifest({
        plan: { max_age_ms: 'bogus' },
        demo: { max_age_ms: 0 },
        benchmark: { max_age_ms: -5 },
      });

      const manifest = await writeFreshnessManifest(
        {},
        outputPath,
        'max-age-sanity.test',
        previousManifest,
      );

      expect(manifest.families.plan.max_age_ms).toBe(null);
      expect(manifest.families.demo.max_age_ms).toBe(null);
      expect(manifest.families.benchmark.max_age_ms).toBe(null);
    });

    it('persists max_age_ms to the written manifest file', async () => {
      const outputPath = path.join(tempDir, 'freshness-manifest.json');
      const previousManifest = makeManifest({
        plan: { max_age_ms: 86_400_000 },
      });

      await writeFreshnessManifest({}, outputPath, 'max-age-sanity.test', previousManifest);

      const written = JSON.parse(await readFile(outputPath, 'utf8'));
      expect(written.families.plan.max_age_ms).toBe(86_400_000);
    });
  });

  describe('loadFreshnessManifest', () => {
    let tempDir;

    beforeEach(async () => {
      tempDir = await mkdtemp(path.join(tmpdir(), 'max-age-sanity-load-'));
    });

    afterEach(async () => {
      await rm(tempDir, {
        recursive: true,
        force: true,
        maxRetries: 10,
        retryDelay: 200,
      });
    });

    it('parses an existing manifest file', async () => {
      const manifestPath = path.join(tempDir, 'freshness-manifest.json');
      await writeFile(
        manifestPath,
        JSON.stringify(makeManifest({ plan: { max_age_ms: SMALL_MAX_AGE_MS } })),
        'utf8',
      );

      const manifest = await loadFreshnessManifest(manifestPath);

      expect(manifest?.families.plan.max_age_ms).toBe(SMALL_MAX_AGE_MS);
    });

    it('returns null when the manifest file is absent', async () => {
      const manifestPath = path.join(tempDir, 'absent.json');

      const manifest = await loadFreshnessManifest(manifestPath);

      expect(manifest).toBe(null);
    });

    it('returns null when the manifest is corrupt', async () => {
      const manifestPath = path.join(tempDir, 'corrupt.json');
      await writeFile(manifestPath, '{ not json', 'utf8');

      const manifest = await loadFreshnessManifest(manifestPath);

      expect(manifest).toBe(null);
    });

    it('returns null when the manifest is not a JSON object', async () => {
      const manifestPath = path.join(tempDir, 'scalar.json');
      await writeFile(manifestPath, '42', 'utf8');

      const manifest = await loadFreshnessManifest(manifestPath);

      expect(manifest).toBe(null);
    });
  });

  describe('validateDatabase — manifest wiring', () => {
    it('threads the previous manifest into the deletion-sweep warnings', async () => {
      const documents = [makeDoc('completed/gone.md')];
      const client = makeStubClient(documents);
      const familyAssignments = new Map([['completed/gone.md', 'completed-plan']]);
      const manifest = makeManifest({
        'completed-plan': {
          fresh: false,
          stalePaths: ['completed/gone.md'],
          lastReindex: STALE_LAST_REINDEX,
          max_age_ms: SMALL_MAX_AGE_MS,
        },
      });

      const result = await validateDatabase({
        client,
        familyAssignments,
        manifest,
      });

      expect(result.warnings).toHaveLength(1);
      expect(result.pass).toBe(true);
    });
  });
});