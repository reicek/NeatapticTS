/* global process, URL */

import {
  beforeEach,
  describe,
  expect,
  jest,
  test,
} from '@jest/globals';
import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

jest.unstable_mockModule('fast-glob', () => ({
  __esModule: true,
  default: jest.fn(),
}));

const { default: fg } = await import('fast-glob');
const {
  buildFamilyFresh,
  sweepDeletedPaths,
  validateSemanticIndex,
  writeFreshnessManifest,
} = await import('../validate-index.mjs');

describe('validate-index per-family freshness', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  test('removes document rows for indexed files that no longer exist on disk', async () => {
    const indexedPaths = ['src/keep.ts', 'src/missing.ts'];
    const diskPaths = ['src/keep.ts'];
    const executed = [];
    const batched = [];
    const client = createMockClient({ indexedPaths, executed, batched });

    fg.mockResolvedValue(diskPaths);

    const deleted = await sweepDeletedPaths(client, 'ts-source');

    expect(deleted).toEqual(['src/missing.ts']);
    expect(
      batched.some(
        (op) =>
          op.sql.includes('DELETE FROM documents') &&
          op.args.includes('src/missing.ts'),
      ),
    ).toBe(true);
    expect(
      batched.some(
        (op) =>
          op.sql.includes('DELETE FROM documents') &&
          op.args.includes('src/keep.ts'),
      ),
    ).toBe(false);
  });

  test('normalises windows-style separators before comparing paths', async () => {
    const indexedPaths = ['src/keep.ts', 'src\\missing.ts'];
    const diskPaths = ['src/keep.ts'];
    const executed = [];
    const batched = [];
    const client = createMockClient({ indexedPaths, executed, batched });

    fg.mockResolvedValue(diskPaths);

    const deleted = await sweepDeletedPaths(client, 'ts-source');

    expect(deleted).toEqual(['src/missing.ts']);
    expect(
      batched.some(
        (op) =>
          op.sql.includes('DELETE FROM documents') &&
          op.args.includes('src\\missing.ts'),
      ),
    ).toBe(true);
  });

  test('validateSemanticIndex exposes all ten families in family_fresh', async () => {
    const result = await validateSemanticIndex({
      documents: [],
      freshnessChecks: [],
      chunks: 0,
    });

    expect(result).toHaveProperty('family_fresh');
    expect(Object.keys(result.family_fresh).sort()).toEqual([
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
    ]);
    for (const entry of Object.values(result.family_fresh)) {
      expect(entry).toEqual({ fresh: true, stalePaths: [] });
    }
  });

  test('stale and missing paths are grouped by family and sorted', async () => {
    const documents = [
      { file_path: 'src/b.ts' },
      { file_path: 'src/a.ts' },
      { file_path: 'README.md' },
      { file_path: 'plans/old.md' },
      { file_path: 'completed/active.md' },
    ];
    const freshnessChecks = [
      { file_path: 'src/b.ts', missing: true },
      { file_path: 'src/a.ts', missing: true },
      { file_path: 'README.md', mtime_ms: 0, file_size: 1, sha256: 'x' },
      { file_path: 'plans/old.md', mtime_ms: 0, file_size: 1, sha256: 'y' },
      { file_path: 'completed/active.md', missing: true },
    ];
    const familyAssignments = new Map([
      ['src/a.ts', 'ts-source'],
      ['src/b.ts', 'ts-source'],
      ['README.md', 'readme'],
      ['plans/old.md', 'plan'],
      ['completed/active.md', 'completed-plan'],
    ]);

    const result = await validateSemanticIndex({
      documents,
      freshnessChecks,
      chunks: 5,
      familyAssignments,
    });

    expect(result.family_fresh['ts-source'].stalePaths).toEqual([
      'src/a.ts',
      'src/b.ts',
    ]);
    expect(result.family_fresh['readme'].stalePaths).toEqual(['README.md']);
    expect(result.family_fresh['plan'].stalePaths).toEqual(['plans/old.md']);
    expect(result.family_fresh['completed-plan'].stalePaths).toEqual([
      'completed/active.md',
    ]);
    expect(result.stale_paths).toEqual([
      'README.md',
      'completed/active.md',
      'plans/old.md',
      'src/a.ts',
      'src/b.ts',
    ]);
  });

  test('completed-plan staleness does not fail the gate', async () => {
    const documents = [{ file_path: 'completed/retired.md' }];
    const freshnessChecks = [{ file_path: 'completed/retired.md', missing: true }];
    const familyAssignments = new Map([
      ['completed/retired.md', 'completed-plan'],
    ]);

    const result = await validateSemanticIndex({
      documents,
      freshnessChecks,
      chunks: 1,
      familyAssignments,
    });

    expect(result.family_fresh['completed-plan'].fresh).toBe(false);
    expect(result.family_fresh['completed-plan'].stalePaths).toEqual([
      'completed/retired.md',
    ]);
    expect(result.failures).not.toEqual(
      expect.arrayContaining([
        expect.stringMatching(/completed\/retired\.md/),
      ]),
    );
    expect(result.pass).toBe(true);
  });

  test('gated family staleness fails the gate', async () => {
    const documents = [{ file_path: 'src/stale.ts' }];
    const freshnessChecks = [{ file_path: 'src/stale.ts', missing: true }];
    const familyAssignments = new Map([['src/stale.ts', 'ts-source']]);

    const result = await validateSemanticIndex({
      documents,
      freshnessChecks,
      chunks: 1,
      familyAssignments,
    });

    expect(result.family_fresh['ts-source'].fresh).toBe(false);
    expect(result.pass).toBe(false);
    expect(result.failures).toEqual(
      expect.arrayContaining([
        'Indexed file is missing: src/stale.ts.',
      ]),
    );
  });

  test('writeFreshnessManifest writes all families with gated metadata', async () => {
    const tempDir = path.join(tmpdir(), `freshness-test-${Date.now()}`);
    await mkdir(tempDir, { recursive: true });
    const outputPath = path.join(tempDir, 'freshness-manifest.json');
    const familyFresh = {
      readme: { fresh: true, stalePaths: [] },
      'ts-source': { fresh: false, stalePaths: ['src/a.ts', 'src/b.ts'] },
      plan: { fresh: true, stalePaths: [] },
      demo: { fresh: true, stalePaths: [] },
      benchmark: { fresh: true, stalePaths: [] },
      'root-doc': { fresh: true, stalePaths: [] },
      'completed-plan': { fresh: true, stalePaths: [] },
    };

    const manifest = await writeFreshnessManifest(familyFresh, outputPath);

    expect(manifest.families['ts-source'].fresh).toBe(false);
    expect(manifest.families['ts-source'].stalePaths).toEqual([
      'src/a.ts',
      'src/b.ts',
    ]);
    expect(manifest.families['ts-source'].gated).toBe(true);
    expect(manifest.families['ts-source'].maxSyncWaitMs).toBe(0);
    expect(manifest.families['plan'].maxSyncWaitMs).toBe(60000);
    expect(manifest.families['completed-plan'].gated).toBe(false);

    const written = JSON.parse(await readFile(outputPath, 'utf8'));
    expect(written.lastReindex).toBe(manifest.lastReindex);
    expect(written.updatedBy).toBe('validate-index.mjs');
    expect(written.families['ts-source'].stalePaths).toEqual([
      'src/a.ts',
      'src/b.ts',
    ]);

    await rm(tempDir, { recursive: true, force: true });
  });

  test('writeFreshnessManifest writes to a temp file and atomically renames it', async () => {
    const tempDir = path.join(tmpdir(), `freshness-atomic-${Date.now()}`);
    await mkdir(tempDir, { recursive: true });
    const outputPath = path.join(tempDir, 'freshness-manifest.json');
    const priorContent = '{ "prior": true }';
    await writeFile(outputPath, priorContent);

    const manifest = await writeFreshnessManifest(
      buildFamilyFresh([], []),
      outputPath,
    );

    const files = await readdir(tempDir);
    expect(files).toContain('freshness-manifest.json');
    expect(files.filter((name) => name.endsWith('.tmp'))).toEqual([]);
    const written = JSON.parse(await readFile(outputPath, 'utf8'));
    expect(written.lastReindex).toBe(manifest.lastReindex);
    expect(written.families).toBeDefined();

    await rm(tempDir, { recursive: true, force: true });
  });

  test('writeFreshnessManifest never writes partial manifest', async () => {
    const tempDir = path.join(tmpdir(), `freshness-concurrent-${Date.now()}`);
    await mkdir(tempDir, { recursive: true });
    const outputPath = path.join(tempDir, 'freshness-manifest.json');
    const familyFresh = buildFamilyFresh([], []);

    const reads = [];
    const writer = writeFreshnessManifest(familyFresh, outputPath);
    for (let i = 0; i < 50; i += 1) {
      reads.push(
        readFile(outputPath, 'utf8').then(
          (data) => ({ ok: true, data }),
          (error) => ({ ok: false, code: error.code }),
        ),
      );
    }

    await writer;
    const results = await Promise.all(reads);

    for (const result of results) {
      if (result.ok) {
        const parsed = JSON.parse(result.data);
        expect(parsed).toHaveProperty('lastReindex');
        expect(parsed).toHaveProperty('families');
      } else {
        expect(result.code).toBe('ENOENT');
      }
    }

    await rm(tempDir, { recursive: true, force: true });
  });

  function createMockClient({ indexedPaths, executed, batched = [] }) {
    return {
      execute: jest.fn(async ({ sql, args }) => {
        executed.push({ sql, args });
        if (sql.includes('SELECT file_path FROM documents')) {
          return { rows: indexedPaths.map((file_path) => ({ file_path })) };
        }
        return { rows: [], rowsAffected: 1 };
      }),
      batch: jest.fn(async (statements) => {
        for (const { sql, args } of statements) {
          batched.push({ sql, args });
        }
        return { rows: [] };
      }),
    };
  }

});
