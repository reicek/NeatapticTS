/* global afterEach, beforeEach, console, describe, expect, it, setTimeout */
/**
 * @module reindex-plan-family.test
 * @description Red-style contract tests for the synchronous post-save plan
 * hook (P2-S1-A) and the pre-dispatch freshness wait (P2-S1-B).
 *
 * Coverage map:
 * - reindexPlanFamily: sync BM25 within the 60s budget, background dense
 *   queue, timeout and failure graceful-degrade paths, path normalization.
 * - waitForPlanFresh: fresh manifest, timeout degrade, mid-write parse-error
 *   tolerance (concurrent atomic write-then-rename in flight).
 * - main: CLI surface for file mode and --wait-fresh mode.
 */
import { jest } from '@jest/globals';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import {
  DEFAULT_MAX_SYNC_WAIT_MS,
  main,
  reindexPlanFamily,
  readPlanManifest,
  waitForPlanFresh,
} from '../reindex-plan-family.mjs';
import { repoRoot } from '../init-schema.mjs';

const TEMP_DIR = path.join(__dirname, '__tmp_reindex_plan_family_test');

/** Build a manifest payload with the plan family in the given state. */
function manifestWithPlanFresh(fresh) {
  return {
    lastReindex: Date.now(),
    updatedBy: 'reindex-plan-family.test',
    families: {
      plan: {
        fresh,
        stalePaths: fresh ? [] : ['plans/app.plans.md'],
        lastReindex: Date.now(),
        maxSyncWaitMs: 60_000,
        gated: true,
      },
    },
  };
}

function okSyncRunner() {
  return jest.fn().mockResolvedValue({ ok: true, timedOut: false });
}

function recordingBackgroundRunner() {
  return jest.fn().mockReturnValue(undefined);
}

beforeEach(() => {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
});

afterEach(() => {
  fs.rmSync(TEMP_DIR, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// reindexPlanFamily (P2-S1-A)
// ---------------------------------------------------------------------------

describe('reindexPlanFamily', () => {
  it('is trivially fresh when no plan files changed', async () => {
    const runSync = okSyncRunner();
    const result = await reindexPlanFamily(['README.md', 'src/x.ts'], {
      runSync,
    });

    expect(result).toEqual({
      syncFresh: true,
      planFresh: true,
      stale: false,
      files: [],
    });
    expect(runSync).not.toHaveBeenCalled();
  });

  it('defaults to the 60s synchronous budget', () => {
    expect(DEFAULT_MAX_SYNC_WAIT_MS).toBe(60_000);
  });

  it('runs BM25 synchronously via build-index --files within the budget', async () => {
    const runSync = okSyncRunner();
    const result = await reindexPlanFamily(
      ['plans/RAG_Index_Freshness_Strategy.plans.md'],
      { runSync, runBackground: recordingBackgroundRunner() },
    );

    expect(result).toEqual({
      syncFresh: true,
      planFresh: true,
      stale: false,
      files: ['plans/RAG_Index_Freshness_Strategy.plans.md'],
    });
    expect(runSync).toHaveBeenCalledWith(
      [
        'node',
        'rag-index/build-index.mjs',
        '--files=plans/RAG_Index_Freshness_Strategy.plans.md',
      ],
      { timeoutMs: 60_000 },
    );
  });

  it('respects an explicit maxSyncWaitMs override', async () => {
    const runSync = okSyncRunner();
    await reindexPlanFamily(['plans/app.plans.md'], {
      runSync,
      runBackground: recordingBackgroundRunner(),
      maxSyncWaitMs: 5_000,
    });

    expect(runSync).toHaveBeenCalledWith(expect.any(Array), {
      timeoutMs: 5_000,
    });
  });

  it('queues dense embedding in the background after a successful sync build', async () => {
    const runBackground = recordingBackgroundRunner();
    await reindexPlanFamily(['plans/app.plans.md', 'plans/other.plans.md'], {
      runSync: okSyncRunner(),
      runBackground,
    });

    expect(runBackground).toHaveBeenCalledWith([
      'node',
      'rag-index/embed-index.mjs',
      '--files=plans/app.plans.md',
      '--files=plans/other.plans.md',
    ]);
  });

  it('returns stale with reason timeout when the sync build exceeds the budget, without throwing', async () => {
    const runSync = jest.fn().mockResolvedValue({ ok: false, timedOut: true });
    const onWarning = jest.fn();
    const result = await reindexPlanFamily(['plans/app.plans.md'], {
      runSync,
      runBackground: recordingBackgroundRunner(),
      onWarning,
    });

    expect(result).toEqual({
      syncFresh: false,
      planFresh: false,
      stale: true,
      reason: 'timeout',
      files: ['plans/app.plans.md'],
    });
    expect(onWarning).toHaveBeenCalledTimes(1);
  });

  it('returns stale with reason build-failed and skips the dense queue on non-zero exit', async () => {
    const runSync = jest.fn().mockResolvedValue({ ok: false });
    const runBackground = recordingBackgroundRunner();
    const result = await reindexPlanFamily(['plans/app.plans.md'], {
      runSync,
      runBackground,
    });

    expect(result.reason).toBe('build-failed');
    expect(runBackground).not.toHaveBeenCalled();
  });

  it('warns through the default console.error sink when not injected', async () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const runSync = jest.fn().mockResolvedValue({ ok: false });

    await reindexPlanFamily(['plans/app.plans.md'], { runSync });

    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining('reindex-plan-family:'),
    );
    errorSpy.mockRestore();
  });

  it('normalizes absolute paths to repo-relative POSIX paths', async () => {
    const runSync = okSyncRunner();
    const absolute = path.join(repoRoot, 'plans', 'app.plans.md');
    const result = await reindexPlanFamily([absolute], {
      runSync,
      runBackground: recordingBackgroundRunner(),
    });

    expect(result.files).toEqual(['plans/app.plans.md']);
  });

  it('deduplicates and lexicographically sorts changed paths', async () => {
    const result = await reindexPlanFamily(
      ['plans/b.plans.md', 'plans/a.plans.md', 'plans/b.plans.md'],
      { runSync: okSyncRunner(), runBackground: recordingBackgroundRunner() },
    );

    expect(result.files).toEqual(['plans/a.plans.md', 'plans/b.plans.md']);
  });

  it('tolerates a non-array changedPaths argument', async () => {
    const result = await reindexPlanFamily(null, { runSync: okSyncRunner() });

    expect(result.files).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// readPlanManifest
// ---------------------------------------------------------------------------

describe('readPlanManifest', () => {
  it('returns null when the manifest does not exist', () => {
    const missing = path.join(TEMP_DIR, 'absent.json');

    expect(readPlanManifest(missing)).toBeNull();
  });

  it('returns null for a partially written manifest without throwing', () => {
    const partial = path.join(TEMP_DIR, 'partial.json');
    fs.writeFileSync(partial, '{ "families": { "plan":', 'utf8');

    expect(readPlanManifest(partial)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// waitForPlanFresh (P2-S1-B)
// ---------------------------------------------------------------------------

describe('waitForPlanFresh', () => {
  it('returns planFresh true when the manifest already reports the plan family fresh', async () => {
    const manifestPath = path.join(TEMP_DIR, 'fresh.json');
    fs.writeFileSync(
      manifestPath,
      JSON.stringify(manifestWithPlanFresh(true)),
      'utf8',
    );
    const result = await waitForPlanFresh({
      manifestPath,
      maxSyncWaitMs: 1_000,
      pollIntervalMs: 25,
    });

    expect(result.planFresh).toBe(true);
    expect(result.stale).toBe(false);
  });

  it('returns stale true on timeout when the plan family stays stale', async () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const manifestPath = path.join(TEMP_DIR, 'stale.json');
    fs.writeFileSync(
      manifestPath,
      JSON.stringify(manifestWithPlanFresh(false)),
      'utf8',
    );
    const result = await waitForPlanFresh({
      manifestPath,
      maxSyncWaitMs: 150,
      pollIntervalMs: 20,
    });

    expect(result.stale).toBe(true);
    expect(result.reason).toBe('timeout');
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining('stale'));
    errorSpy.mockRestore();
  });

  it('does not throw when the manifest is mid-write and reports not-fresh', async () => {
    const manifestPath = path.join(TEMP_DIR, 'midwrite.json');
    fs.writeFileSync(manifestPath, '{ "families": { "plan":', 'utf8');

    const result = await waitForPlanFresh({
      manifestPath,
      maxSyncWaitMs: 120,
      pollIntervalMs: 20,
      onWarning: jest.fn(),
    });

    expect(result.planFresh).toBe(false);
    expect(result.stale).toBe(true);
  });

  it('recovers once a mid-write manifest becomes fresh (concurrent write + poll)', async () => {
    const manifestPath = path.join(TEMP_DIR, 'recovering.json');
    fs.writeFileSync(manifestPath, '{ "families":', 'utf8');
    setTimeout(() => {
      fs.writeFileSync(
        manifestPath,
        JSON.stringify(manifestWithPlanFresh(true)),
        'utf8',
      );
    }, 40);

    const result = await waitForPlanFresh({
      manifestPath,
      maxSyncWaitMs: 2_000,
      pollIntervalMs: 10,
    });

    expect(result.planFresh).toBe(true);
  });

  it('honors an injected manifest reader over the real file', async () => {
    const freshManifest = manifestWithPlanFresh(true);
    const readManifest = jest
      .fn()
      .mockReturnValueOnce(null)
      .mockReturnValueOnce(freshManifest);
    const result = await waitForPlanFresh({
      readManifest,
      maxSyncWaitMs: 1_000,
      pollIntervalMs: 10,
    });

    expect(result.planFresh).toBe(true);
    expect(readManifest.mock.calls.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// main (CLI surface)
// ---------------------------------------------------------------------------

describe('main CLI surface', () => {
  it('prints a trivially-fresh report when invoked with no --files', async () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    const result = await main([]);

    expect(result.syncFresh).toBe(true);
    expect(logSpy).toHaveBeenCalledWith(
      expect.stringContaining('"syncFresh": true'),
    );
    logSpy.mockRestore();
  });

  it('supports --wait-fresh mode with injected dependencies', async () => {
    const result = await main(['--wait-fresh', '--max-wait-ms=500'], {
      readManifest: () => manifestWithPlanFresh(true),
      log: () => {},
    });

    expect(result.planFresh).toBe(true);
  });

  it('forwards --files to the injected sync runner', async () => {
    const runSync = okSyncRunner();
    const runBackground = recordingBackgroundRunner();
    await main(['--files=plans/a.plans.md'], {
      runSync,
      runBackground,
      log: () => {},
    });

    expect(runSync).toHaveBeenCalledWith(
      ['node', 'rag-index/build-index.mjs', '--files=plans/a.plans.md'],
      { timeoutMs: 60_000 },
    );
  });
});
