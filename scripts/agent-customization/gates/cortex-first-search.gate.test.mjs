/**
 * @module cortex-first-search.gate.test
 * @description Coverage tests for cortex-first-search.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/cortex-first-search.gate.mjs',
);

let mockIndexReport;
let mockMcpReport;

jest.unstable_mockModule('../../../rag-index/init-schema.mjs', () => ({
  defaultDatabasePath: '/mock/db.sqlite',
  repoRoot: '/mock',
}));
jest.unstable_mockModule('../../../rag-index/validate-index.mjs', () => ({
  validateDatabase: async () => mockIndexReport,
  validateSemanticIndex: async () => mockIndexReport,
}));
jest.unstable_mockModule('./cortex-mcp-smoke.mjs', () => ({
  runCortexMcpSmoke: async () => mockMcpReport,
}));

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

async function importGateMain(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    await withArgv([process.execPath, GATE_PATH, ...argv], async () => {
      await import('./cortex-first-search.gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('cortex-first-search gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockIndexReport = { pass: true, documents: 5, chunks: 42 };
    mockMcpReport = { pass: true, evidence: { searchResults: 3 } };
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('runCortexFirstSearchGate', () => {
    it('returns pass=true when both validators pass', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate({
        databasePath: '/custom/db.sqlite',
        indexValidator: async () => ({ pass: true, documents: 2, chunks: 9 }),
        mcpSmoke: async () => ({ pass: true, evidence: { searchResults: 4 } }),
      });
      assert.equal(result.pass, true);
      assert.equal(result.evidence.index_documents, 2);
      assert.equal(result.evidence.index_chunks, 9);
      assert.equal(result.evidence.index_fresh, true);
      assert.equal(result.evidence.corpus_mcp_alive, true);
      assert.equal(result.evidence.corpus_search_results, 4);
      assert.equal(result.fixHint, null);
      assert.equal(result.owner, 'repo-cortex-workflow');
    });

    it('returns pass=false when only index fails', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate({
        databasePath: '/custom/db.sqlite',
        indexValidator: async () => ({ pass: false, fixHint: 'rebuild index' }),
        mcpSmoke: async () => ({ pass: true, evidence: { searchResults: 1 } }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.fixHint, 'rebuild index');
    });

    it('returns pass=false when only mcp fails', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate({
        databasePath: '/custom/db.sqlite',
        indexValidator: async () => ({ pass: true, documents: 1, chunks: 1 }),
        mcpSmoke: async () => ({ pass: false, fixHint: 'start mcp server' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.fixHint, 'start mcp server');
    });

    it('returns pass=false when both fail and prefers index fixHint', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate({
        databasePath: '/custom/db.sqlite',
        indexValidator: async () => ({ pass: false, fixHint: 'index hint' }),
        mcpSmoke: async () => ({ pass: false, fixHint: 'mcp hint' }),
      });
      assert.equal(result.pass, false);
      assert.equal(result.fixHint, 'index hint');
    });

    it('falls back to corpusMcpReport fixHint when index has none', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate({
        databasePath: '/custom/db.sqlite',
        indexValidator: async () => ({ pass: false }),
        mcpSmoke: async () => ({ pass: false, fixHint: 'mcp hint' }),
      });
      assert.equal(result.fixHint, 'mcp hint');
    });

    it('falls back to DEFAULT_FIX_HINT when neither has fixHint', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate({
        databasePath: '/custom/db.sqlite',
        indexValidator: async () => ({ pass: false }),
        mcpSmoke: async () => ({ pass: false }),
      });
      assert.ok(result.fixHint.includes('build-index.mjs'));
    });

    it('coalesces missing evidence fields to 0/false', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate({
        databasePath: '/custom/db.sqlite',
        indexValidator: async () => ({ pass: true }),
        mcpSmoke: async () => ({ pass: true }),
      });
      assert.equal(result.evidence.index_documents, 0);
      assert.equal(result.evidence.index_chunks, 0);
      assert.equal(result.evidence.index_fresh, true);
      assert.equal(result.evidence.corpus_search_results, 0);
    });

    it('uses default dependencies when no options provided', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate();
      assert.equal(result.pass, true);
    });

    it('uses defaultDatabasePath when databasePath not provided', async () => {
      const { runCortexFirstSearchGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./cortex-first-search.gate.mjs'),
      );
      const result = await runCortexFirstSearchGate({
        indexValidator: async () => ({ pass: true }),
        mcpSmoke: async () => ({ pass: true }),
      });
      assert.ok(result.evidence.database_path.includes('mock'));
    });
  });

  describe('main via import.meta.url guard', () => {
    it('emits JSON with --json when pass=true', async () => {
      const logs = await importGateMain(['--json']);
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
    });

    it('emits PASS text without --json when pass=true', async () => {
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.equal(process.exitCode, 0);
    });

    it('emits FAIL text with fixHint when pass=false', async () => {
      mockIndexReport = { pass: false };
      mockMcpReport = { pass: true };
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockIndexReport = { pass: false };
      mockMcpReport = { pass: false };
      const logs = await importGateMain(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });

    it('prints usage with --help', async () => {
      const logs = await importGateMain(['--help']);
      assert.ok(
        logs.some((l) => l.includes('Cortex-first search prerequisite gate')),
      );
    });

    it('passes --databasePath option through to gate', async () => {
      const logs = await importGateMain([
        '--json',
        '--databasePath=/override/db.sqlite',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(
        parsed.evidence.database_path,
        path.resolve('/override/db.sqlite'),
      );
    });
  });

  describe('import.meta.url guard', () => {
    it('does not run main when argv[1] does not match', async () => {
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        jest.resetModules();
        await withArgv([process.execPath, 'dummy'], async () => {
          await import('./cortex-first-search.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});
