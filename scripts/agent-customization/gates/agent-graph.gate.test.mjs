/**
 * @module agent-graph.gate.test
 * @description Coverage tests for agent-graph.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/agent-graph.gate.mjs',
);

let mockReport;

jest.unstable_mockModule('../validate-agent-graph.mjs', () => ({
  runValidateAgentGraph: async () => mockReport,
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
      await import('./agent-graph.gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('agent-graph gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockReport = {
      ok: true,
      issues: [],
      graph: [],
      inventory: { summary: { by_tier: { 1: 5 } } },
    };
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('runAgentGraphGate', () => {
    it('returns pass=true when graph is valid', async () => {
      const { runAgentGraphGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./agent-graph.gate.mjs'),
      );
      const result = await runAgentGraphGate();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.ok, true);
      assert.equal(result.evidence.issueCount, 0);
    });

    it('returns pass=false when graph has issues', async () => {
      mockReport = {
        ok: false,
        issues: [{ message: 'cycle detected' }],
        graph: [{ name: 'a' }],
        inventory: { summary: { by_tier: { 1: 3 } } },
      };
      const { runAgentGraphGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./agent-graph.gate.mjs'),
      );
      const result = await runAgentGraphGate();
      assert.equal(result.pass, false);
      assert.equal(result.evidence.issueCount, 1);
      assert.ok(result.fixHint.includes('cycle detected'));
    });

    it('handles missing optional fields with null coalescing', async () => {
      mockReport = { ok: true };
      const { runAgentGraphGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./agent-graph.gate.mjs'),
      );
      const result = await runAgentGraphGate();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.issueCount, 0);
      assert.equal(result.evidence.agentCount, 0);
      assert.equal(result.evidence.byTier, null);
      assert.deepEqual(result.evidence.issues, []);
    });

    it('accepts custom workspaceRoot', async () => {
      const { runAgentGraphGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./agent-graph.gate.mjs'),
      );
      await runAgentGraphGate({ workspaceRoot: '/custom/path' });
    });

    it('handles ok=false with missing issues (null coalescing in fixHint)', async () => {
      mockReport = { ok: false };
      const { runAgentGraphGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./agent-graph.gate.mjs'),
      );
      const result = await runAgentGraphGate();
      assert.equal(result.pass, false);
      assert.ok(result.fixHint.includes('Fix agent graph issues'));
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
      mockReport = {
        ok: false,
        issues: [{ message: 'bad ref' }],
        graph: [],
      };
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockReport = {
        ok: false,
        issues: [{ message: 'bad ref' }],
        graph: [],
      };
      const logs = await importGateMain(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
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
          await import('./agent-graph.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});