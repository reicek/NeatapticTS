/**
 * @module tier-enforcement-gate.test
 * @description Coverage tests for tier-enforcement-gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/tier-enforcement-gate.mjs',
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
      await import('./tier-enforcement-gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('tier-enforcement-gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockReport = {
      ok: true,
      issues: [],
      inventory: { summary: { by_tier: { 1: 5 }, user_invocable_total: 3 } },
    };
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('runTierEnforcementGate', () => {
    it('returns pass=true when no tier issues', async () => {
      const { runTierEnforcementGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./tier-enforcement-gate.mjs'),
      );
      const result = await runTierEnforcementGate();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.issueCount, 0);
    });

    it('returns pass=false when tier issues exist', async () => {
      mockReport = {
        ok: false,
        issues: [
          { message: 'tier violation: agent X has invalid tier' },
          { message: 'user-invocable rule broken' },
          { message: 'unrelated issue' },
        ],
        inventory: { summary: { by_tier: { 1: 3 }, user_invocable_total: 2 } },
      };
      const { runTierEnforcementGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./tier-enforcement-gate.mjs'),
      );
      const result = await runTierEnforcementGate();
      assert.equal(result.pass, false);
      assert.equal(result.evidence.issueCount, 2);
      assert.ok(result.fixHint.includes('tier violation'));
    });

    it('handles missing optional fields with null coalescing', async () => {
      mockReport = { ok: true };
      const { runTierEnforcementGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./tier-enforcement-gate.mjs'),
      );
      const result = await runTierEnforcementGate();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.byTier, null);
      assert.equal(result.evidence.userInvocableTotal, null);
      assert.deepEqual(result.evidence.issues, []);
    });

    it('filters only tier/user-invocable issues', async () => {
      mockReport = {
        ok: false,
        issues: [{ message: 'some other error' }],
      };
      const { runTierEnforcementGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./tier-enforcement-gate.mjs'),
      );
      const result = await runTierEnforcementGate();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.issueCount, 0);
    });

    it('accepts custom workspaceRoot', async () => {
      const { runTierEnforcementGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./tier-enforcement-gate.mjs'),
      );
      await runTierEnforcementGate({ workspaceRoot: '/custom' });
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
        issues: [{ message: 'tier violation: bad' }],
      };
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockReport = {
        ok: false,
        issues: [{ message: 'tier violation: bad' }],
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
          await import('./tier-enforcement-gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});
