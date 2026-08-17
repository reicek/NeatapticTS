/**
 * @module agent-quality.gate.test
 * @description Coverage tests for agent-quality.gate.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/agent-quality.gate.mjs',
);

let mockReport;

jest.unstable_mockModule('../validate-agent-quality.mjs', () => ({
  runValidateAgentQuality: async () => mockReport,
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
      await import('./agent-quality.gate.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

describe('agent-quality gate', () => {
  let originalExitCode;

  beforeEach(() => {
    mockReport = {
      ok: true,
      agents: [{ path: 'a', name: 'a', tier: 1, counts: { errors: 0, warnings: 0 } }],
      issues: [],
      counts: { errors: 0, warnings: 0 },
      contractDocument: 'doc',
    };
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('runAgentQualityGate', () => {
    it('returns pass=true when all agents valid', async () => {
      const { runAgentQualityGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./agent-quality.gate.mjs'),
      );
      const result = await runAgentQualityGate();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.issueCount, 0);
    });

    it('returns pass=false when agents have errors', async () => {
      mockReport = {
        ok: false,
        agents: [
          { path: 'a', name: 'a', tier: 1, counts: { errors: 2, warnings: 1 } },
          { path: 'b', name: 'b', tier: 2, counts: { errors: 0, warnings: 0 } },
        ],
        issues: [{ message: 'missing section' }],
        counts: { errors: 2, warnings: 1 },
        contractDocument: 'doc',
      };
      const { runAgentQualityGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./agent-quality.gate.mjs'),
      );
      const result = await runAgentQualityGate();
      assert.equal(result.pass, false);
      assert.equal(result.evidence.failingAgents.length, 1);
      assert.equal(result.evidence.failingAgents[0].errors, 2);
    });

    it('returns pass=true when agents have only warnings', async () => {
      mockReport = {
        ok: true,
        agents: [{ path: 'a', name: 'a', tier: 1, counts: { errors: 0, warnings: 5 } }],
        issues: [],
        counts: { errors: 0, warnings: 5 },
        contractDocument: 'doc',
      };
      const { runAgentQualityGate } = await withArgv(
        [process.execPath, 'dummy'],
        () => import('./agent-quality.gate.mjs'),
      );
      const result = await runAgentQualityGate();
      assert.equal(result.pass, true);
      assert.equal(result.evidence.failingAgents.length, 0);
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
        agents: [{ path: 'a', name: 'a', tier: 1, counts: { errors: 1, warnings: 0 } }],
        issues: [{ message: 'bad' }],
        counts: { errors: 1, warnings: 0 },
        contractDocument: 'doc',
      };
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockReport = {
        ok: false,
        agents: [{ path: 'a', name: 'a', tier: 1, counts: { errors: 1, warnings: 0 } }],
        issues: [{ message: 'bad' }],
        counts: { errors: 1, warnings: 0 },
        contractDocument: 'doc',
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
          await import('./agent-quality.gate.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});