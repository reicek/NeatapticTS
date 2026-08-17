import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const SCRIPT_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/tier-audit-report.mjs',
);

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  parseArgs: jest.fn(),
  printUsage: jest.fn(),
  repoRoot: REPO_ROOT,
}));

jest.unstable_mockModule('./validate-agent-graph.mjs', () => ({
  collectTierInventory: jest.fn(),
  runValidateAgentGraph: jest.fn(),
}));

const mockInventory = {
  agents: [
    {
      name: '01-planning',
      tier: 1,
      tier_label: 'Orchestrator',
      user_invocable: true,
      delegates_to: ['02-researching'],
      violations: [],
    },
    {
      name: '03-red-testing',
      tier: 1,
      tier_label: 'Orchestrator',
      user_invocable: false,
      delegates_to: [],
      violations: [{ message: 'missing field' }],
    },
  ],
  summary: {
    total: 2,
    by_tier: { 1: 2, 2: 0, 3: 0, 4: 0 },
    violation_count: 1,
    user_invocable_total: 1,
  },
  violations: [],
};

const mockValidation = {
  ok: false,
  issues: [{ path: '.github/agents/03-red-testing.agent.md', message: 'violation' }],
};

const mockInventoryEmpty = {
  agents: [
    {
      name: 'test-agent',
      tier: 1,
      tier_label: 'Test',
      user_invocable: true,
      delegates_to: [],
      violations: [],
    },
  ],
  summary: {
    total: 1,
    by_tier: { 1: 1, 2: 0, 3: 0, 4: 0 },
    violation_count: 0,
    user_invocable_total: 1,
  },
  violations: [],
};

const mockValidationOk = {
  ok: true,
  issues: [],
};

describe('tier-audit-report', () => {
  describe('runTierAuditReport (direct import)', () => {
    let runTierAuditReport;

    beforeAll(async () => {
      const mockUtils = await import('./customization-utils.mjs');
      const mockVAG = await import('./validate-agent-graph.mjs');
      mockUtils.parseArgs.mockReturnValue({ help: false, json: false });
      mockVAG.collectTierInventory.mockResolvedValue(mockInventory);
      mockVAG.runValidateAgentGraph.mockResolvedValue(mockValidation);
      runTierAuditReport = (await import('./tier-audit-report.mjs'))
        .runTierAuditReport;
    });

    it('returns a structured report with name', async () => {
      const report = await runTierAuditReport();
      assert.strictEqual(report.name, 'tier audit report');
    });

    it('returns a boolean ok', async () => {
      const report = await runTierAuditReport();
      assert.ok(typeof report.ok === 'boolean');
    });

    it('returns an array of agents', async () => {
      const report = await runTierAuditReport();
      assert.ok(Array.isArray(report.agents));
    });

    it('returns an array of issues', async () => {
      const report = await runTierAuditReport();
      assert.ok(Array.isArray(report.issues));
    });

    it('returns an array of notes', async () => {
      const report = await runTierAuditReport();
      assert.ok(Array.isArray(report.notes));
    });

    it('returns a summary object', async () => {
      const report = await runTierAuditReport();
      assert.ok(report.summary);
    });

    it('returns a numeric total in summary', async () => {
      const report = await runTierAuditReport();
      assert.ok(typeof report.summary.total === 'number');
    });

    it('includes tier counts in summary', async () => {
      const report = await runTierAuditReport();
      assert.ok(report.summary.by_tier);
    });
  });

  describe('CLI entry (--help)', () => {
    it('prints usage and exits 0', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH, '--help'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      assert.strictEqual(result.status, 0);
      assert.ok(result.stdout.includes('audit report'));
    });
  });

  describe('CLI entry (--json)', () => {
    it('emits JSON report', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH, '--json'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      const payload = JSON.parse(result.stdout.trim());
      assert.strictEqual(payload.name, 'tier audit report');
    });
  });

  describe('CLI entry (markdown)', () => {
    it('emits markdown report', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      assert.ok(result.stdout.includes('# Tier Audit Report'));
    });
  });

  describe('module-level help block', () => {
    let mockUtils;

    beforeEach(async () => {
      jest.resetModules();
      mockUtils = await import('./customization-utils.mjs');
      mockUtils.parseArgs.mockReturnValue({ help: true });
    });

    it('calls printUsage when --help is passed', async () => {
      const origExit = process.exit;
      process.exit = (code) => { throw new Error(`EXIT:${code}`); };
      try {
        await import('./tier-audit-report.mjs');
      } catch {
        // process.exit throws
      }
      process.exit = origExit;
      assert.ok(mockUtils.printUsage.mock.calls.length >= 1);
    });
  });

  describe('main() via import guard (JSON output)', () => {
    let mockUtils;
    let mockVAG;

    beforeEach(async () => {
      jest.resetModules();
      mockUtils = await import('./customization-utils.mjs');
      mockVAG = await import('./validate-agent-graph.mjs');
      mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
      mockVAG.collectTierInventory.mockResolvedValue(mockInventory);
      mockVAG.runValidateAgentGraph.mockResolvedValue(mockValidation);
    });

    it('outputs JSON report to console', async () => {
      const origArgv = process.argv;
      const logs = [];
      const origLog = console.log;
      console.log = (...args) => logs.push(args.join(' '));
      process.argv = ['node', SCRIPT_PATH, '--json'];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      const report = JSON.parse(logs[0]);
      assert.strictEqual(report.name, 'tier audit report');
    });

    it('sets exitCode 1 when validation fails', async () => {
      const origArgv = process.argv;
      const origExitCode = process.exitCode;
      const origLog = console.log;
      console.log = () => {};
      process.argv = ['node', SCRIPT_PATH, '--json'];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      assert.strictEqual(process.exitCode, 1);
      process.exitCode = origExitCode;
    });
  });

  describe('main() via import guard (Markdown output)', () => {
    let mockUtils;
    let mockVAG;

    beforeEach(async () => {
      jest.resetModules();
      mockUtils = await import('./customization-utils.mjs');
      mockVAG = await import('./validate-agent-graph.mjs');
      mockUtils.parseArgs.mockReturnValue({ help: false, json: false });
      mockVAG.collectTierInventory.mockResolvedValue(mockInventory);
      mockVAG.runValidateAgentGraph.mockResolvedValue(mockValidation);
    });

    it('outputs markdown report with delegate text for agents with delegates', async () => {
      const origArgv = process.argv;
      const logs = [];
      const origLog = console.log;
      console.log = (...args) => logs.push(args.join(' '));
      process.argv = ['node', SCRIPT_PATH];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      assert.ok(logs[0].includes('02-researching'));
    });

    it('outputs markdown report with em-dash for agents without delegates', async () => {
      const origArgv = process.argv;
      const logs = [];
      const origLog = console.log;
      console.log = (...args) => logs.push(args.join(' '));
      process.argv = ['node', SCRIPT_PATH];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      assert.ok(logs[0].includes('03-red-testing | 1 (Orchestrator) | false | — | missing field'));
    });

    it('outputs markdown report with violation message joined by br', async () => {
      const origArgv = process.argv;
      const logs = [];
      const origLog = console.log;
      console.log = (...args) => logs.push(args.join(' '));
      process.argv = ['node', SCRIPT_PATH];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      assert.ok(logs[0].includes('missing field'));
    });

    it('outputs markdown report with user-invocable true for first agent', async () => {
      const origArgv = process.argv;
      const logs = [];
      const origLog = console.log;
      console.log = (...args) => logs.push(args.join(' '));
      process.argv = ['node', SCRIPT_PATH];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      assert.ok(logs[0].includes('01-planning | 1 (Orchestrator) | true'));
    });

    it('outputs markdown report with issues section listing violations', async () => {
      const origArgv = process.argv;
      const logs = [];
      const origLog = console.log;
      console.log = (...args) => logs.push(args.join(' '));
      process.argv = ['node', SCRIPT_PATH];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      assert.ok(logs[0].includes('03-red-testing.agent.md: violation'));
    });
  });

  describe('main() via import guard (Markdown with no issues)', () => {
    let mockUtils;
    let mockVAG;

    beforeEach(async () => {
      jest.resetModules();
      mockUtils = await import('./customization-utils.mjs');
      mockVAG = await import('./validate-agent-graph.mjs');
      mockUtils.parseArgs.mockReturnValue({ help: false, json: false });
      mockVAG.collectTierInventory.mockResolvedValue(mockInventoryEmpty);
      mockVAG.runValidateAgentGraph.mockResolvedValue(mockValidationOk);
    });

    it('outputs markdown report with None when no issues', async () => {
      const origArgv = process.argv;
      const logs = [];
      const origLog = console.log;
      console.log = (...args) => logs.push(args.join(' '));
      process.argv = ['node', SCRIPT_PATH];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      assert.ok(logs[0].includes('- None'));
    });

    it('sets exitCode 0 when validation passes', async () => {
      const origArgv = process.argv;
      const origExitCode = process.exitCode;
      const origLog = console.log;
      console.log = () => {};
      process.argv = ['node', SCRIPT_PATH];
      await import('./tier-audit-report.mjs');
      process.argv = origArgv;
      console.log = origLog;
      assert.strictEqual(process.exitCode, 0);
      process.exitCode = origExitCode;
    });
  });
});