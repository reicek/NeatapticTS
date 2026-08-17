import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const SCRIPT_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/tier-inventory.mjs',
);

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  parseArgs: jest.fn(),
  printUsage: jest.fn(),
  writeReport: jest.fn(),
  repoRoot: REPO_ROOT,
}));

jest.unstable_mockModule('./validate-agent-graph.mjs', () => ({
  collectTierInventory: jest.fn(),
}));

const mockInventory = {
  agents: [
    {
      name: '01-planning',
      tier: 1,
      file: '.github/agents/01-planning.agent.md',
      tier_label: 'Numbered SDLC Orchestrators',
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

describe('tier-inventory', () => {
  describe('runTierInventory (direct import)', () => {
    let runTierInventory;

    beforeAll(async () => {
      const mockUtils = await import('./customization-utils.mjs');
      const mockVAG = await import('./validate-agent-graph.mjs');
      mockUtils.parseArgs.mockReturnValue({ help: false, json: false });
      mockVAG.collectTierInventory.mockResolvedValue(mockInventory);
      runTierInventory = (await import('./tier-inventory.mjs'))
        .runTierInventory;
    });

    it('returns a structured inventory with name', async () => {
      const inv = await runTierInventory();
      assert.strictEqual(inv.name, 'agent tier inventory');
    });

    it('returns an array of agents', async () => {
      const inv = await runTierInventory();
      assert.ok(Array.isArray(inv.agents));
    });

    it('returns a summary object', async () => {
      const inv = await runTierInventory();
      assert.ok(inv.summary);
    });

    it('returns a numeric total in summary', async () => {
      const inv = await runTierInventory();
      assert.ok(typeof inv.summary.total === 'number');
    });

    it('returns by_tier in summary', async () => {
      const inv = await runTierInventory();
      assert.ok(inv.summary.by_tier);
    });

    it('each agent has a name', async () => {
      const inv = await runTierInventory();
      assert.ok(inv.agents[0].name);
    });

    it('each agent has a numeric tier', async () => {
      const inv = await runTierInventory();
      assert.ok(typeof inv.agents[0].tier === 'number');
    });

    it('each agent has a file path', async () => {
      const inv = await runTierInventory();
      assert.ok(inv.agents[0].file);
    });
  });

  describe('CLI entry (--help)', () => {
    it('prints usage and exits 0', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH, '--help'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      assert.strictEqual(result.status, 0);
      assert.ok(result.stdout.includes('Inventory'));
    });
  });

  describe('CLI entry (--json)', () => {
    it('emits JSON inventory', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH, '--json'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      const payload = JSON.parse(result.stdout.trim());
      assert.strictEqual(payload.name, 'agent tier inventory');
    });
  });

  describe('CLI entry (markdown)', () => {
    it('emits summary text', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      assert.ok(result.stdout.includes('agent tier inventory'));
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
        await import('./tier-inventory.mjs');
      } catch {
        // process.exit throws
      }
      process.exit = origExit;
      assert.ok(mockUtils.printUsage.mock.calls.length >= 1);
    });
  });

  describe('main() via import guard', () => {
    let mockUtils;
    let mockVAG;

    beforeEach(async () => {
      jest.resetModules();
      mockUtils = await import('./customization-utils.mjs');
      mockVAG = await import('./validate-agent-graph.mjs');
      mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
      mockVAG.collectTierInventory.mockResolvedValue(mockInventory);
    });

    it('calls writeReport when run as main script', async () => {
      const origArgv = process.argv;
      process.argv = ['node', SCRIPT_PATH, '--json'];
      await import('./tier-inventory.mjs');
      process.argv = origArgv;
      assert.ok(mockUtils.writeReport.mock.calls.length >= 1);
    });

    it('sets exitCode 0 when report is ok', async () => {
      const origArgv = process.argv;
      const origExitCode = process.exitCode;
      process.argv = ['node', SCRIPT_PATH, '--json'];
      await import('./tier-inventory.mjs');
      process.argv = origArgv;
      assert.strictEqual(process.exitCode, 0);
      process.exitCode = origExitCode;
    });

    it('sets exitCode 1 when report has violations', async () => {
      mockVAG.collectTierInventory.mockResolvedValue({
        ...mockInventory,
        violations: [{ message: 'error' }],
        summary: { ...mockInventory.summary, violation_count: 1 },
      });
      const origArgv = process.argv;
      const origExitCode = process.exitCode;
      process.argv = ['node', SCRIPT_PATH, '--json'];
      await import('./tier-inventory.mjs');
      process.argv = origArgv;
      assert.strictEqual(process.exitCode, 1);
      process.exitCode = origExitCode;
    });
  });
});