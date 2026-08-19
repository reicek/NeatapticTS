/**
 * @module plan-sync.gate.test
 * @description Coverage tests for plan-sync.gate.mjs (top-level await, no exports).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/plan-sync.gate.mjs',
);

// --- Mock state -----------------------------------------------------------
let mockReadFileFn;
let mockReaddirFn;
let mockExtractStatusFn;

jest.unstable_mockModule('node:fs/promises', () => ({
  readdir: async (dirPath, opts) => mockReaddirFn(dirPath, opts),
  readFile: async (filePath, opts) => mockReadFileFn(filePath, opts),
}));

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) => ({
    json: argv.includes('--json'),
    help: argv.includes('--help'),
  }),
  repoRoot: REPO_ROOT,
  extractStatus: (text) => mockExtractStatusFn(text),
}));

// --- Helper ---------------------------------------------------------------
async function importGate(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    const originalArgv = process.argv;
    process.argv = [process.execPath, GATE_PATH, ...argv];
    await import('./plan-sync.gate.mjs');
    await new Promise((r) => setTimeout(r, 200));
    process.argv = originalArgv;
  } finally {
    console.log = originalLog;
  }
  return logs;
}

// --- Tests ----------------------------------------------------------------
describe('plan-sync gate', () => {
  let originalExitCode;

  beforeEach(() => {
    originalExitCode = process.exitCode;
    process.exitCode = undefined;
    jest.resetModules();
    mockReadFileFn = async () => '';
    mockReaddirFn = async () => [];
    mockExtractStatusFn = () => 'DONE';
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('runPlanSyncGate', () => {
    it('returns pass=false when README/Roadmap cannot be read', async () => {
      mockReadFileFn = async () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.readmeFound, false);
      assert.equal(parsed.evidence.roadmapFound, false);
      assert.ok(parsed.evidence.error);
    });

    it('returns pass=false when plans/ directory cannot be read', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README content';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap content';
        return '';
      };
      mockReaddirFn = async () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.ok(parsed.evidence.error);
      assert.equal(parsed.evidence.scannedDir, 'plans/');
    });

    it('returns pass=true when no plan files exist', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap';
        return '';
      };
      mockReaddirFn = async () => [];
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.plansChecked, 0);
    });

    it('returns pass=true when no WIP plans exist', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap';
        if (filePath.endsWith('plan1.md')) return 'Status: [DONE]';
        return '';
      };
      mockReaddirFn = async () => [{ name: 'plan1.md', isFile: () => true }];
      mockExtractStatusFn = () => 'DONE';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans.length, 0);
      assert.equal(parsed.evidence.plansChecked, 1);
    });

    it('returns pass=false when WIP plan is missing from README', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README without plan name';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap with plan1.md';
        if (filePath.endsWith('plan1.md')) return 'Status: [WIP]';
        return '';
      };
      mockReaddirFn = async () => [{ name: 'plan1.md', isFile: () => true }];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.missingFromReadme.length, 1);
      assert.equal(parsed.evidence.missingFromRoadmap.length, 0);
    });

    it('returns pass=false when WIP plan is missing from Roadmap', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README with plan1.md';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap without plan name';
        if (filePath.endsWith('plan1.md')) return 'Status: [WIP]';
        return '';
      };
      mockReaddirFn = async () => [{ name: 'plan1.md', isFile: () => true }];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.missingFromReadme.length, 0);
      assert.equal(parsed.evidence.missingFromRoadmap.length, 1);
    });

    it('returns pass=true when WIP plan is registered in both', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README with plan1.md';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap with plan1.md';
        if (filePath.endsWith('plan1.md')) return 'Status: [WIP]';
        return '';
      };
      mockReaddirFn = async () => [{ name: 'plan1.md', isFile: () => true }];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans.length, 1);
    });

    it('skips unreadable plan files', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap';
        if (filePath.endsWith('plan1.md')) throw new Error('ENOENT');
        return '';
      };
      mockReaddirFn = async () => [{ name: 'plan1.md', isFile: () => true }];
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans.length, 0);
      assert.equal(parsed.evidence.plansChecked, 1);
    });

    it('filters out non-plan entries (dirs, .logs.md, README, Roadmap, non-.md)', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README with plan1.md';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap with plan1.md';
        if (filePath.endsWith('plan1.md')) return 'Status: [WIP]';
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
        { name: 'subdir', isFile: () => false },
        { name: 'plan1.logs.md', isFile: () => true },
        { name: 'README.md', isFile: () => true },
        { name: 'Roadmap.md', isFile: () => true },
        { name: 'data.json', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.evidence.plansChecked, 1);
      assert.equal(parsed.evidence.wipPlans.length, 1);
    });
  });

  describe('output format', () => {
    it('emits JSON with --json when pass=true', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap';
        return '';
      };
      mockReaddirFn = async () => [];
      const logs = await importGate(['--json']);
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(process.exitCode, 0);
    });

    it('emits PASS text without --json when pass=true', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('README.md')) return 'README';
        if (filePath.endsWith('Roadmap.md')) return 'Roadmap';
        return '';
      };
      mockReaddirFn = async () => [];
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.ok(logs.some((l) => l.includes('plan-sync gate')));
      assert.equal(process.exitCode, 0);
    });

    it('emits FAIL text with fixHint when pass=false', async () => {
      mockReadFileFn = async () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockReadFileFn = async () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });
  });
});
