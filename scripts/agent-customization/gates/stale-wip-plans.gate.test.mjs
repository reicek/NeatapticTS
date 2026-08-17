/**
 * @module stale-wip-plans.gate.test
 * @description Coverage tests for stale-wip-plans.gate.mjs (top-level await, no exports).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/stale-wip-plans.gate.mjs',
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
    await import('./stale-wip-plans.gate.mjs');
    await new Promise((r) => setTimeout(r, 200));
    process.argv = originalArgv;
  } finally {
    console.log = originalLog;
  }
  return logs;
}

// --- Tests ----------------------------------------------------------------
describe('stale-wip-plans gate', () => {
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

  describe('runStaleWipGate', () => {
    it('returns pass=false when plans/ directory cannot be read', async () => {
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
      mockReaddirFn = async () => [];
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.plansChecked, 0);
      assert.equal(parsed.evidence.plansFound, 0);
    });

    it('skips plans that are not WIP (detectStalePlan returns null)', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return '## Implementation phases\n### Phase 1 [DONE]\n';
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'DONE';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.stalePlans.length, 0);
      assert.equal(parsed.evidence.plansChecked, 1);
    });

    it('skips WIP plans without "## Implementation phases" section', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) return 'No impl section here.\n';
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.stalePlans.length, 0);
    });

    it('skips WIP plans with no phase markers in impl section', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return '## Implementation phases\nSome text without markers.\n';
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.stalePlans.length, 0);
    });

    it('detects stale plan when all phase markers are DONE', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return (
            '## Implementation phases\n' +
            '### Phase 1: Setup [DONE]\n' +
            '### Phase 2: Build [DONE]\n' +
            "status: '[DONE]'\n"
          );
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.stalePlans.length, 1);
      assert.equal(parsed.evidence.stalePlans[0].plan, 'plans/plan1.md');
      assert.equal(parsed.evidence.stalePlans[0].phaseCount, 3);
    });

    it('does not flag plan as stale when some markers are WIP', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return (
            '## Implementation phases\n' +
            '### Phase 1: Setup [DONE]\n' +
            '### Phase 2: Build [WIP]\n'
          );
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.stalePlans.length, 0);
    });

    it('does not flag plan as stale when some markers are PLANNED', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return (
            '## Implementation phases\n' +
            '### Phase 1: Setup [DONE]\n' +
            '### Phase 2: Build [PLANNED]\n'
          );
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.stalePlans.length, 0);
    });

    it('skips unreadable plan files', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) throw new Error('ENOENT');
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.plansChecked, 0);
      assert.equal(parsed.evidence.plansFound, 1);
    });

    it('filters out non-plan entries (dirs, .logs.md, README, Roadmap, non-.md)', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return (
            '## Implementation phases\n### Phase 1 [DONE]\n'
          );
        }
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
      mockExtractStatusFn = () => 'DONE';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.evidence.plansFound, 1);
      assert.equal(parsed.evidence.plansChecked, 1);
    });

    it('handles YAML status line markers in collectPhaseMarkers', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return (
            '## Implementation phases\n' +
            "status: '[DONE]'\n" +
            "status: '[WIP]'\n" +
            "someOther: 'value'\n"
          );
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      // Has DONE and WIP markers → not stale (hasOpenWork is true)
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.stalePlans.length, 0);
    });

    it('handles heading lines without status markers', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return (
            '## Implementation phases\n' +
            '### Phase 1: Setup (no status)\n' +
            '### Phase 2: Build [DONE]\n'
          );
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      // Only one marker [DONE] → stale
      assert.equal(parsed.pass, false);
      assert.equal(parsed.evidence.stalePlans.length, 1);
      assert.equal(parsed.evidence.stalePlans[0].phaseCount, 1);
    });
  });

  describe('output format', () => {
    it('emits JSON with --json when pass=true', async () => {
      mockReaddirFn = async () => [];
      const logs = await importGate(['--json']);
      assert.equal(logs.length, 1);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(process.exitCode, 0);
    });

    it('emits PASS text without --json when pass=true', async () => {
      mockReaddirFn = async () => [];
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.ok(logs.some((l) => l.includes('stale-wip-plans gate')));
      assert.equal(process.exitCode, 0);
    });

    it('emits FAIL text with fixHint and stale plan listing when pass=false', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return (
            '## Implementation phases\n### Phase 1 [DONE]\n'
          );
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.ok(logs.some((l) => l.includes(' -')));
      assert.ok(logs.some((l) => l.includes('plans/plan1.md')));
      assert.ok(logs.some((l) => l.includes('all [DONE]')));
      assert.equal(process.exitCode, 1);
    });

    it('emits JSON with --json when pass=false', async () => {
      mockReadFileFn = async (filePath) => {
        if (filePath.endsWith('plan1.md')) {
          return (
            '## Implementation phases\n### Phase 1 [DONE]\n'
          );
        }
        return '';
      };
      mockReaddirFn = async () => [
        { name: 'plan1.md', isFile: () => true },
      ];
      mockExtractStatusFn = () => 'WIP';
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });
  });
});