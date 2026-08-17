/**
 * @module specialist-review.gate.test
 * @description Coverage tests for specialist-review.gate.mjs (top-level await, no exports).
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/specialist-review.gate.mjs',
);

// --- Mock state -----------------------------------------------------------
let mockReadFileSyncFn;
let mockReaddirSyncFn;
let mockReadFileCalls;

jest.unstable_mockModule('node:fs', () => ({
  readFileSync: (p, opts) => {
    const key = String(p);
    mockReadFileCalls[key] = (mockReadFileCalls[key] || 0) + 1;
    return mockReadFileSyncFn(p, opts, mockReadFileCalls[key]);
  },
  readdirSync: (p) => mockReaddirSyncFn(p),
}));

jest.unstable_mockModule('../customization-utils.mjs', () => ({
  parseArgs: (argv) => {
    const planArg = argv.find((a) => a.startsWith('--plan='));
    return {
      json: argv.includes('--json'),
      help: argv.includes('--help'),
      plan: planArg ? planArg.slice('--plan='.length) : null,
    };
  },
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
    await import('./specialist-review.gate.mjs');
    await new Promise((r) => setTimeout(r, 200));
    process.argv = originalArgv;
  } finally {
    console.log = originalLog;
  }
  return logs;
}

// Content helpers
const WIP_WITH_SPECIALIST =
  '[WIP]\n## VALIDATION_EVIDENCE\nimplementation-pattern-scout: APPROVE\n';
const WIP_WITH_TRIVIAL =
  '[WIP]\n## VALIDATION_EVIDENCE\nseverity: TRIVIAL\n';
const WIP_NO_EVIDENCE =
  '[WIP]\nSome plan content without validation evidence section.\n';
const WIP_NO_MARKERS =
  '[WIP]\n## VALIDATION_EVIDENCE\nSome evidence but no specialist markers.\n';
const NO_WIP = 'Completed plan with no WIP markers.\n';

// --- Tests ----------------------------------------------------------------
describe('specialist-review gate', () => {
  let originalExitCode;

  beforeEach(() => {
    originalExitCode = process.exitCode;
    process.exitCode = undefined;
    jest.resetModules();
    mockReadFileCalls = {};
    mockReadFileSyncFn = () => '';
    mockReaddirSyncFn = () => [];
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('no WIP plans', () => {
    it('returns pass=true when no plans found (empty directory)', async () => {
      mockReaddirSyncFn = () => [];
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 0);
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=true when plans do not contain [WIP]', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'plan2.plans.md'];
      mockReadFileSyncFn = () => NO_WIP;
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 0);
      assert.equal(process.exitCode, 0);
    });

    it('emits PASS text without --json when no WIP plans', async () => {
      mockReaddirSyncFn = () => [];
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.ok(logs.some((l) => l.includes('no [WIP] plans')));
      assert.equal(process.exitCode, 0);
    });

    it('filters readFileSync errors in wipPlans filter', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md'];
      mockReadFileSyncFn = () => {
        throw new Error('ENOENT');
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 0);
    });

    it('filters non-.plans.md files from readdirSync', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'README.md', 'data.txt'];
      mockReadFileSyncFn = () => NO_WIP;
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 0);
    });
  });

  describe('with WIP plans via --plan', () => {
    it('returns pass=true when specialist review evidence found', async () => {
      mockReadFileSyncFn = () => WIP_WITH_SPECIALIST;
      const logs = await importGate([
        '--json',
        '--plan=plans/test.plans.md',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 1);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(parsed.evidence.planResults[0].trivialExempt, false);
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=true when TRIVIAL exemption marker found', async () => {
      mockReadFileSyncFn = () => WIP_WITH_TRIVIAL;
      const logs = await importGate([
        '--json',
        '--plan=plans/test.plans.md',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.ok(parsed.evidence.planResults[0].found);
      assert.equal(parsed.evidence.planResults[0].trivialExempt, true);
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=false when no VALIDATION_EVIDENCE section', async () => {
      mockReadFileSyncFn = () => WIP_NO_EVIDENCE;
      const logs = await importGate([
        '--json',
        '--plan=plans/test.plans.md',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no VALIDATION_EVIDENCE section found',
      );
      assert.equal(process.exitCode, 1);
    });

    it('returns pass=false when no specialist review markers found', async () => {
      mockReadFileSyncFn = () => WIP_NO_MARKERS;
      const logs = await importGate([
        '--json',
        '--plan=plans/test.plans.md',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no specialist review evidence found',
      );
      assert.equal(process.exitCode, 1);
    });

    it('returns found=false when file not readable on second call', async () => {
      mockReadFileSyncFn = (p, opts, callNum) => {
        if (callNum === 1) return '[WIP]\n'; // filter call
        throw new Error('ENOENT'); // checkPlan call
      };
      const logs = await importGate([
        '--json',
        '--plan=plans/test.plans.md',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'file not readable',
      );
    });

    it('returns found=false when content no longer has [WIP] on second call', async () => {
      mockReadFileSyncFn = (p, opts, callNum) => {
        if (callNum === 1) return '[WIP]\n'; // filter call
        return 'No WIP here.\n'; // checkPlan call
      };
      const logs = await importGate([
        '--json',
        '--plan=plans/test.plans.md',
      ]);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(
        parsed.evidence.planResults[0].reason,
        'no [WIP] sections',
      );
    });
  });

  describe('with WIP plans via directory scan', () => {
    it('returns pass=true when all WIP plans have specialist review', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'plan2.plans.md'];
      mockReadFileSyncFn = () => WIP_WITH_SPECIALIST;
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
      assert.equal(parsed.evidence.wipPlans, 2);
      assert.equal(process.exitCode, 0);
    });

    it('returns pass=false when some WIP plans lack specialist review', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'plan2.plans.md'];
      let callCount = 0;
      mockReadFileSyncFn = () => {
        callCount++;
        // Calls 1-2: filter (both WIP). Calls 3-4: checkPlan.
        if (callCount <= 2) return WIP_WITH_SPECIALIST;
        return WIP_NO_EVIDENCE;
      };
      const logs = await importGate(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
      assert.equal(process.exitCode, 1);
    });
  });

  describe('text output', () => {
    it('emits PASS text when all WIP plans have specialist review', async () => {
      mockReadFileSyncFn = () => WIP_WITH_SPECIALIST;
      const logs = await importGate(['--plan=plans/test.plans.md']);
      assert.ok(logs.some((l) => l.includes('PASS')));
      assert.ok(logs.some((l) => l.includes('specialist-review gate')));
      assert.equal(process.exitCode, 0);
    });

    it('emits FAIL text with plan listing and fixHint when pass=false', async () => {
      mockReadFileSyncFn = () => WIP_NO_EVIDENCE;
      const logs = await importGate(['--plan=plans/test.plans.md']);
      assert.ok(logs.some((l) => l.includes('FAIL')));
      assert.ok(logs.some((l) => l.includes('test.plans.md')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
      assert.equal(process.exitCode, 1);
    });

    it('covers if(!r.found) false branch with mixed found results', async () => {
      mockReaddirSyncFn = () => ['plan1.plans.md', 'plan2.plans.md'];
      mockReadFileSyncFn = (p, opts, callNum) => {
        if (callNum === 1) return WIP_WITH_SPECIALIST;
        const key = String(p);
        if (key.includes('plan1')) return WIP_WITH_SPECIALIST;
        return WIP_NO_EVIDENCE;
      };
      const logs = await importGate([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
    });
  });
});